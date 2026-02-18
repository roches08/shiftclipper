import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rq import get_current_job

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:  # pragma: no cover
    DeepSort = None

from common.config import normalize_setup, resolve_device

log = logging.getLogger("worker")
DEBUG_MODE = os.getenv("WORKER_DEBUG", "0") == "1"
HEARTBEAT_SECONDS = float(os.getenv("WORKER_HEARTBEAT_SECONDS", "5"))
STALL_TIMEOUT_S = float(os.getenv("WORKER_STALL_TIMEOUT_SECONDS", "120"))

BASE_DIR = Path(__file__).resolve().parents[1]
JOBS_DIR = Path(os.getenv("JOBS_DIR", str(BASE_DIR / "data" / "jobs"))).resolve()
_EASYOCR_WARMED = False


def _is_cuda_fork_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return "cannot re-initialize cuda in forked subprocess" in msg


def _build_ocr_reader(device: str):
    if easyocr is None:
        return None, False
    use_gpu = device.startswith("cuda")
    return easyocr.Reader(["en"], gpu=use_gpu), use_gpu


def warm_easyocr_models(device: str = "cpu") -> None:
    global _EASYOCR_WARMED
    if _EASYOCR_WARMED or easyocr is None:
        return
    _, warm_gpu = _build_ocr_reader(device)
    _EASYOCR_WARMED = True
    log.info("EasyOCR models ready (gpu=%s)", warm_gpu)


@dataclass
class TrackingParams:
    detect_stride: int = 2
    yolo_imgsz: int = 512
    ocr_every_n: int = 8
    ocr_max_crops_per_frame: int = 1
    ocr_disable: bool = False
    yolo_batch: int = 4
    tracker_type: str = "bytetrack"
    ocr_min_conf: float = 0.22
    lock_seconds_after_confirm: float = 4.0
    gap_merge_seconds: float = 1.5
    lost_timeout: float = 4.0
    reacquire_window_seconds: float = 8.0
    reacquire_score_lock_threshold: float = 0.30
    min_track_seconds: float = 0.75
    min_clip_seconds: float = 1.0
    post_roll: float = 2.0
    score_lock_threshold: float = 0.55
    score_unlock_threshold: float = 0.33
    seed_lock_seconds: float = 8.0
    seed_iou_min: float = 0.12
    seed_dist_max: float = 0.16
    seed_bonus: float = 0.80
    seed_window_s: float = 3.0
    color_weight: float = 0.35
    motion_weight: float = 0.30
    ocr_weight: float = 0.35
    identity_weight: float = 0.5
    color_tolerance: int = 26
    ocr_confirm_m: int = 2
    ocr_confirm_k: int = 5
    ocr_veto_conf: float = 0.92
    ocr_veto_seconds: float = 1.0
    bench_zone_ratio: float = 0.8
    closeup_bbox_area_ratio: float = 0.18
    allow_unconfirmed_clips: bool = False
    allow_seed_clips: bool = True
    tracking_mode: str = "clip"
    verify_mode: bool = False
    debug_overlay: bool = False
    debug_timeline: bool = True
    use_rink_mask: bool = True
    use_bench_mask: bool = True
    use_reid: bool = True
    loss_timeout_sec: float = 1.5
    reacquire_max_sec: float = 2.0
    reacquire_confirm_frames: int = 5
    reid_sim_threshold: float = 0.35
    max_clip_len_sec: float = 90.0
    allow_bench_reacquire: bool = False
    edge_margin_px: float = 2.0
    reid_every_n_frames: int = 4
    reid_max_candidates: int = 5
    reid_alpha: float = 0.7
    reid_min_px: int = 24
    reid_sharpness_threshold: float = 15.0


class ClipEndReason(str, Enum):
    LOST_TIMEOUT = "lost_timeout"
    BENCH_ENTER = "bench_enter"
    PERIOD_END = "period_end"
    VIDEO_END = "video_end"
    MANUAL_STOP = "manual_stop"
    MAX_LEN_GUARD = "max_len_guard"


def _ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _hex_to_hsv(hex_color: str) -> Tuple[int, int, int]:
    c = (hex_color or "").strip().lstrip("#")
    if len(c) != 6:
        return (60, 80, 80)
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    arr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0][0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def _color_score(frame: np.ndarray, box: Tuple[int, int, int, int], jersey_hsv: Tuple[int, int, int], tol: int) -> float:
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = jersey_hsv
    lower = np.array([max(0, h - tol), max(20, s - 80), max(20, v - 80)], dtype=np.uint8)
    upper = np.array([min(179, h + tol), min(255, s + 80), min(255, v + 80)], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(mask.mean() / 255.0)


def _mean_hsv(frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)


def _hsv_distance(a: Optional[np.ndarray], b: Tuple[int, int, int]) -> float:
    if a is None:
        return 999.0
    ah, as_, av = float(a[0]), float(a[1]), float(a[2])
    bh, bs, bv = float(b[0]), float(b[1]), float(b[2])
    dh = min(abs(ah - bh), 180.0 - abs(ah - bh)) / 180.0
    ds = abs(as_ - bs) / 255.0
    dv = abs(av - bv) / 255.0
    return float(dh + 0.5 * ds + 0.25 * dv)


def _parse_digits(txt: str) -> Optional[str]:
    d = re.sub(r"\D+", "", txt or "")
    return d or None


def _clip(box, w, h):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1)); x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(1, min(h, y2))
    return (x1, y1, x2, y2)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(1, ua)


def _expand_box(box: Tuple[int, int, int, int], scale: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, (x2 - x1) * scale)
    bh = max(1.0, (y2 - y1) * scale)
    return _clip((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), w, h)




def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(1e-9, (yj - yi)) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _bbox_feet_point(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def _normalize_polygon(poly: List[List[float]], width: int, height: int, normalized: bool) -> List[Tuple[float, float]]:
    out = []
    for pt in poly or []:
        if len(pt) != 2:
            continue
        px, py = float(pt[0]), float(pt[1])
        if normalized:
            px *= width
            py *= height
        out.append((px, py))
    return out


def _poly_contains_with_margin(point: Tuple[float, float], polygon: List[Tuple[float, float]], margin: float) -> bool:
    if _point_in_polygon(point, polygon):
        return True
    x, y = point
    for px, py in polygon:
        if abs(px - x) <= margin and abs(py - y) <= margin:
            return True
    return False


def _crop_sharpness(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-6 or bn < 1e-6:
        return -1.0
    return float(np.dot(a, b) / (an * bn))


def _embed_crop(crop: np.ndarray) -> Optional[np.ndarray]:
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).astype(np.float32).flatten()
    n = np.linalg.norm(hist)
    if n < 1e-6:
        return None
    return hist / n


def _compute_clip_end_for_loss(last_seen_time: float, loss_timeout: float, video_end_time: float) -> float:
    return min(video_end_time, max(0.0, last_seen_time + max(0.0, loss_timeout)))


def _append_segment(segments: List[Tuple[float, float, str]], start_time: float, end_time: float, reason: ClipEndReason) -> None:
    if end_time > start_time:
        segments.append((float(start_time), float(end_time), reason.value))
def _merge_segments(
    segments: List[Tuple[float, float, str]],
    *,
    min_clip_seconds: float,
    gap_merge_seconds: float,
    tracking_mode: str,
    reacquire_window_seconds: float,
) -> List[Tuple[float, float, str]]:
    merged: List[Tuple[float, float, str]] = []
    raw_segments = sorted(segments, key=lambda x: x[0])

    if tracking_mode != "shift":
        for a, b, reason in raw_segments:
            if (b - a) < min_clip_seconds:
                continue
            if merged and a - merged[-1][1] <= gap_merge_seconds:
                merged_reason = "seed_clip" if (reason == "seed_clip" or merged[-1][2] == "seed_clip") else "merged"
                merged[-1] = (merged[-1][0], max(merged[-1][1], b), merged_reason)
            else:
                merged.append((a, b, reason))
        return merged

    previous = None
    merged_last = False
    for a, b, reason in raw_segments:
        if (b - a) < min_clip_seconds:
            continue

        if previous is None:
            previous = (a, b, reason)
            merged_last = False
            continue

        prev_a, prev_b, prev_reason = previous
        gap = a - prev_b
        can_merge = (
            not merged_last
            and gap <= gap_merge_seconds
            and gap <= reacquire_window_seconds
            and prev_reason == "unlock"
            and reason == "unlock"
        )
        if can_merge:
            merged_reason = "merged"
            merged.append((prev_a, max(prev_b, b), merged_reason))
            previous = None
            merged_last = True
            continue

        merged.append(previous)
        previous = (a, b, reason)
        merged_last = False

    if previous is not None:
        merged.append(previous)

    return merged


class FFmpegError(RuntimeError):
    def __init__(self, cmd: List[str], returncode: int, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"ffmpeg command failed ({returncode}): {' '.join(cmd)}\n{stderr}")


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise FFmpegError(cmd, p.returncode, err)


def video_probe(video_path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"ffprobe failed: {err}")

    payload = json.loads(p.stdout or "{}")
    streams = payload.get("streams") or []
    video_stream = next((st for st in streams if st.get("codec_type") == "video"), {})

    def _parse_rate(val: str) -> float:
        txt = str(val or "0/1")
        if "/" in txt:
            a, b = txt.split("/", 1)
            num = float(a or 0.0)
            den = float(b or 1.0)
            return num / den if den else 0.0
        return float(txt or 0.0)

    avg_rate_raw = str(video_stream.get("avg_frame_rate") or "0/1")
    real_rate_raw = str(video_stream.get("r_frame_rate") or "0/1")
    avg_rate = _parse_rate(avg_rate_raw)
    real_rate = _parse_rate(real_rate_raw)

    fmt = payload.get("format") or {}
    duration = video_stream.get("duration") or fmt.get("duration")
    bit_rate = video_stream.get("bit_rate") or fmt.get("bit_rate")

    return {
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": avg_rate if avg_rate > 0 else real_rate,
        "codec": video_stream.get("codec_name"),
        "bit_rate": int(bit_rate) if bit_rate not in (None, "") else None,
        "duration": float(duration) if duration not in (None, "") else None,
        "is_vfr": abs(avg_rate - real_rate) > 0.01,
        "avg_frame_rate": avg_rate_raw,
        "r_frame_rate": real_rate_raw,
    }


def normalize_video_input(in_path: str, out_path: str, probe: Dict[str, Any]) -> None:
    fps = float(probe.get("fps") or 30.0)
    fps = 30.0 if fps <= 0 else fps
    width = int(probe.get("width") or 0)
    height = int(probe.get("height") or 0)

    vf_parts = [f"fps={fps:.6f}"]
    if max(width, height) > 1280:
        if width >= height:
            vf_parts.append("scale=1280:-2")
        else:
            vf_parts.append("scale=-2:1280")

    _run([
        "ffmpeg", "-y", "-i", in_path,
        "-vf", ",".join(vf_parts),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        out_path,
    ])


def cut_clip(in_path: str, start: float, end: float, out_path: str) -> None:
    dur = max(0.01, end - start)
    _run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", in_path, "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", out_path])


def concat_clips(paths: List[str], out_path: str) -> None:
    if not paths:
        return
    lst = out_path + ".txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{p}'\n")
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out_path])


@dataclass
class _LiteTrack:
    track_id: int
    box: Tuple[int, int, int, int]
    confidence: float
    confirmed: bool = True

    def is_confirmed(self):
        return self.confirmed

    def to_ltrb(self):
        return self.box


class LightweightIOUTracker:
    def __init__(self, max_misses: int = 12, min_match_iou: float = 0.12):
        self.max_misses = max_misses
        self.min_match_iou = min_match_iou
        self._next_id = 1
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def update_tracks(self, detections, frame=None):
        det_boxes = []
        for xywh, conf, _ in detections:
            x, y, w, h = [int(v) for v in xywh]
            det_boxes.append(((x, y, x + w, y + h), float(conf)))

        updated = set()
        used = set()
        for tid, track in list(self._tracks.items()):
            best_idx, best_score = -1, 0.0
            for i, (box, conf) in enumerate(det_boxes):
                if i in used:
                    continue
                score = _iou(track["box"], box)
                if score > best_score:
                    best_idx, best_score = i, score
            if best_idx >= 0 and best_score >= self.min_match_iou:
                box, conf = det_boxes[best_idx]
                track.update({"box": box, "conf": conf, "misses": 0, "hits": track["hits"] + 1})
                updated.add(tid)
                used.add(best_idx)
            else:
                track["misses"] += 1

        for i, (box, conf) in enumerate(det_boxes):
            if i in used:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"box": box, "conf": conf, "hits": 1, "misses": 0}
            updated.add(tid)

        for tid in list(self._tracks.keys()):
            if self._tracks[tid]["misses"] > self.max_misses:
                del self._tracks[tid]

        output = []
        for tid, t in self._tracks.items():
            if t["misses"] == 0:
                output.append(_LiteTrack(track_id=tid, box=t["box"], confidence=t["conf"], confirmed=t["hits"] >= 1))
        return output


def _perf_breakdown(perf: Dict[str, float], frames: int) -> Dict[str, Any]:
    comps = ["frame_read_ms", "yolo_ms", "deepsort_ms", "ocr_ms", "loop_ms"]
    avg = {k: (perf[k] / max(1, frames)) for k in comps}
    loop_total = max(1e-6, perf["loop_ms"])
    pct = {k: (100.0 * perf[k] / loop_total) for k in comps if k != "loop_ms"}
    return {"avg_ms": avg, "pct_of_loop": pct}


def track_presence(video_path: str, setup: Dict[str, Any], heartbeat=None, cancel_check=None):
    if cv2 is None or YOLO is None:
        raise RuntimeError("tracking dependencies unavailable")

    params = TrackingParams(**{k: v for k, v in setup.items() if k in TrackingParams.__annotations__})
    if "ocr_every_n_frames" in setup:
        params.ocr_every_n = max(1, int(setup.get("ocr_every_n_frames", params.ocr_every_n)))
    params.detect_stride = max(1, int(setup.get("detect_stride", params.detect_stride)))
    params.yolo_imgsz = max(320, int(setup.get("yolo_imgsz", params.yolo_imgsz)))
    params.yolo_batch = max(1, int(setup.get("yolo_batch", params.yolo_batch)))
    params.ocr_disable = bool(setup.get("ocr_disable", params.ocr_disable))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov8s.pt")
    device, yolo_device = resolve_device()
    is_cuda = device.startswith("cuda")
    warm_easyocr_models(device)
    ocr, ocr_gpu = _build_ocr_reader(device)

    tracker_type = str(setup.get("tracker_type", "bytetrack") or "bytetrack").lower()
    if tracker_type != "bytetrack":
        raise RuntimeError(f"Unsupported tracker_type={tracker_type}. ShiftClipper requires tracker_type=bytetrack")
    tracker = LightweightIOUTracker(max_misses=max(3, int(params.lost_timeout * fps / max(1, params.detect_stride))))

    detect_stride = params.detect_stride
    ocr_every_n = max(1, int(params.ocr_every_n))
    yolo_batch = params.yolo_batch

    log.info(
        "runtime device=%s yolo_device_arg=%s tracker=%s torch=%s cuda_available=%s gpu=%s imgsz=%s stride=%s batch=%s",
        device,
        yolo_device,
        tracker_type,
        getattr(torch, "__version__", "n/a") if torch else "missing",
        bool(torch and torch.cuda.is_available()),
        torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "-",
        params.yolo_imgsz,
        detect_stride,
        yolo_batch,
    )

    jersey_hsv = _hex_to_hsv(setup.get("jersey_color", "#203524"))
    opponent_hsv = _hex_to_hsv(setup.get("opponent_color", "#ffffff"))
    player_num = re.sub(r"\D+", "", str(setup.get("player_number") or ""))

    state = "SEARCH"
    lock_state = "SEARCHING"
    lost_since = None
    locked_track_id = None
    last_box = None
    last_seen = -1.0
    lost_start_time = None
    reacquire_until = 0.0
    clip_start_time: Optional[float] = None
    clip_last_seen_time: Optional[float] = None
    gate_hold_start: Optional[float] = None
    clip_bench_enter_time: Optional[float] = None
    clip_end_reason: Optional[str] = None
    reacquire_hits: Dict[int, int] = {}
    target_embed: Optional[np.ndarray] = None
    target_embed_history: List[List[float]] = []
    embed_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    ocr_penalties: Dict[int, float] = {}

    seed_clicks = []
    for click in setup.get("clicks") or []:
        seed_t = max(0.0, float(click.get("t", 0.0)))
        seed_clicks.append({
            "t": seed_t,
            "seed_frame": int(seed_t * fps),
            "seed_px": (int(float(click.get("x", 0.0)) * w), int(float(click.get("y", 0.0)) * h)),
            "acquired": False,
            "matched_track_id": None,
            "matched_dist": None,
            "matched_iou": None,
            "seed_clip_emitted": False,
        })

    seed_window_s = float(setup.get("seed_window_s", params.seed_window_s))
    seed_dist_max = float(setup.get("seed_dist_max", params.seed_dist_max))
    seed_iou_min = float(setup.get("seed_iou_min", params.seed_iou_min))
    seed_bonus_val = float(setup.get("seed_bonus", params.seed_bonus))
    ocr_veto_conf = float(setup.get("ocr_veto_conf", params.ocr_veto_conf))
    ocr_veto_conf_seed_hard = 0.95
    tracking_mode = str(setup.get("tracking_mode") or params.tracking_mode).lower()

    polygon_normalized = bool(setup.get("polygon_coords_normalized", True))
    rink_polygon = _normalize_polygon(setup.get("rink_polygon") or [], w, h, polygon_normalized)
    bench_polygons = [_normalize_polygon(poly, w, h, polygon_normalized) for poly in (setup.get("bench_polygons") or [])]

    loss_timeout_sec = float(setup.get("loss_timeout_sec", setup.get("LOSS_TIMEOUT_SEC", params.loss_timeout_sec)))
    reacquire_max_sec = float(setup.get("reacquire_max_sec", setup.get("REACQUIRE_MAX_SEC", params.reacquire_max_sec)))
    reacquire_confirm_frames = max(1, int(setup.get("reacquire_confirm_frames", setup.get("REACQUIRE_CONFIRM_FRAMES", params.reacquire_confirm_frames))))
    reid_sim_threshold = float(setup.get("reid_sim_threshold", setup.get("REID_SIM_THRESHOLD", params.reid_sim_threshold)))
    edge_margin_px = float(setup.get("edge_margin_px", params.edge_margin_px))

    segments, shifts = [], []
    shift_start = None
    shift_state = "OFF_ICE"
    present_prev = False
    timeline = []
    overlay_path = None
    writer = None
    if params.debug_overlay:
        overlay_path = str(Path(video_path).with_name("debug_overlay.mp4"))
        writer = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    perf = {
        "frame_read_ms": 0.0,
        "yolo_ms": 0.0,
        "deepsort_ms": 0.0,
        "ocr_ms": 0.0,
        "loop_ms": 0.0,
        "ocr_calls": 0,
        "frames": 0,
        "effective_fps": 0.0,
        "device": device,
        "yolo_device": yolo_device,
        "tracker_type": tracker_type,
    }

    frame_idx = 0
    processed_idx = 0
    last_hb = 0.0
    started_at = time.perf_counter()
    seg_start = 0.0

    while True:
        if cancel_check and cancel_check():
            raise RuntimeError("cancelled")

        batch_frames, batch_indices, batch_times = [], [], []
        while len(batch_frames) < yolo_batch:
            read_start = time.perf_counter()
            ok, frame = cap.read()
            perf["frame_read_ms"] += (time.perf_counter() - read_start) * 1000.0
            if not ok:
                break
            current_idx = frame_idx
            frame_idx += 1
            if current_idx % detect_stride != 0:
                continue
            batch_frames.append(frame)
            batch_indices.append(current_idx)
            batch_times.append(current_idx / fps)

        if not batch_frames:
            break

        yolo_start = time.perf_counter()
        source = batch_frames if len(batch_frames) > 1 else batch_frames[0]
        preds = model.predict(source=source, conf=0.25, device=yolo_device, imgsz=params.yolo_imgsz, half=is_cuda, verbose=False)
        perf["yolo_ms"] += (time.perf_counter() - yolo_start) * 1000.0
        if not isinstance(preds, list):
            preds = [preds]

        for frame, p, current_idx, t in zip(batch_frames, preds, batch_indices, batch_times):
            frame_loop_start = time.perf_counter()
            boxes_xyxy = p.boxes.xyxy.detach().cpu().numpy() if p.boxes is not None else np.empty((0, 4))
            confs = p.boxes.conf.detach().cpu().numpy() if p.boxes is not None else np.empty((0,))
            classes = p.boxes.cls.detach().cpu().numpy() if p.boxes is not None else np.empty((0,))

            dets = []
            for xyxy, conf, cls in zip(boxes_xyxy, confs, classes):
                if int(cls) != 0:
                    continue
                x1, y1, x2, y2 = _clip(xyxy, w, h)
                dets.append(([x1, y1, x2 - x1, y2 - y1], float(conf), "person"))

            ds_start = time.perf_counter()
            tracks = tracker.update_tracks(dets, frame=frame)
            perf["deepsort_ms"] += (time.perf_counter() - ds_start) * 1000.0

            active_seed, click_dist = None, float("inf")
            for seed in seed_clicks:
                dt = t - seed["t"]
                if 0.0 <= dt <= seed_window_s and dt < click_dist:
                    active_seed = seed
                    click_dist = dt

            nearest_seed_track_id = None
            nearest_seed_dist = float("inf")
            nearest_seed_iou = 0.0
            in_seed_window = active_seed is not None
            if active_seed is not None:
                px, py = active_seed["seed_px"]
                click_box_half = max(8, int(0.04 * math.hypot(w, h)))
                click_box = _clip((px - click_box_half, py - click_box_half, px + click_box_half, py + click_box_half), w, h)
                for tr in tracks:
                    if not tr.is_confirmed():
                        continue
                    box = _clip(tr.to_ltrb(), w, h)
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    dist = math.hypot(cx - px, cy - py) / max(1.0, math.hypot(w, h))
                    iou = _iou(_expand_box(box, 1.15, w, h), click_box)
                    match_score = min(1.0, dist / max(1e-6, seed_dist_max)) - iou
                    if match_score < (min(1.0, nearest_seed_dist / max(1e-6, seed_dist_max)) - nearest_seed_iou):
                        nearest_seed_dist = dist
                        nearest_seed_iou = iou
                        nearest_seed_track_id = tr.track_id

            seed_bonus_track_id = None
            seed_match_active = (
                active_seed is not None
                and nearest_seed_track_id is not None
                and (nearest_seed_dist <= seed_dist_max or nearest_seed_iou >= seed_iou_min)
            )
            if seed_match_active:
                seed_bonus_track_id = nearest_seed_track_id
                active_seed["acquired"] = True
                active_seed["matched_track_id"] = nearest_seed_track_id
                active_seed["matched_dist"] = nearest_seed_dist
                active_seed["matched_iou"] = nearest_seed_iou
                timeline.append({
                    "t": t,
                    "event": "seed_match",
                    "click_t": active_seed["t"],
                    "chosen_track_id": nearest_seed_track_id,
                    "dist": nearest_seed_dist,
                    "iou": nearest_seed_iou,
                    "bonus_applied": True,
                    "in_seed_window": in_seed_window,
                })
            elif active_seed is not None:
                timeline.append({
                    "t": t,
                    "event": "seed_match",
                    "click_t": active_seed["t"],
                    "chosen_track_id": nearest_seed_track_id,
                    "dist": nearest_seed_dist if nearest_seed_track_id is not None else None,
                    "iou": nearest_seed_iou if nearest_seed_track_id is not None else None,
                    "bonus_applied": False,
                    "in_seed_window": in_seed_window,
                })

            best, ocr_used = None, 0
            candidates = []
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                box = _clip(tr.to_ltrb(), w, h)
                feet = _bbox_feet_point(box)
                in_rink = True if not params.use_rink_mask or not rink_polygon else _poly_contains_with_margin(feet, rink_polygon, edge_margin_px)
                in_bench = any(_poly_contains_with_margin(feet, poly, edge_margin_px) for poly in bench_polygons if poly)
                if params.use_rink_mask and rink_polygon and not in_rink:
                    continue
                if state in {"SEARCH", "LOST"} and params.use_bench_mask and in_bench and not bool(setup.get("allow_bench_reacquire", params.allow_bench_reacquire)):
                    continue
                cscore = _color_score(frame, box, jersey_hsv, params.color_tolerance)
                mean_hsv = _mean_hsv(frame, box)
                target_dist = _hsv_distance(mean_hsv, jersey_hsv)
                opponent_dist = _hsv_distance(mean_hsv, opponent_hsv)
                opponent_penalty = 0.25 if opponent_dist <= target_dist + 0.03 else 0.0
                motion = _iou(box, last_box) if last_box else 0.0
                identity_bonus = params.identity_weight if (locked_track_id is not None and tr.track_id == locked_track_id) else 0.0
                seed_bonus = seed_bonus_val if (seed_bonus_track_id is not None and tr.track_id == seed_bonus_track_id) else 0.0
                score = params.color_weight * cscore + params.motion_weight * motion + identity_bonus + seed_bonus - opponent_penalty
                emb = None
                sim = -1.0
                crop = frame[box[1]:box[3], box[0]:box[2]]
                sharpness = _crop_sharpness(crop) if crop.size else 0.0
                if params.use_reid and crop.size and (box[2] - box[0]) >= params.reid_min_px and (box[3] - box[1]) >= params.reid_min_px and sharpness >= params.reid_sharpness_threshold:
                    cache_key = (current_idx, tr.track_id)
                    emb = embed_cache.get(cache_key)
                    if emb is None:
                        emb = _embed_crop(crop)
                        embed_cache[cache_key] = emb
                    sim = _cosine_similarity(target_embed, emb) if emb is not None else -1.0
                candidates.append({
                    "track_id": tr.track_id,
                    "box": box,
                    "score": score,
                    "color": cscore,
                    "motion": motion,
                    "ocr_txt": None,
                    "ocr_conf": 0.0,
                    "ocr_match": 0.0,
                    "ocr_veto": False,
                    "seed_match": seed_bonus > 0.0,
                    "seed_dist": nearest_seed_dist if tr.track_id == seed_bonus_track_id else None,
                    "seed_iou": nearest_seed_iou if tr.track_id == seed_bonus_track_id else None,
                    "in_rink": in_rink,
                    "in_bench": in_bench,
                    "embed": emb,
                    "sim": sim,
                    "sharpness": sharpness,
                    "opponent_penalty": opponent_penalty,
                    "target_dist": target_dist,
                    "opponent_dist": opponent_dist,
                })

            max_ocr_crops = max(1, int(setup.get("ocr_max_crops_per_frame", params.ocr_max_crops_per_frame)))
            should_run_ocr = (
                ocr is not None
                and not params.ocr_disable
                and (current_idx % ocr_every_n == 0)
                and bool(candidates)
            )
            if should_run_ocr:
                candidates.sort(key=lambda c: c["score"], reverse=True)
                for cand in candidates[:max_ocr_crops]:
                    if ocr_used >= max_ocr_crops:
                        break
                    x1, y1, x2, y2 = cand["box"]
                    bh, bw = max(1, y2 - y1), max(1, x2 - x1)
                    cx1 = x1 + int(0.08 * bw)
                    cx2 = x2 - int(0.08 * bw)
                    cy1 = y1 + int(0.18 * bh)
                    cy2 = y1 + int(0.72 * bh)
                    crop = frame[max(0, cy1):min(h, cy2), max(0, cx1):min(w, cx2)]
                    if crop.size == 0:
                        continue
                    ocr_start = time.perf_counter()
                    try:
                        rr = ocr.readtext(crop, detail=1, paragraph=False)
                        perf["ocr_calls"] += 1
                        ocr_used += 1
                        for _, txt, cf in rr:
                            d = _parse_digits(str(txt))
                            if not d:
                                continue
                            cand["ocr_txt"] = d
                            cand["ocr_conf"] = float(cf)
                            if player_num and d == player_num and cand["ocr_conf"] >= params.ocr_min_conf:
                                cand["ocr_match"] = 1.0
                                break
                            mismatch_veto = player_num and d != player_num and cand["ocr_conf"] >= ocr_veto_conf
                            if mismatch_veto:
                                in_seed_window_for_cand = bool(active_seed is not None and cand.get("seed_match"))
                                hard_seed_veto = bool(cand["ocr_conf"] >= ocr_veto_conf_seed_hard)
                                if in_seed_window_for_cand and not hard_seed_veto:
                                    timeline.append({
                                        "t": t,
                                        "event": "seed_veto_suppressed",
                                        "click_t": active_seed["t"],
                                        "track_id": cand["track_id"],
                                        "ocr_txt": d,
                                        "ocr_conf": cand["ocr_conf"],
                                    })
                                    continue
                                cand["ocr_veto"] = True
                                penalty_sec = max(0.1, float(setup.get("ocr_veto_seconds", params.ocr_veto_seconds)))
                                ocr_penalties[cand["track_id"]] = t + penalty_sec
                                timeline.append({"t": t, "event": "ocr_veto", "track_id": cand["track_id"], "ocr_txt": d, "ocr_conf": cand["ocr_conf"], "penalty_seconds": penalty_sec})
                                break
                    except Exception as exc:
                        if ocr_gpu and is_cuda and _is_cuda_fork_error(exc):
                            raise RuntimeError("CUDA OCR initialization failed in worker process; CPU fallback is disabled") from exc
                    finally:
                        perf["ocr_ms"] += (time.perf_counter() - ocr_start) * 1000.0

            for cand in candidates:
                box = cand["box"]
                area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(1.0, float(w * h))
                is_closeup = area_ratio >= float(setup.get("closeup_bbox_area_ratio", params.closeup_bbox_area_ratio))
                strong_ocr_match = bool(player_num and cand.get("ocr_txt") == player_num and cand.get("ocr_conf", 0.0) >= ocr_veto_conf)
                cand["closeup_blocked"] = bool(player_num) and is_closeup and not cand.get("seed_match") and not strong_ocr_match
                penalty_active = t < ocr_penalties.get(cand["track_id"], 0.0)
                if penalty_active:
                    cand["score"] *= 0.65
                if cand["closeup_blocked"]:
                    cand["score"] -= 0.75
                cand["score"] += params.ocr_weight * cand["ocr_match"]
                if params.use_reid and cand.get("sim", -1.0) >= 0:
                    cand["score"] += 0.5 * cand["sim"]
                if best is None or cand["score"] > best["score"]:
                    best = cand

            prev_state = state
            prev_lock_state = lock_state
            reacquire_threshold = float(setup.get("reacquire_score_lock_threshold", 0.40))
            lock_threshold = reacquire_threshold if t <= reacquire_until else params.score_lock_threshold
            unlock_threshold = float(setup.get("score_unlock_threshold", params.score_unlock_threshold))
            allow_seed_relaxed_lock = bool(best and best.get("seed_match"))
            if allow_seed_relaxed_lock:
                lock_threshold = min(lock_threshold, unlock_threshold)
            if best:
                last_box = best["box"]

            sim_ok = bool(best and (not params.use_reid or best.get("sim", -1.0) >= reid_sim_threshold or state == "LOCKED"))
            if best and sim_ok and best["score"] >= lock_threshold and not best.get("closeup_blocked", False):
                if state in {"SEARCH", "LOST"}:
                    reacquire_hits[best["track_id"]] = reacquire_hits.get(best["track_id"], 0) + 1
                    if reacquire_hits[best["track_id"]] < reacquire_confirm_frames:
                        state = "SEARCH"
                    else:
                        timeline.append({"t": t, "event": "reacquire", "reason": "confirm_frames", "hits": reacquire_hits[best["track_id"]], "track_id": best["track_id"]})
                        state = "LOCKED"
                else:
                    state = "LOCKED"
                if state == "LOCKED":
                    if locked_track_id is not None and best["track_id"] != locked_track_id and best.get("motion", 0.0) < 0.15:
                        timeline.append({
                            "t": t,
                            "event": "swap_suspicion",
                            "from_track_id": locked_track_id,
                            "to_track_id": best["track_id"],
                            "motion_iou": best.get("motion", 0.0),
                            "identity_score": best.get("score"),
                            "frame_idx": current_idx,
                        })
                    locked_track_id = best["track_id"]
                    last_seen = t
                    clip_last_seen_time = t
                    lost_since = None
                    lost_start_time = None
                    reacquire_until = 0.0
                    reacquire_hits.clear()
                    if clip_start_time is None:
                        clip_start_time = t
                    if player_num and (best.get("ocr_match", 0.0) >= 1.0 or best.get("seed_match")):
                        lock_state = "CONFIRMED"
                    elif player_num:
                        lock_state = "PROVISIONAL"
                    else:
                        lock_state = "CONFIRMED"
                    if best.get("embed") is not None and (current_idx % max(1, int(setup.get("reid_every_n_frames", params.reid_every_n_frames))) == 0):
                        if target_embed is None:
                            target_embed = best["embed"]
                        else:
                            alpha = float(setup.get("reid_alpha", params.reid_alpha))
                            target_embed = target_embed * alpha + best["embed"] * (1.0 - alpha)
                            n = np.linalg.norm(target_embed)
                            if n > 1e-6:
                                target_embed = target_embed / n
                        if target_embed is not None:
                            target_embed_history.append(target_embed.tolist()[:16])
            elif state == "LOCKED":
                if best and best["score"] >= unlock_threshold:
                    lost_since = None
                else:
                    if lost_since is None:
                        lost_since = t
                    if (t - lost_since) >= max(0.0, float(setup.get("lost_timeout", params.lost_timeout))):
                        lost_start_time = t
                        state = "LOST"
                        reacquire_until = t + max(0.0, reacquire_max_sec)
                        timeline.append({"t": t, "event": "lost", "reason": "score_below_threshold_timeout", "score": best["score"] if best else None, "threshold": unlock_threshold, "lost_timeout": float(setup.get("lost_timeout", params.lost_timeout))})
            elif state in {"SEARCH", "LOST"}:
                if lost_start_time is not None and t > reacquire_until:
                    clip_end_reason = ClipEndReason.LOST_TIMEOUT.value

            allow_unconfirmed = bool(setup.get("allow_unconfirmed_clips", params.allow_unconfirmed_clips))
            clip_score_threshold = max(0.60, float(params.score_lock_threshold))
            identity_score = float(best["score"]) if best else -1.0
            base_gate = (
                state == "LOCKED"
                and lock_state in {"CONFIRMED", "PROVISIONAL"}
                and identity_score >= clip_score_threshold
            )
            if base_gate and gate_hold_start is None:
                gate_hold_start = t
            elif not base_gate:
                gate_hold_start = None
            gate_held = bool(base_gate and gate_hold_start is not None and (t - gate_hold_start) >= float(setup.get("min_track_seconds", params.min_track_seconds)))
            present = bool(
                gate_held
                and (
                    lock_state == "CONFIRMED"
                    or (lock_state == "PROVISIONAL" and allow_unconfirmed)
                )
            )
            clip_reason = "allowed"
            if lock_state == "SEARCHING":
                clip_reason = "searching"
            elif state != "LOCKED":
                clip_reason = "state_not_locked"
            elif lock_state == "PROVISIONAL" and not allow_unconfirmed:
                clip_reason = "provisional_disallowed"
            elif identity_score < clip_score_threshold:
                clip_reason = "identity_below_threshold"
            elif not gate_held:
                clip_reason = "min_track_not_met"
            timeline.append({
                "t": t,
                "event": "clip_allowed" if present else "clip_blocked",
                "reason": clip_reason,
                "identity_score": identity_score,
                "lock_state": lock_state,
                "allow_unconfirmed_clips": allow_unconfirmed,
                "min_track_seconds": float(setup.get("min_track_seconds", params.min_track_seconds)),
                "clip_score_threshold": clip_score_threshold,
                "frame_idx": current_idx,
            })

            if present and not present_prev:
                seg_start = t
                clip_start_time = t
                if shift_state == "OFF_ICE":
                    shift_state = "ON_ICE"
                    shift_start = t
            if present and best and params.use_bench_mask and best.get("in_bench"):
                clip_bench_enter_time = clip_bench_enter_time or t
                if (t - clip_bench_enter_time) >= max(0.0, float(setup.get("bench_confirm_sec", 0.35))):
                    clip_end_reason = ClipEndReason.BENCH_ENTER.value
            else:
                clip_bench_enter_time = None

            if present and clip_start_time is not None and (t - clip_start_time) > float(setup.get("max_clip_len_sec", params.max_clip_len_sec)):
                clip_end_reason = "safety_close_max_len"

            if clip_end_reason == ClipEndReason.LOST_TIMEOUT.value and clip_start_time is not None:
                seg_end = _compute_clip_end_for_loss(clip_last_seen_time or last_seen or t, loss_timeout_sec, frame_idx / max(fps, 1.0))
                _append_segment(segments, clip_start_time, seg_end, ClipEndReason.LOST_TIMEOUT)
                present = False
            elif clip_end_reason == ClipEndReason.BENCH_ENTER.value and clip_start_time is not None:
                _append_segment(segments, clip_start_time, t, ClipEndReason.BENCH_ENTER)
                present = False
            elif clip_end_reason == "safety_close_max_len" and clip_start_time is not None:
                _append_segment(segments, clip_start_time, t, ClipEndReason.MAX_LEN_GUARD)
                present = False

            if (not present) and present_prev and clip_end_reason is None and clip_start_time is not None:
                _append_segment(segments, clip_start_time, t, ClipEndReason.LOST_TIMEOUT)
                clip_end_reason = "clip_blocked"

            if (not present) and present_prev:
                seg_end = t
                if shift_state == "ON_ICE" and shift_start is not None:
                    shifts.append((shift_start, seg_end))
                    shift_state = "OFF_ICE"
                    shift_start = None
                timeline.append({"t": t, "event": "clip_end", "reason": clip_end_reason or "clip_blocked", "start_time": clip_start_time, "end_time": seg_end})
                clip_start_time = None
                clip_end_reason = None

            if prev_state != state:
                timeline.append({"t": t, "event": "state", "from": prev_state, "to": state})
            if prev_lock_state != lock_state:
                timeline.append({"t": t, "event": "lock_state", "from": prev_lock_state, "to": lock_state})

            log.info(
                "decision frame_time=%.3f status=%s candidates=%d chosen_id=%s sim=%.3f in_rink=%s in_bench=%s end_reason=%s",
                t,
                state,
                len(candidates),
                best.get("track_id") if best else None,
                float(best.get("sim", -1.0)) if best else -1.0,
                bool(best.get("in_rink", True)) if best else None,
                bool(best.get("in_bench", False)) if best else None,
                clip_end_reason,
            )

            if writer is not None:
                draw = frame.copy()
                if best:
                    x1, y1, x2, y2 = best["box"]
                    cv2.rectangle(draw, (x1, y1), (x2, y2), (40, 220, 40), 2)
                    cv2.putText(draw, f"score:{best['score']:.2f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(draw, f"state:{state} lock:{lock_state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
                writer.write(draw)

            processed_idx += 1
            perf["frames"] = processed_idx
            perf["loop_ms"] += (time.perf_counter() - frame_loop_start) * 1000.0
            elapsed = max(1e-6, time.perf_counter() - started_at)
            perf["effective_fps"] = processed_idx / elapsed

            if heartbeat and (time.time() - last_hb) >= HEARTBEAT_SECONDS:
                breakdown = _perf_breakdown(perf, processed_idx)
                perf_summary = {
                    "eff_fps": perf["effective_fps"],
                    "frames": processed_idx,
                    "frame_idx": current_idx,
                    "total_frames": total,
                    "ocr_calls": perf["ocr_calls"],
                    "device": device,
                    "yolo_device": yolo_device,
                    "tracker_type": tracker_type,
                    "totals": {k: perf[k] for k in ["frame_read_ms", "yolo_ms", "deepsort_ms", "ocr_ms", "loop_ms", "ocr_calls"]},
                    **breakdown,
                }
                heartbeat(current_idx, total, t, perf_summary)
                last_hb = time.time()

            present_prev = present

    cap.release()
    if writer is not None:
        writer.release()
    if present_prev:
        eof_t = frame_idx / max(fps, 1.0)
        _append_segment(segments, clip_start_time or seg_start, eof_t, ClipEndReason.VIDEO_END)
        timeline.append({"t": eof_t, "event": "clip_end", "reason": ClipEndReason.VIDEO_END.value})
        if shift_state == "ON_ICE" and shift_start is not None:
            shifts.append((shift_start, eof_t))

    if bool(setup.get("allow_seed_clips", params.allow_seed_clips)):
        clip_center_lead = float(setup.get("extend_sec", 20.0))
        for seed in seed_clicks:
            if not seed.get("acquired") or seed.get("seed_clip_emitted"):
                continue
            clip_start = max(0.0, seed["t"] - clip_center_lead)
            clip_end = max(clip_start + params.min_clip_seconds, seed["t"])
            seed["seed_clip_emitted"] = True
            segments.append((clip_start, clip_end, "seed_clip"))
            timeline.append({
                "t": seed["t"],
                "event": "seed_clip_emitted",
                "click_t": seed["t"],
                "clip_start": clip_start,
                "clip_end": clip_end,
            })

    merged = _merge_segments(
        segments,
        min_clip_seconds=params.min_clip_seconds,
        gap_merge_seconds=float(setup.get("gap_merge_seconds", params.gap_merge_seconds)),
        tracking_mode=tracking_mode,
        reacquire_window_seconds=float(setup.get("reacquire_window_seconds", params.reacquire_window_seconds)),
    )

    perf_summary = {
        "eff_fps": perf["effective_fps"],
        "frames": processed_idx,
        "total_frames": total,
        "ocr_calls": perf["ocr_calls"],
        "device": device,
        "yolo_device": yolo_device,
        "tracker_type": tracker_type,
        "totals": {k: perf[k] for k in ["frame_read_ms", "yolo_ms", "deepsort_ms", "ocr_ms", "loop_ms", "ocr_calls"]},
        **_perf_breakdown(perf, processed_idx),
    }
    return {
        "segments": merged,
        "shifts": shifts,
        "timeline": timeline,
        "debug_overlay_path": overlay_path,
        "fps": fps,
        "perf": perf_summary,
        "target_embed_history": target_embed_history,
    }

def process_job(job_id: str) -> Dict[str, Any]:
    _ensure_dirs()
    cur = get_current_job()
    stage = {"name": "queued", "updated": time.time(), "stalled": False}
    stop_watchdog = False

    def watchdog():
        while not stop_watchdog:
            time.sleep(2)
            if time.time() - stage["updated"] > STALL_TIMEOUT_S:
                stage["stalled"] = True
                return

    threading.Thread(target=watchdog, daemon=True).start()

    job_dir = JOBS_DIR / job_id
    meta_path = job_dir / "job.json"
    setup_path = job_dir / "setup.json"
    if not meta_path.exists():
        raise RuntimeError("missing job json")

    meta = json.loads(meta_path.read_text())
    raw_setup = json.loads(setup_path.read_text()) if setup_path.exists() else meta.get("setup", {})
    setup = normalize_setup(raw_setup)

    def write_status(status, stage_name, progress, message, **extra):
        stage.update({"name": stage_name, "updated": time.time()})
        meta.update({"status": status, "stage": stage_name, "progress": int(progress), "message": message, "updated_at": time.time(), **extra})
        meta_path.write_text(json.dumps(meta, indent=2))
        if cur:
            cur.meta = {**(cur.meta or {}), "stage": stage_name, "progress": int(progress), "message": message, **extra}
            cur.save_meta()
        log.info("job_id=%s stage=%s progress=%s message=%s", job_id, stage_name, progress, message, extra={"job_id": job_id, "stage": stage_name})

    def cancel_check() -> bool:
        if not meta_path.exists():
            return False
        try:
            latest_meta = json.loads(meta_path.read_text())
        except Exception:
            return False
        return bool(latest_meta.get("cancel_requested"))

    try:
        in_path = meta.get("video_path") or str(job_dir / "in.mp4")
        if not Path(in_path).exists():
            raise RuntimeError("missing input")

        verify_mode = bool(setup.get("verify_mode", False))
        tracking_mode = str(setup.get("tracking_mode") or "clip").lower()

        device, _ = resolve_device()
        cuda_available = bool(torch and torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "-"
        log.info(
            "job_id=%s gpu_check cuda_available=%s gpu_name=%s chosen_device=%s",
            job_id,
            cuda_available,
            gpu_name,
            device,
            extra={"job_id": job_id, "stage": "queued"},
        )

        write_status("processing", "queued", 10, "Queued for processing.")

        if verify_mode:
            meta.update({
                "status": "verified",
                "stage": "verified",
                "progress": 100,
                "message": "Verify mode: no clips/combined generated.",
                "clips": [],
                "combined_path": None,
                "combined_url": None,
                "artifacts": {"clips": [], "combined_path": None, "combined_url": None},
            })
            (job_dir / "results.json").write_text(json.dumps(meta, indent=2))
            meta_path.write_text(json.dumps(meta, indent=2))
            return meta

        def hb(frame_idx, total, t, perf=None):
            if cancel_check():
                raise RuntimeError("cancelled")
            if stage["stalled"]:
                raise RuntimeError(f"Stalled in {stage['name']}")
            pct = int(min(70, 10 + (frame_idx / max(1, total)) * 60))
            extra = {"perf": perf} if perf else {}
            write_status("processing", "tracking", pct, f"Tracking in progress ({t:.1f}s)", **extra)

        probe = video_probe(in_path)
        probe_path = job_dir / "probe.json"
        probe_path.write_text(json.dumps(probe, indent=2))

        normalize_video = bool(setup.get("normalize_video", False))
        tracking_input_path = in_path
        if normalize_video:
            tracking_input_path = str(job_dir / "normalized.mp4")
            log.info(
                "job_id=%s normalization enabled source=%s target=%s",
                job_id,
                in_path,
                tracking_input_path,
                extra={"job_id": job_id, "stage": "tracking"},
            )
            write_status("processing", "preflight", 11, "Normalizing video")
            normalize_video_input(in_path, tracking_input_path, probe)
        else:
            log.info(
                "job_id=%s normalization disabled source=%s",
                job_id,
                in_path,
                extra={"job_id": job_id, "stage": "tracking"},
            )

        meta["transcode_used"] = normalize_video
        meta_path.write_text(json.dumps(meta, indent=2))

        write_status("processing", "tracking", 12, "Tracking in progress")
        data = track_presence(tracking_input_path, setup, heartbeat=hb, cancel_check=cancel_check)

        clips_dir = job_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        write_status("processing", "clips", 72, "Creating clips")

        clips = []
        clip_paths = []
        segment_count = max(1, len(data["segments"]))
        for i, (a, b, _) in enumerate(data["segments"], start=1):
            outp = clips_dir / f"clip_{i:03d}.mp4"
            cut_clip(tracking_input_path, a, b + float(setup.get("post_roll", setup.get("extend_sec", 2.0))), str(outp))
            clip_paths.append(str(outp))
            clips.append({"start": a, "end": b, "path": str(outp), "url": f"/data/jobs/{job_id}/clips/{outp.name}"})
            clip_progress = 70 + int((i / segment_count) * 20)
            write_status("processing", "clips", min(90, clip_progress), f"Creating clips ({i}/{segment_count})")

        combined_path = None
        combined_url = None
        if bool(setup.get("generate_combined", True)) and clip_paths:
            write_status("processing", "combined", 92, "Combining video")
            combined_path = str(job_dir / "combined.mp4")
            concat_clips(clip_paths, combined_path)
            write_status("processing", "combined", 98, "Combining video")
            if Path(combined_path).exists():
                combined_url = f"/data/jobs/{job_id}/combined.mp4"

        checked_paths = [str(Path(c["path"])) for c in clips]
        if combined_path:
            checked_paths.append(str(Path(combined_path)))
        missing_outputs = [p for p in checked_paths if not Path(p).exists()]
        if missing_outputs:
            raise RuntimeError(f"Missing expected outputs: {missing_outputs}")
        log.info(
            "job_id=%s outputs clips_dir=%s combined=%s clip_count=%s",
            job_id,
            str(clips_dir),
            combined_path,
            len(clips),
            extra={"job_id": job_id, "stage": "clips"},
        )

        if bool(setup.get("debug_timeline", True)):
            (job_dir / "debug_timeline.json").write_text(json.dumps(data["timeline"], indent=2))
            (job_dir / "debug.json").write_text(json.dumps({"setup": setup, "timeline": data["timeline"]}, indent=2))

        hard_events = {"ocr_veto", "swap_suspicion", "reacquire", "lost"}
        hard_packets = [e for e in data.get("timeline", []) if e.get("event") in hard_events]
        dataset_dir = job_dir / "datasets" / "hard_moments"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / "events.json"
        dataset_path.write_text(json.dumps({"job_id": job_id, "events": hard_packets}, indent=2))

        shifts_json = []
        total_toi = 0.0
        for s, e in data["shifts"]:
            if not (math.isfinite(s) and math.isfinite(e)):
                continue
            if e <= s:
                continue
            dur = max(0.0, e - s)
            total_toi += dur
            shifts_json.append({"start": s, "end": e, "duration": dur, "confidence": 0.7})

        shifts_json_path = job_dir / "shifts.json"
        shifts_json_path.write_text(json.dumps(shifts_json, indent=2))
        if not shifts_json_path.exists():
            raise RuntimeError("Missing shifts.json output")

        artifacts = {
            "clips": clips,
            "combined_path": combined_path,
            "combined_url": combined_url,
            "debug_overlay_path": data.get("debug_overlay_path"),
            "debug_overlay_url": f"/data/jobs/{job_id}/debug_overlay.mp4" if data.get("debug_overlay_path") else None,
            "debug_timeline_path": str(job_dir / "debug_timeline.json") if (job_dir / "debug_timeline.json").exists() else None,
            "debug_timeline_url": f"/data/jobs/{job_id}/debug_timeline.json" if (job_dir / "debug_timeline.json").exists() else None,
            "debug_json_path": str(job_dir / "debug.json") if (job_dir / "debug.json").exists() else None,
            "debug_json_url": f"/data/jobs/{job_id}/debug.json" if (job_dir / "debug.json").exists() else None,
            "hard_moments_dataset_path": str(dataset_path),
            "hard_moments_dataset_url": f"/data/jobs/{job_id}/datasets/hard_moments/events.json",
        }
        artifacts["list"] = [
            {"type": "clip", "path": c["path"], "url": c["url"]} for c in clips
        ]
        if combined_path:
            artifacts["list"].append({"type": "combined", "path": combined_path, "url": combined_url})
        if artifacts.get("debug_overlay_path"):
            artifacts["list"].append({"type": "debug_overlay", "path": artifacts["debug_overlay_path"], "url": artifacts["debug_overlay_url"]})
        if artifacts.get("debug_timeline_path"):
            artifacts["list"].append({"type": "debug_timeline", "path": artifacts["debug_timeline_path"], "url": artifacts["debug_timeline_url"]})

        terminal_status = "done"
        terminal_stage = "done"
        terminal_message = "Processing complete"
        if tracking_mode == "shift" and not shifts_json:
            terminal_status = "done_no_shifts"
            terminal_stage = "done"
            terminal_message = "Run completed with no shifts detected."
        elif tracking_mode == "clip" and not clips and not combined_path:
            raise RuntimeError("No clips created; see debug overlay/timeline (checked clips and combined outputs)")

        if stage["stalled"]:
            raise RuntimeError(f"Stalled in {stage['name']}")

        write_status(terminal_status, terminal_stage, 100, terminal_message)
        meta.update({
            "clips": clips,
            "combined_path": combined_path,
            "combined_url": combined_url,
            "artifacts": artifacts,
            "shifts": shifts_json,
            "total_toi": total_toi,
            "shift_count": len(shifts_json),
            "perf": data.get("perf"),
            "segments_summary": {"count": len(data.get("segments", [])), "segments": data.get("segments", [])},
            "status": terminal_status,
            "stage": terminal_stage,
            "progress": 100,
            "message": terminal_message,
            "updated_at": time.time(),
            "transcode_used": bool(meta.get("transcode_used", False)),
        })
        meta_path.write_text(json.dumps(meta, indent=2))
        (job_dir / "results.json").write_text(json.dumps(meta, indent=2))
        return meta
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        ffmpeg_stderr = None
        if isinstance(e, FFmpegError):
            ffmpeg_stderr = e.stderr
        if str(e).lower() == "cancelled":
            meta.update({"status": "cancelled", "stage": "cancelled", "progress": 100, "message": "Job cancelled.", "error": None, "updated_at": time.time()})
            meta_path.write_text(json.dumps(meta, indent=2))
            (job_dir / "results.json").write_text(json.dumps(meta, indent=2))
            if cur:
                cur.meta = {**(cur.meta or {}), "stage": "cancelled", "progress": 100, "message": "Job cancelled."}
                cur.save_meta()
            log.info("job_id=%s cancelled stage=%s", job_id, stage['name'], extra={"job_id": job_id, "stage": "cancelled"})
            raise
        if stage["stalled"] or "stalled" in str(e).lower():
            err = f"Stalled in {stage['name']}"
        is_cuda_error = "CUDA not available but GPU required" in str(e)
        meta.update({
            "status": "error" if is_cuda_error else "failed",
            "stage": "error" if is_cuda_error else "failed",
            "progress": 100,
            "message": err,
            "error": ffmpeg_stderr or traceback.format_exc(),
            "updated_at": time.time(),
            "transcode_used": bool(meta.get("transcode_used", False)),
        })
        meta_path.write_text(json.dumps(meta, indent=2))
        (job_dir / "results.json").write_text(json.dumps(meta, indent=2))
        if cur:
            cur.meta = {**(cur.meta or {}), "stage": ("error" if is_cuda_error else "failed"), "progress": 100, "message": err}
            cur.save_meta()
        log.error("job_id=%s failed stage=%s error=%s", job_id, stage['name'], err)
        raise
    finally:
        stop_watchdog = True


def self_test_task(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"ok": True, "payload": payload or {}, "worker_pid": os.getpid(), "updated_at": time.time()}
