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
    try:
        return easyocr.Reader(["en"], gpu=use_gpu), use_gpu
    except Exception as exc:
        if use_gpu and _is_cuda_fork_error(exc):
            log.warning("Falling back to CPU OCR due to CUDA fork constraint")
            return easyocr.Reader(["en"], gpu=False), False
        raise


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
    gap_merge_seconds: float = 2.5
    lost_timeout: float = 4.0
    reacquire_window_seconds: float = 8.0
    reacquire_score_lock_threshold: float = 0.40
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

    tracker_type = "bytetrack"
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
    player_num = re.sub(r"\D+", "", str(setup.get("player_number") or ""))

    state = "SEARCHING"
    lock_state = "UNLOCKED"
    lost_since = None
    locked_track_id = None
    last_box = None
    last_seen = -1.0
    reacquire_until = 0.0
    ocr_penalized_tracks: Dict[int, float] = {}
    locked_since = None

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
                cscore = _color_score(frame, box, jersey_hsv, params.color_tolerance)
                motion = _iou(box, last_box) if last_box else 0.0
                identity_bonus = params.identity_weight if (locked_track_id is not None and tr.track_id == locked_track_id) else 0.0
                seed_bonus = seed_bonus_val if (seed_bonus_track_id is not None and tr.track_id == seed_bonus_track_id) else 0.0
                score = params.color_weight * cscore + params.motion_weight * motion + identity_bonus + seed_bonus
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
                    "ocr_penalty_active": t < ocr_penalized_tracks.get(tr.track_id, 0.0),
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
                                ocr_penalized_tracks[cand["track_id"]] = t + max(0.1, float(setup.get("ocr_veto_seconds", params.ocr_veto_seconds)))
                                timeline.append({"t": t, "event": "ocr_veto", "track_id": cand["track_id"], "ocr_txt": d, "ocr_conf": cand["ocr_conf"]})
                                break
                    except Exception as exc:
                        if ocr_gpu and is_cuda and _is_cuda_fork_error(exc):
                            ocr = easyocr.Reader(["en"], gpu=False)
                            ocr_gpu = False
                    finally:
                        perf["ocr_ms"] += (time.perf_counter() - ocr_start) * 1000.0

            for cand in candidates:
                box = cand["box"]
                area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(1.0, float(w * h))
                is_closeup = area_ratio >= float(setup.get("closeup_bbox_area_ratio", params.closeup_bbox_area_ratio))
                strong_ocr_match = bool(player_num and cand.get("ocr_txt") == player_num and cand.get("ocr_conf", 0.0) >= ocr_veto_conf)
                cand["closeup_blocked"] = bool(player_num) and is_closeup and not cand.get("seed_match") and not strong_ocr_match
                if cand["ocr_veto"] or cand.get("ocr_penalty_active"):
                    cand["score"] -= 0.22
                elif cand["closeup_blocked"]:
                    cand["score"] -= 0.75
                cand["score"] += params.ocr_weight * cand["ocr_match"]
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

            if best and best["score"] >= lock_threshold and not best.get("closeup_blocked", False):
                if state != "LOCKED":
                    timeline.append({"t": t, "event": "lock", "reason": "score", "score": best["score"], "threshold": lock_threshold})
                    locked_since = t
                state = "LOCKED"
                locked_track_id = best["track_id"]
                last_seen = t
                lost_since = None
                if player_num and (best.get("ocr_match", 0.0) >= 1.0 or best.get("seed_match")):
                    lock_state = "CONFIRMED"
                elif player_num:
                    lock_state = "PROVISIONAL"
                else:
                    lock_state = "CONFIRMED"
                if t <= reacquire_until:
                    timeline.append({"t": t, "event": "reacquire", "reason": "score_recovered", "score": best["score"]})
                    reacquire_until = 0.0
            elif state == "LOCKED":
                if best and best["score"] >= unlock_threshold:
                    lost_since = None
                elif lost_since is None:
                    lost_since = t
                    timeline.append({"t": t, "event": "lost", "reason": "score_below_threshold", "score": best["score"] if best else None, "threshold": unlock_threshold})
                elif (t - lost_since) > params.lost_timeout:
                    state = "SEARCHING"
                    lock_state = "UNLOCKED"
                    locked_track_id = None
                    locked_since = None
                    reacquire_until = t + float(setup.get("reacquire_window_seconds", params.reacquire_window_seconds))
                    timeline.append({"t": t, "event": "unlock", "reason": "lost_timeout", "lost_for": t - lost_since, "score": best["score"] if best else None, "threshold": unlock_threshold})
                    timeline.append({"t": t, "event": "reacquire", "reason": "window_open", "until": reacquire_until})
            else:
                if t > reacquire_until:
                    lost_since = None

            allow_unconfirmed = bool(setup.get("allow_unconfirmed_clips", params.allow_unconfirmed_clips))
            clip_identity_threshold = 0.60
            identity_score = best["score"] if best else 0.0
            locked_for = (t - locked_since) if (locked_since is not None and state == "LOCKED") else 0.0
            min_track_seconds = float(setup.get("min_track_seconds", params.min_track_seconds))
            if state != "LOCKED":
                present = False
                clip_reason = "state_not_locked"
            elif identity_score < clip_identity_threshold:
                present = False
                clip_reason = "identity_below_threshold"
            elif locked_for < min_track_seconds:
                present = False
                clip_reason = "min_track_not_met"
            elif player_num and lock_state == "PROVISIONAL" and not allow_unconfirmed:
                present = False
                clip_reason = "provisional_not_allowed"
            else:
                present = True
                clip_reason = "locked_identity_ok"

            clip_event = {
                "t": t,
                "event": "clip_allowed" if present else "clip_blocked",
                "reason": clip_reason,
                "identity_score": identity_score,
                "lock_state": lock_state,
                "state": state,
                "identity_threshold": clip_identity_threshold,
                "min_track_seconds": min_track_seconds,
                "score_unlock_threshold": unlock_threshold,
                "allow_unconfirmed_clips": allow_unconfirmed,
            }
            if not timeline or timeline[-1].get("event") != clip_event["event"] or timeline[-1].get("reason") != clip_reason:
                timeline.append(clip_event)
            if present and not present_prev:
                seg_start = t
                if shift_state == "OFF_ICE":
                    shift_state = "ON_ICE"
                    shift_start = t
            if (not present) and present_prev:
                seg_end = t
                segments.append((seg_start, seg_end, "unlock"))
                if shift_state == "ON_ICE" and shift_start is not None:
                    shifts.append((shift_start, seg_end))
                    shift_state = "OFF_ICE"
                    shift_start = None

            if prev_state != state:
                timeline.append({"t": t, "event": "state", "from": prev_state, "to": state})
            if prev_lock_state != lock_state:
                timeline.append({"t": t, "event": "lock_state", "from": prev_lock_state, "to": lock_state})

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
        segments.append((seg_start, eof_t, "eof"))
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

    merged = []
    for a, b, reason in sorted(segments, key=lambda x: x[0]):
        if (b - a) < params.min_clip_seconds:
            continue
        if merged and a - merged[-1][1] <= params.gap_merge_seconds:
            merged_reason = "seed_clip" if (reason == "seed_clip" or merged[-1][2] == "seed_clip") else "merged"
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), merged_reason)
        else:
            merged.append((a, b, reason))

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
    setup_path.write_text(json.dumps(setup, indent=2))

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

        write_status("processing", "tracking", 12, "Tracking in progress")
        data = track_presence(in_path, setup, heartbeat=hb, cancel_check=cancel_check)

        clips_dir = job_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        write_status("processing", "clips", 72, "Creating clips")

        clips = []
        clip_paths = []
        segment_count = max(1, len(data["segments"]))
        for i, (a, b, _) in enumerate(data["segments"], start=1):
            outp = clips_dir / f"clip_{i:03d}.mp4"
            cut_clip(in_path, a, b + float(setup.get("post_roll", setup.get("extend_sec", 2.0))), str(outp))
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
            (job_dir / "debug.json").write_text(json.dumps(data["timeline"], indent=2))

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
        meta.update({
            "status": "failed",
            "stage": "failed",
            "progress": 100,
            "message": err,
            "error": ffmpeg_stderr or traceback.format_exc(),
            "updated_at": time.time(),
        })
        meta_path.write_text(json.dumps(meta, indent=2))
        (job_dir / "results.json").write_text(json.dumps(meta, indent=2))
        if cur:
            cur.meta = {**(cur.meta or {}), "stage": "failed", "progress": 100, "message": err}
            cur.save_meta()
        log.error("job_id=%s failed stage=%s error=%s", job_id, stage['name'], err)
        raise
    finally:
        stop_watchdog = True


def self_test_task(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"ok": True, "payload": payload or {}, "worker_pid": os.getpid(), "updated_at": time.time()}
