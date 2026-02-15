import os
import json
import time
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ultralytics provides built-in tracking (ByteTrack/BoT-SORT configs).
from ultralytics import YOLO

# Optional OCR (used as "confirmation / re-acquire", not every frame)
try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None  # type: ignore


# ----------------------------
# Paths / job helpers
# ----------------------------

BASE = Path(__file__).resolve().parents[1]  # .../Projects
DATA_DIR = BASE / "data" / "jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_MODEL = None
_OCR = None


def job_dir(job_id: str) -> Path:
    return DATA_DIR / job_id


def read_json(p: Path, default: Any) -> Any:
    if not p.exists():
        return default
    return json.loads(p.read_text())


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def set_status(
    job_id: str,
    status: str,
    *,
    stage: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    jd = job_dir(job_id)
    stp = jd / "status.json"
    prev = read_json(stp, {})
    now = int(time.time())
    prev.update(
        {
            "job_id": job_id,
            "status": status,
            "updated_at": now,
        }
    )
    if stage is not None:
        prev["stage"] = stage
    if progress is not None:
        prev["progress"] = progress
    if message is not None:
        prev["message"] = message
    if extra:
        prev.update(extra)
    write_json(stp, prev)


# ----------------------------
# Video helpers (ffmpeg)
# ----------------------------

def run(cmd: List[str]) -> None:
    subprocess.check_call(cmd)


def ffprobe_meta(path: str) -> Dict[str, Any]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            path,
        ],
        text=True,
    )
    return json.loads(out)


def video_duration_seconds(path: str) -> float:
    meta = ffprobe_meta(path)
    if "format" in meta and "duration" in meta["format"]:
        return float(meta["format"]["duration"])
    # fallback
    for s in meta.get("streams", []):
        if s.get("codec_type") == "video" and "duration" in s:
            return float(s["duration"])
    raise RuntimeError("Could not determine video duration")


def make_proxy(in_path: str, out_path: str) -> None:
    # Proxy for web playback + faster seeking.
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            in_path,
            "-vf",
            "scale=1280:-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            out_path,
        ]
    )


def ffmpeg_extract_clip(src: str, start: float, end: float, dst: str) -> None:
    start = max(0.0, float(start))
    end = max(start + 0.001, float(end))
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            src,
            "-c",
            "copy",
            dst,
        ]
    )


def concat_clips(clip_paths: List[str], out_path: str) -> None:
    tmp_list = Path(out_path).with_suffix(".txt")
    tmp_list.write_text("".join([f"file '{p}'\n" for p in clip_paths]))
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(tmp_list),
            "-c",
            "copy",
            out_path,
        ]
    )
    tmp_list.unlink(missing_ok=True)


# ----------------------------
# Tracking config
# ----------------------------

@dataclass
class TrackConfig:
    # Detection
    imgsz: int = 960
    yolo_conf: float = 0.25

    # Tracking
    tracker_cfg: str = "bytetrack.yaml"  # built-in ultralytics tracker configs

    # Presence logic
    detect_stride: int = 2        # sample every N frames (lower = more accurate, slower)
    gap_merge: float = 1.25       # seconds: merge small gaps inside a shift
    min_shift_len: float = 4.0    # seconds: ignore tiny detections
    preroll: float = 2.5          # seconds before presence start
    postroll: float = 1.75        # seconds after presence end

    # Re-acquire logic
    max_lost_seconds: float = 3.0
    dist_gate_norm: float = 0.22  # normalized distance gate for re-acquire
    color_threshold: float = 1.03 # jersey match must beat opponent-ish

    # OCR logic (only run when bbox large enough and not too often)
    ocr_min_height_px: int = 90
    ocr_every_n_hits: int = 20    # run OCR every N "present" hits


CFG = TrackConfig()


def get_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        # A good default. Swap to yolov8m.pt / yolov8l.pt for more accuracy (slower).
        _MODEL = YOLO("yolov8s.pt")
    return _MODEL


def get_ocr():
    global _OCR
    if _OCR is None and easyocr is not None:
        # English digits is enough; we whitelist digits later.
        _OCR = easyocr.Reader(["en"], gpu=True)
    return _OCR


# ----------------------------
# Color + OCR utilities
# ----------------------------

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hc = hex_color.strip().lstrip("#")
    if len(hc) != 6:
        return (0, 0, 0)
    r = int(hc[0:2], 16)
    g = int(hc[2:4], 16)
    b = int(hc[4:6], 16)
    return (b, g, r)


def mean_color_score(frame_bgr: np.ndarray, box_xyxy: np.ndarray, jersey_hex: str, opp_hex: Optional[str]) -> float:
    """
    Returns a score >1.0 if crop looks closer to jersey than to opponent/neutral.
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return 0.0

    crop = frame_bgr[y1:y2, x1:x2]
    # Use upper-body-ish area (jersey more likely than skates/ice)
    hh = crop.shape[0]
    crop = crop[int(hh * 0.15): int(hh * 0.70), :]

    if crop.size == 0:
        return 0.0

    mean = crop.reshape(-1, 3).mean(axis=0)  # BGR

    jersey = np.array(hex_to_bgr(jersey_hex), dtype=np.float32)
    d_jersey = np.linalg.norm(mean - jersey) + 1e-6

    if opp_hex:
        opp = np.array(hex_to_bgr(opp_hex), dtype=np.float32)
        d_opp = np.linalg.norm(mean - opp) + 1e-6
        # score >1 means closer to jersey than opponent
        return float(d_opp / d_jersey)
    else:
        # compare against a neutral mid-gray
        neutral = np.array([128, 128, 128], dtype=np.float32)
        d_neu = np.linalg.norm(mean - neutral) + 1e-6
        return float(d_neu / d_jersey)


def ocr_digits(frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> str:
    reader = get_ocr()
    if reader is None:
        return ""

    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return ""

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.shape[0] < CFG.ocr_min_height_px:
        return ""

    # Focus on chest region (numbers)
    hh = crop.shape[0]
    crop = crop[int(hh * 0.18): int(hh * 0.75), :]

    # Preprocess a bit
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results = reader.readtext(bw, detail=0, allowlist="0123456789")
    if not results:
        return ""
    # Join and keep digits
    s = "".join(results)
    s = "".join([c for c in s if c.isdigit()])
    return s


def norm_center_dist(a_xyxy: np.ndarray, b_xyxy: np.ndarray, W: int, H: int) -> float:
    ax = (a_xyxy[0] + a_xyxy[2]) / 2.0
    ay = (a_xyxy[1] + a_xyxy[3]) / 2.0
    bx = (b_xyxy[0] + b_xyxy[2]) / 2.0
    by = (b_xyxy[1] + b_xyxy[3]) / 2.0
    return float(math.hypot(ax - bx, ay - by) / (math.hypot(W, H) + 1e-6))


# ----------------------------
# "Pro" tracking core
# ----------------------------

def pick_target_track_id_from_clicks(
    video_path: str,
    clicks: List[Dict[str, float]],
    jersey_hex: str,
    opponent_hex: Optional[str],
    target_number: Optional[str],
) -> int:
    """
    Uses a short tracking run around click timestamps to find the most consistent track id.
    """
    model = get_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # We'll evaluate each click by seeking to its frame and grabbing ~1 second of frames for stable IDs.
    votes: Dict[int, float] = {}

    for c in clicks:
        t = float(c["t"])
        cx = float(c["x"]) * W
        cy = float(c["y"]) * H
        f0 = int(t * fps)
        f0 = max(0, min(total_frames - 2, f0))

        # window of frames around click
        start = max(0, f0 - int(0.2 * fps))
        end = min(total_frames - 1, f0 + int(0.9 * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        frame_idxs = []
        for fi in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            frame_idxs.append(fi)

        if not frames:
            continue

        # Run tracking on these frames
        # We use predict on numpy frames with persist, same tracker, to keep IDs stable in window
        results_iter = model.track(
            source=frames,
            stream=True,
            persist=True,
            tracker=CFG.tracker_cfg,
            imgsz=CFG.imgsz,
            conf=CFG.yolo_conf,
            classes=[0],  # person
            verbose=False,
        )

        for frame, r in zip(frames, results_iter):
            if r.boxes is None or r.boxes.xyxy is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = None
            try:
                ids = r.boxes.id
            except Exception:
                ids = None
            if ids is None:
                continue
            ids = ids.cpu().numpy().astype(int)

            # pick any track whose bbox contains click point
            for box, tid in zip(xyxy, ids):
                x1, y1, x2, y2 = box
                if cx >= x1 and cx <= x2 and cy >= y1 and cy <= y2:
                    cs = mean_color_score(frame, box, jersey_hex, opponent_hex)
                    if cs < CFG.color_threshold:
                        continue

                    bonus = 1.0
                    if target_number:
                        s = ocr_digits(frame, box)
                        if s == target_number:
                            bonus = 3.0  # strong vote
                        elif s and target_number in s:
                            bonus = 2.0

                    votes[tid] = votes.get(tid, 0.0) + (1.0 * cs * bonus)

    cap.release()

    if not votes:
        # fallback: no click matched, choose nothing -> will fail loudly
        raise RuntimeError("Could not identify target from clicks. Try clicking on the player when clearly visible.")

    # best track id
    return max(votes.items(), key=lambda kv: kv[1])[0]


def track_presence_spans(
    video_path: str,
    target_id: int,
    jersey_hex: str,
    opponent_hex: Optional[str],
    target_number: Optional[str],
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    Run tracking through the full video; mark presence when target_id is present.
    Includes light re-acquire logic using color + distance + optional OCR.
    """
    model = get_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else video_duration_seconds(video_path)

    # Build a frame generator with stride (so tracking is feasible).
    def frame_gen():
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi % CFG.detect_stride == 0:
                yield frame
            fi += 1

    present_times: List[float] = []
    last_box: Optional[np.ndarray] = None
    last_seen_t: Optional[float] = None
    hits_since_ocr = 0
    confirmed_number_hits = 0

    # tracking stream
    results_iter = model.track(
        source=frame_gen(),
        stream=True,
        persist=True,
        tracker=CFG.tracker_cfg,
        imgsz=CFG.imgsz,
        conf=CFG.yolo_conf,
        classes=[0],
        verbose=False,
    )

    # We need our own time counter (because stride skips frames)
    sampled_index = 0
    sampled_dt = (CFG.detect_stride / fps)

    for r in results_iter:
        t = sampled_index * sampled_dt
        sampled_index += 1

        chosen_box = None
        chosen_tid = None

        if r.boxes is not None and r.boxes.xyxy is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = None
            try:
                ids = r.boxes.id
            except Exception:
                ids = None
            if ids is not None:
                ids = ids.cpu().numpy().astype(int)

                # 1) direct hit: same track id
                for box, tid in zip(xyxy, ids):
                    if tid == target_id:
                        chosen_box = box
                        chosen_tid = tid
                        break

                # 2) reacquire: if missing too long, try best candidate near last_box
                if chosen_box is None and last_box is not None and last_seen_t is not None:
                    if (t - last_seen_t) <= CFG.max_lost_seconds:
                        best = None
                        best_score = 0.0
                        for box, tid in zip(xyxy, ids):
                            cs = mean_color_score(r.orig_img, box, jersey_hex, opponent_hex)  # type: ignore
                            if cs < CFG.color_threshold:
                                continue
                            dist = norm_center_dist(last_box, box, W, H)
                            if dist > CFG.dist_gate_norm:
                                continue

                            score = cs * (1.0 / (1e-6 + dist))
                            best = (box, tid)
                            best_score = score if score > best_score else best_score

                        if best is not None:
                            chosen_box, chosen_tid = best

                            # Optional: OCR confirms if large enough (not every frame)
                            if target_number and (hits_since_ocr >= CFG.ocr_every_n_hits):
                                s = ocr_digits(r.orig_img, chosen_box)  # type: ignore
                                hits_since_ocr = 0
                                if s == target_number or (s and target_number in s):
                                    confirmed_number_hits += 1

        if chosen_box is not None:
            # record presence
            present_times.append(t)
            last_box = chosen_box
            last_seen_t = t
            hits_since_ocr += 1
        else:
            # not present
            pass

    cap.release()

    # Build spans from present_times
    spans: List[Tuple[float, float]] = []
    if present_times:
        start = present_times[0]
        prev = present_times[0]
        for tt in present_times[1:]:
            if tt - prev <= CFG.gap_merge:
                prev = tt
            else:
                spans.append((start, prev))
                start = tt
                prev = tt
        spans.append((start, prev))

    # Expand with preroll/postroll, filter short
    expanded: List[Tuple[float, float]] = []
    for a, b in spans:
        a2 = max(0.0, a - CFG.preroll)
        b2 = min(duration_s, b + CFG.postroll)
        if (b2 - a2) >= CFG.min_shift_len:
            expanded.append((a2, b2))

    # Merge overlaps after expansion
    merged: List[Tuple[float, float]] = []
    for a, b in sorted(expanded):
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    debug = {
        "fps": fps,
        "W": W,
        "H": H,
        "duration_s": duration_s,
        "detect_stride": CFG.detect_stride,
        "gap_merge": CFG.gap_merge,
        "preroll": CFG.preroll,
        "postroll": CFG.postroll,
        "min_shift_len": CFG.min_shift_len,
        "color_threshold": CFG.color_threshold,
        "dist_gate_norm": CFG.dist_gate_norm,
        "max_lost_seconds": CFG.max_lost_seconds,
        "confirmed_number_hits": confirmed_number_hits,
        "target_track_id": target_id,
        "present_samples": len(present_times),
        "span_count": len(merged),
    }
    return merged, debug


# ----------------------------
# Main RQ entrypoint
# ----------------------------

def process_job(job_id: str) -> Dict[str, Any]:
    jd = job_dir(job_id)
    in_path = jd / "in.mp4"
    setup_path = jd / "setup.json"
    if not in_path.exists():
        raise RuntimeError("Input video not found for job")

    setup = read_json(setup_path, {})
    camera_mode = setup.get("camera_mode", "broadcast")

    # Inputs from UI
    target_number = str(setup.get("player_number", "")).strip()
    if target_number == "":
        target_number = None

    jersey_hex = setup.get("jersey_color", "#203524") or "#203524"
    opponent_hex = setup.get("opponent_color", None)  # optional but recommended

    clicks = setup.get("clicks", []) or []
    clicks_count = len(clicks)

    # You wanted 2–3 clicks max: we allow 1, but warn.
    if clicks_count < 1:
        raise RuntimeError("Need at least 1 click to seed the player. Recommended: 2–3 clicks.")

    set_status(job_id, "processing", stage="proxy", progress=5, message="Preparing proxy...")
    proxy_path = jd / "input_proxy.mp4"
    if not proxy_path.exists():
        make_proxy(str(in_path), str(proxy_path))

    set_status(job_id, "processing", stage="tracking", progress=20, message="Tracking player...")

    # Step 1: Pick target track id using clicks + color + (optional) OCR.
    target_id = pick_target_track_id_from_clicks(
        str(proxy_path),
        clicks=clicks,
        jersey_hex=jersey_hex,
        opponent_hex=opponent_hex,
        target_number=target_number,
    )

    # Step 2: Track through full video and build shift spans.
    spans, debug = track_presence_spans(
        str(proxy_path),
        target_id=target_id,
        jersey_hex=jersey_hex,
        opponent_hex=opponent_hex,
        target_number=target_number,
    )

    if not spans:
        set_status(
            job_id,
            "done",
            stage="done",
            progress=100,
            message="No shifts found (player never confidently detected). Try 2–3 clicks when clearly visible.",
            extra={
                "proxy_ready": True,
                "proxy_path": str(proxy_path),
                "proxy_url": f"/data/jobs/{job_id}/input_proxy.mp4",
                "debug": debug,
            },
        )
        out = read_json(jd / "status.json", {})
        write_json(jd / "results.json", out)
        return out

    set_status(job_id, "processing", stage="clipping", progress=70, message=f"Exporting {len(spans)} clips...")

    clips_dir = jd / "clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    for i, (a, b) in enumerate(spans, start=1):
        outp = clips_dir / f"clip_{i:03d}.mp4"
        ffmpeg_extract_clip(str(in_path), a, b, str(outp))  # clip from original for best quality
        clips.append(
            {
                "start": float(a),
                "end": float(b),
                "path": str(outp),
                "url": f"/data/jobs/{job_id}/clips/{outp.name}",
            }
        )

    combined_path = jd / "combined.mp4"
    try:
        concat_clips([c["path"] for c in clips], str(combined_path))
        combined_url = f"/data/jobs/{job_id}/combined.mp4"
    except Exception:
        combined_url = None

    set_status(
        job_id,
        "done",
        stage="done",
        progress=100,
        message="Done.",
        extra={
            "proxy_ready": True,
            "proxy_path": str(proxy_path),
            "proxy_url": f"/data/jobs/{job_id}/input_proxy.mp4",
            "clips": clips,
            "combined_path": str(combined_path),
            "combined_url": combined_url,
            "setup": setup,
            "clicks_count": clicks_count,
            "debug": debug,
        },
    )

    out = read_json(jd / "status.json", {})
    write_json(jd / "results.json", out)
    return out

