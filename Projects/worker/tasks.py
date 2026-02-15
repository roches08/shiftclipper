import os
import json
import math
import time
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# Paths / storage
# =========================

DATA_DIR = Path(os.environ.get("SHIFTCLIPPER_DATA_DIR", "Projects/data")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Presets (Speed vs Accuracy)
# =========================

PRESETS = {
    # MAX ACCURACY (default)
    "accuracy": {
        "yolo_weights": os.environ.get("YOLO_WEIGHTS", "yolov8x.pt"),
        "yolo_imgsz": int(os.environ.get("YOLO_IMGSZ", "960")),
        "yolo_conf": float(os.environ.get("YOLO_CONF", "0.25")),
        "detect_stride": int(os.environ.get("DETECT_STRIDE", "1")),  # stride=1 = detect every frame
        "roi_pad_locked": 0.55,
        "roi_pad_lost": 0.95,
        "lost_reacquire_after": 0.55,  # seconds
        "global_reacquire_every": 2.0,  # seconds
        "sticky_seconds": 1.2,
        "scene_cut_thresh": 0.22,
        "min_clip_len": 3.0,
        "gap_merge": 1.25,
        "pre_roll": 1.25,
        "post_roll": 0.90,
        "seed_window_s": 0.35,  # multi-frame sampling around each click
        "seed_offsets": (-0.33, 0.0, 0.33),
        "present_score_thresh": 0.60,
        "max_area_factor": 2.8,  # relative to seed median
        "min_area_factor": 0.35,
        "max_aspect_delta": 0.55,
    },

    # Faster, less accurate
    "speed": {
        "yolo_weights": os.environ.get("YOLO_WEIGHTS", "yolov8s.pt"),
        "yolo_imgsz": int(os.environ.get("YOLO_IMGSZ", "640")),
        "yolo_conf": float(os.environ.get("YOLO_CONF", "0.30")),
        "detect_stride": int(os.environ.get("DETECT_STRIDE", "2")),
        "roi_pad_locked": 0.55,
        "roi_pad_lost": 0.90,
        "lost_reacquire_after": 0.60,
        "global_reacquire_every": 2.0,
        "sticky_seconds": 1.2,
        "scene_cut_thresh": 0.22,
        "min_clip_len": 3.0,
        "gap_merge": 1.25,
        "pre_roll": 1.25,
        "post_roll": 0.90,
        "seed_window_s": 0.30,
        "seed_offsets": (0.0,),
        "present_score_thresh": 0.62,
        "max_area_factor": 3.2,
        "min_area_factor": 0.30,
        "max_aspect_delta": 0.70,
    },
}

def _preset() -> Dict[str, Any]:
    name = os.environ.get("SHIFTCLIPPER_PRESET", "accuracy").strip().lower()
    return PRESETS.get(name, PRESETS["accuracy"])

# =========================
# Utilities
# =========================

def _job_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "clips").mkdir(parents=True, exist_ok=True)
    return d

def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())

def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2))

def _set_status(job_id: str, **fields: Any) -> None:
    d = _job_dir(job_id)
    status_path = d / "status.json"
    cur = _read_json(status_path, {})
    cur.update(fields)
    cur["job_id"] = job_id
    cur["updated_at"] = time.time()
    _write_json(status_path, cur)

def _ffmpeg_cut(in_path: str, out_path: str, start: float, end: float) -> None:
    # -ss before -i is faster but less accurate; for accuracy use after -i
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c", "copy",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _ffmpeg_concat(file_list_path: str, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", file_list_path,
        "-c", "copy",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    s = (hex_color or "").strip().lstrip("#")
    if len(s) != 6:
        return (0, 0, 0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)

def _bgr_to_hsv_color(bgr: Tuple[int, int, int]) -> np.ndarray:
    bgr_arr = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    return hsv

def _hsv_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Hue wrap-around
    dh = min(abs(a[0] - b[0]), 180.0 - abs(a[0] - b[0])) / 90.0
    ds = abs(a[1] - b[1]) / 255.0
    dv = abs(a[2] - b[2]) / 255.0
    return float(dh + 0.6 * ds + 0.4 * dv)

def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)

def _torso_crop(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    # torso-ish: upper-middle region
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    tx1 = x1 + int(0.15 * w)
    tx2 = x2 - int(0.15 * w)
    ty1 = y1 + int(0.18 * h)
    ty2 = y1 + int(0.72 * h)
    tx1, ty1, tx2, ty2 = _clip_box(tx1, ty1, tx2, ty2, frame.shape[1], frame.shape[0])
    return frame[ty1:ty2, tx1:tx2]

def _mean_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # robust mean: ignore very dark pixels (ice/shadows) a bit
    v = hsv[:, :, 2].astype(np.float32)
    mask = (v > 35).astype(np.uint8)
    if mask.sum() < 50:
        m = hsv.reshape(-1, 3).mean(axis=0)
        return m.astype(np.float32)
    m = cv2.mean(hsv, mask=mask)[0:3]
    return np.array(m, dtype=np.float32)

# =========================
# YOLO
# =========================

_MODEL: Optional[YOLO] = None
_MODEL_INFO: Dict[str, Any] = {}

def _load_model() -> YOLO:
    global _MODEL, _MODEL_INFO
    p = _preset()
    weights = p["yolo_weights"]
    if _MODEL is None or _MODEL_INFO.get("weights") != weights:
        _MODEL = YOLO(weights)
        _MODEL_INFO = {"weights": weights}
    return _MODEL

def _yolo_person_boxes(frame_bgr: np.ndarray, conf: float, imgsz: int) -> List[Tuple[int, int, int, int, float]]:
    model = _load_model()
    # Ultralytics class 0 == person (COCO)
    results = model.predict(
        source=frame_bgr,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else None
    )
    out: List[Tuple[int, int, int, int, float]] = []
    if not results:
        return out
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return out

    h, w = frame_bgr.shape[:2]
    for b in r.boxes:
        cls = int(b.cls.item())
        if cls != 0:
            continue
        xyxy = b.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy.tolist()
        score = float(b.conf.item())
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)
        out.append((x1, y1, x2, y2, score))
    return out

# =========================
# Candidate scoring
# =========================

@dataclass
class SeedProfile:
    hsv: np.ndarray
    area_med: float
    aspect_med: float

def _box_area(b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    return float(max(1, x2 - x1) * max(1, y2 - y1))

def _box_aspect(b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return float(w) / float(h)

def _center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _score_candidate(
    frame_bgr: np.ndarray,
    cand: Tuple[int, int, int, int, float],
    click_xy: Optional[Tuple[float, float]],
    seed: SeedProfile,
    jersey_hsv: np.ndarray,
    prefer_green: bool,
    p: Dict[str, Any],
) -> float:
    x1, y1, x2, y2, det_conf = cand
    box = (x1, y1, x2, y2)

    area = _box_area(box)
    aspect = _box_aspect(box)

    # Seed-based gating (prevents giant wrong locks)
    a_min = seed.area_med * p["min_area_factor"]
    a_max = seed.area_med * p["max_area_factor"]
    if not (a_min <= area <= a_max):
        return -1.0

    if abs(aspect - seed.aspect_med) > p["max_aspect_delta"]:
        return -1.0

    # Color similarity on torso crop
    torso = _torso_crop(frame_bgr, box)
    hsv_mean = _mean_hsv(torso)
    d_seed = _hsv_distance(hsv_mean, seed.hsv)

    # Jersey color preference (your green)
    d_jersey = _hsv_distance(hsv_mean, jersey_hsv)
    jersey_score = math.exp(-2.4 * d_jersey)

    # Click proximity (only during seed selection)
    prox = 0.0
    if click_xy is not None:
        cx, cy = _center(box)
        dx = (cx - click_xy[0]) / max(1.0, frame_bgr.shape[1])
        dy = (cy - click_xy[1]) / max(1.0, frame_bgr.shape[0])
        dist = math.sqrt(dx * dx + dy * dy)
        prox = math.exp(-6.0 * dist)

    # Combine:
    # - detection confidence matters
    # - must look like the seed (d_seed)
    # - prefer jersey color strongly (green)
    seed_score = math.exp(-2.2 * d_seed)

    # If you picked a jersey color, weight it hard.
    # prefer_green just means "we have a jersey target", not literally green.
    w_color = 0.55 if prefer_green else 0.35

    score = (
        0.30 * det_conf +
        0.35 * seed_score +
        w_color * jersey_score +
        0.10 * prox
    )
    return float(score)

# =========================
# Seeding from 1–3 clicks (multi-frame sampling)
# =========================

def _frame_at(cap: cv2.VideoCapture, t: float, fps: float, total_frames: int) -> Tuple[Optional[np.ndarray], int]:
    frame_idx = int(round(t * fps))
    frame_idx = max(0, min(total_frames - 1, frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, frame_idx
    return frame, frame_idx

def _seed_from_clicks(video_path: str, clicks: List[Dict[str, Any]], jersey_hex: str) -> Dict[str, Any]:
    p = _preset()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    jersey_hsv = _bgr_to_hsv_color(_hex_to_bgr(jersey_hex))
    prefer_jersey = True if jersey_hex else False

    picked_boxes: List[Tuple[int, int, int, int]] = []
    picked_hsvs: List[np.ndarray] = []
    seed_frames_debug: List[Dict[str, Any]] = []

    conf = float(p["yolo_conf"])
    imgsz = int(p["yolo_imgsz"])

    offsets = p["seed_offsets"]

    # Use each click, but sample a few frames around it and pick best candidate.
    for c in clicks:
        t0 = float(c["t"])
        x_norm = float(c["x"])
        y_norm = float(c["y"])
        click_px = (x_norm * W, y_norm * H)

        best = None
        best_score = -1.0
        best_info = None

        for dt in offsets:
            t = max(0.0, t0 + float(dt))
            frame, fi = _frame_at(cap, t, fps, total_frames)
            if frame is None:
                continue

            boxes = _yolo_person_boxes(frame, conf=conf, imgsz=imgsz)
            if not boxes:
                seed_frames_debug.append({"t": t, "frame": fi, "dets": 0, "picked": None})
                continue

            # Temporary seed estimate if we have prior picks; otherwise allow broader.
            if picked_boxes:
                areas = [ _box_area(b) for b in picked_boxes ]
                aspects = [ _box_aspect(b) for b in picked_boxes ]
                seed_prof = SeedProfile(
                    hsv=np.mean(np.stack(picked_hsvs), axis=0),
                    area_med=float(np.median(areas)),
                    aspect_med=float(np.median(aspects)),
                )
            else:
                # Loose initial seed: use jersey color as initial "seed"
                seed_prof = SeedProfile(
                    hsv=jersey_hsv,
                    area_med=(W * H) * 0.02,   # loose guess
                    aspect_med=0.42,           # upright-ish
                )

            for (x1, y1, x2, y2, det_conf) in boxes:
                score = _score_candidate(
                    frame,
                    (x1, y1, x2, y2, det_conf),
                    click_xy=click_px,
                    seed=seed_prof,
                    jersey_hsv=jersey_hsv,
                    prefer_green=prefer_jersey,
                    p=p
                )
                if score > best_score:
                    best_score = score
                    best = (x1, y1, x2, y2)
                    best_info = {"t": t, "frame": fi, "dets": len(boxes), "picked": [x1, y1, x2, y2], "score": score}

            seed_frames_debug.append(best_info or {"t": t, "frame": fi, "dets": len(boxes), "picked": None})

        if best is not None:
            torso = _torso_crop(frame, best)  # type: ignore
            hsv_mean = _mean_hsv(torso)
            picked_boxes.append(best)
            picked_hsvs.append(hsv_mean)

    cap.release()

    if len(picked_boxes) == 0:
        return {
            "ok": False,
            "reason": "No seed boxes found from clicks.",
            "seed_count": 0,
            "fps": fps,
            "W": W,
            "H": H,
            "seed_frames": seed_frames_debug
        }

    areas = [ _box_area(b) for b in picked_boxes ]
    aspects = [ _box_aspect(b) for b in picked_boxes ]

    seed_prof = {
        "ok": True,
        "seed_bbox": list(map(int, picked_boxes[-1])),
        "seed_count": len(picked_boxes),
        "seed_hsv": np.mean(np.stack(picked_hsvs), axis=0).tolist(),
        "seed_area_med": float(np.median(areas)),
        "seed_aspect_med": float(np.median(aspects)),
        "fps": fps,
        "W": W,
        "H": H,
        "seed_frames": seed_frames_debug
    }
    return seed_prof

# =========================
# Presence spans
# =========================

def _build_presence_spans(times_present: List[float], gap_merge: float, min_len: float) -> List[Tuple[float, float]]:
    if not times_present:
        return []
    times_present = sorted(times_present)
    spans: List[Tuple[float, float]] = []
    start = times_present[0]
    prev = times_present[0]
    for t in times_present[1:]:
        if t - prev <= gap_merge:
            prev = t
            continue
        if (prev - start) >= min_len:
            spans.append((start, prev))
        start = t
        prev = t
    if (prev - start) >= min_len:
        spans.append((start, prev))
    return spans

# =========================
# Main job processing
# =========================

def process_job(job_id: str) -> Dict[str, Any]:
    d = _job_dir(job_id)
    status_path = d / "status.json"
    status = _read_json(status_path, {})
    setup = status.get("setup", {}) or {}

    video_path = status.get("video_path") or str(d / "input.mov")
    proxy_path = status.get("proxy_path")

    camera_mode = setup.get("camera_mode", "broadcast")
    player_number = setup.get("player_number", "")
    jersey_color = setup.get("jersey_color", "")
    clicks = setup.get("clicks", []) or []
    extend_sec = float(setup.get("extend_sec", 20))
    verify_mode = bool(setup.get("verify_mode", True))

    p = _preset()

    _set_status(job_id,
        status="running",
        stage="seeding",
        progress=5,
        message="Seeding from clicks..."
    )

    seed = _seed_from_clicks(video_path, clicks, jersey_color)
    if not seed.get("ok"):
        _set_status(job_id,
            status="done",
            stage="done",
            progress=100,
            message=f"Done (failed to seed): {seed.get('reason','seed failed')}",
            clips=[],
            debug={"seed": seed}
        )
        return _read_json(status_path, {})

    seed_prof = SeedProfile(
        hsv=np.array(seed["seed_hsv"], dtype=np.float32),
        area_med=float(seed["seed_area_med"]),
        aspect_med=float(seed["seed_aspect_med"]),
    )
    jersey_hsv = _bgr_to_hsv_color(_hex_to_bgr(jersey_color)) if jersey_color else seed_prof.hsv

    _set_status(job_id,
        stage="tracking",
        progress=10,
        message="Tracking..."
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or float(seed.get("fps", 30.0))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or int(seed.get("W", 0)))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or int(seed.get("H", 0)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (total_frames / fps) if total_frames > 0 else 0.0

    # Tracking state
    last_box: Optional[Tuple[int, int, int, int]] = tuple(seed["seed_bbox"])  # type: ignore
    last_seen_t: Optional[float] = None
    present_times: List[float] = []
    debug_samples: List[Dict[str, Any]] = []

    detect_stride = int(p["detect_stride"])
    conf = float(p["yolo_conf"])
    imgsz = int(p["yolo_imgsz"])

    roi_pad_locked = float(p["roi_pad_locked"])
    roi_pad_lost = float(p["roi_pad_lost"])
    lost_reacquire_after = float(p["lost_reacquire_after"])
    global_reacquire_every = float(p["global_reacquire_every"])
    present_score_thresh = float(p["present_score_thresh"])

    # For periodic full-frame reacquire
    next_global_reacq_t = 0.0

    def roi_from_box(box: Tuple[int, int, int, int], pad_frac: float) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        px = int(pad_frac * bw)
        py = int(pad_frac * bh)
        rx1 = x1 - px
        ry1 = y1 - py
        rx2 = x2 + px
        ry2 = y2 + py
        return _clip_box(rx1, ry1, rx2, ry2, W, H)

    # iterate frames
    frame_idx = 0
    sampled_detect_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t = frame_idx / fps

        # do detection only on stride
        do_detect = (frame_idx % detect_stride == 0)

        chosen = None
        chosen_score = None
        boxes_n = 0
        used_roi = False
        reason = ""

        # Determine if we consider ourselves "lost"
        lost = False
        if last_seen_t is None:
            lost = True
        else:
            if (t - last_seen_t) >= lost_reacquire_after:
                lost = True

        if do_detect:
            sampled_detect_frames += 1

            # Decide whether to run ROI-only or full-frame
            run_full = False
            if lost:
                run_full = True
                reason = "lost->full"
            elif t >= next_global_reacq_t:
                run_full = True
                reason = "periodic_full"
            else:
                run_full = False
                reason = "roi"

            if run_full or last_box is None:
                dets = _yolo_person_boxes(frame, conf=conf, imgsz=imgsz)
                used_roi = True  # logically "search mode"
            else:
                # ROI detect
                pad = roi_pad_lost if lost else roi_pad_locked
                rx1, ry1, rx2, ry2 = roi_from_box(last_box, pad)
                roi = frame[ry1:ry2, rx1:rx2]
                dets_roi = _yolo_person_boxes(roi, conf=conf, imgsz=imgsz)
                # map back to full coords
                dets = []
                for (x1, y1, x2, y2, sc) in dets_roi:
                    dets.append((x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1, sc))
                used_roi = True

            boxes_n = len(dets)

            # Pick best candidate vs last_box center (if available)
            best_score = -1.0
            best_box = None

            # Click not used during runtime tracking; use last center as "anchor"
            click_xy = None
            if last_box is not None:
                cx, cy = _center(last_box)
                click_xy = (cx, cy)

            for cand in dets:
                score = _score_candidate(
                    frame, cand, click_xy, seed_prof, jersey_hsv, True if jersey_color else False, p
                )
                if score > best_score:
                    best_score = score
                    best_box = (cand[0], cand[1], cand[2], cand[3])

            if best_box is not None and best_score >= present_score_thresh:
                chosen = list(map(int, best_box))
                chosen_score = float(best_score)
                last_box = tuple(best_box)
                last_seen_t = t
                present_times.append(t)

                # schedule next periodic full reacquire
                if reason in ("periodic_full", "lost->full"):
                    next_global_reacq_t = t + global_reacquire_every
            else:
                # no confident match
                chosen = None
                chosen_score = float(best_score) if best_score >= 0 else None

        # store a small debug sample stream (not every frame, to keep JSON sane)
        if frame_idx < 120 or (frame_idx % int(max(1, fps)) == 0):  # first ~2s + then 1/sec
            debug_samples.append({
                "t": float(t),
                "present": bool(last_seen_t is not None and (t - last_seen_t) <= p["sticky_seconds"]),
                "chosen": chosen,
                "score": chosen_score,
                "boxes": int(boxes_n),
                "roi": bool(used_roi),
                "reason": reason,
                "lost_since": None if last_seen_t is None else float(t - last_seen_t),
            })

        # progress update occasionally
        if frame_idx % int(max(1, fps * 2)) == 0 and total_frames > 0:
            prog = 10 + int(85.0 * (frame_idx / max(1, total_frames)))
            _set_status(job_id, progress=min(95, prog), message="Tracking...")

        frame_idx += 1

    cap.release()

    # build spans
    spans = _build_presence_spans(
        times_present=present_times,
        gap_merge=float(p["gap_merge"]),
        min_len=float(p["min_clip_len"])
    )

    # extend spans
    extended: List[Tuple[float, float]] = []
    for s, e in spans:
        ss = max(0.0, s - float(p["pre_roll"]))
        ee = min(duration_s, e + float(p["post_roll"]))
        # optional extend_sec if you want longer segments
        if extend_sec > 0:
            # Only extend "end" by extend_sec if span is short; prevents gigantic clips
            if (ee - ss) < extend_sec:
                ee = min(duration_s, ss + extend_sec)
        extended.append((ss, ee))

    # merge overlaps after extension
    merged: List[Tuple[float, float]] = []
    for s, e in sorted(extended):
        if not merged:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            if s <= pe + 0.05:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))

    # cut clips
    _set_status(job_id, stage="cutting", progress=95, message="Cutting clips...")

    clips = []
    clips_dir = d / "clips"
    for i, (s, e) in enumerate(merged, start=1):
        out_path = clips_dir / f"clip_{i:03d}.mp4"
        try:
            _ffmpeg_cut(video_path, str(out_path), s, e)
            clips.append({
                "start": float(s),
                "end": float(e),
                "path": str(out_path),
                "url": f"/data/jobs/{job_id}/clips/{out_path.name}"
            })
        except Exception as ex:
            # keep going
            clips.append({
                "start": float(s),
                "end": float(e),
                "path": str(out_path),
                "url": f"/data/jobs/{job_id}/clips/{out_path.name}",
                "error": str(ex)
            })

    # concat combined
    combined_path = d / "combined.mp4"
    if clips:
        file_list = d / "concat.txt"
        file_list.write_text("\n".join([f"file '{c['path']}'" for c in clips if "path" in c]) + "\n")
        try:
            _ffmpeg_concat(str(file_list), str(combined_path))
        except Exception:
            # fallback: just copy first clip if concat fails
            shutil.copyfile(clips[0]["path"], combined_path)

    debug = {
        "seed": seed,
        "fps": float(fps),
        "W": int(W),
        "H": int(H),
        "total_frames": int(total_frames),
        "duration_s": float(duration_s),
        "min_clip_len": float(p["min_clip_len"]),
        "gap_merge": float(p["gap_merge"]),
        "sticky_seconds": float(p["sticky_seconds"]),
        "pre_roll": float(p["pre_roll"]),
        "post_roll": float(p["post_roll"]),
        "detect_stride": int(detect_stride),
        "yolo_conf": float(conf),
        "yolo_weights": str(p["yolo_weights"]),
        "yolo_imgsz": int(imgsz),
        "roi_enable": True,
        "roi_pad_frac_locked": float(roi_pad_locked),
        "roi_pad_frac_lost": float(roi_pad_lost),
        "lost_reacquire_after": float(lost_reacquire_after),
        "global_reacquire_every": float(global_reacquire_every),
        "scene_cut_thresh": float(p["scene_cut_thresh"]),
        "sampled_detect_frames": int(sampled_detect_frames),
        "debug_samples": debug_samples,
        "spans": [[float(a), float(b)] for (a, b) in merged],
        "target_track_id": None,
    }

    _set_status(
        job_id,
        status="done",
        stage="done",
        progress=100,
        message="Done.",
        clips=clips,
        combined_path=str(combined_path),
        combined_url=f"/data/jobs/{job_id}/combined.mp4",
        proxy_url=status.get("proxy_url"),
        debug=debug
    )
    return _read_json(status_path, {})


