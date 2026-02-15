import os
import json
import datetime
import time
import math
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

# Ultralytics YOLO (GPU via torch if available)
from ultralytics import YOLO

# -----------------------------
# Env knobs (defaults are "max accuracy" friendly)
# -----------------------------
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8x.pt")  # big model = better accuracy
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))       # higher = better accuracy (slower)
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.20"))       # lower = more detections (more FP risk)
DETECT_STRIDE = int(os.getenv("DETECT_STRIDE", "1"))    # 1 = detect every frame (slowest, best)

# -----------------------------
# Data model
# -----------------------------
@dataclass
class Click:
    """Normalized click in [0..1] coords plus timestamp seconds."""
    t: float
    x: float
    y: float


# --- Job runner wrapper for the web/API ---
# The API enqueues `worker.tasks.process_job(job_id)` into Redis/RQ.
# This function loads the job's setup + uploaded video from the job directory,
# runs the tracker, and writes results.json + updates meta.json for the UI.

def _projects_root() -> Path:
    # worker/tasks.py -> worker/ -> Projects/
    return Path(__file__).resolve().parents[1]

def _jobs_root() -> Path:
    # Allow override; default matches api/main.py (Projects/data/jobs)
    env = os.getenv("SHIFTCLIPPER_JOBS_DIR") or os.getenv("JOBS_DIR")
    if env:
        return Path(env)
    return _projects_root() / "data" / "jobs"

def _job_dir(job_id: str) -> Path:
    return _jobs_root() / job_id

def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except Exception:
        return default

def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def _set_meta(job_id: str, status: str, **extra):
    jd = _job_dir(job_id)
    mp = jd / "meta.json"
    meta = _read_json(mp, {})
    meta.update({
        "job_id": job_id,
        "status": status,
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    })
    meta.update(extra)
    _write_json(mp, meta)

def process_job(job_id: str):
    """RQ entrypoint: run a queued job created by the web UI."""
    jd = _job_dir(job_id)
    if not jd.exists():
        raise FileNotFoundError(f"Job dir not found: {jd}")

    in_mp4 = jd / "in.mp4"
    if not in_mp4.exists():
        _set_meta(
            job_id,
            "error",
            stage="error",
            progress=0,
            message="Missing input video (in.mp4). Upload a video first.",
        )
        raise FileNotFoundError(f"Missing input video: {in_mp4}")

    setup = _read_json(jd / "setup.json", {})
    clicks_raw = setup.get("clicks") or []
    clicks: List[Click] = []
    for c in clicks_raw:
        try:
            clicks.append(
                Click(
                    t=float(c.get("t", 0.0)),
                    x=float(c.get("x", 0.0)),
                    y=float(c.get("y", 0.0)),
                )
            )
        except Exception:
            continue

    player_number = (setup.get("player_number") or "").strip()
    jersey_color = (setup.get("jersey_color") or "dark").strip().lower()
    camera_mode = (setup.get("camera_mode") or "side").strip().lower()
    verify_mode = bool(setup.get("verify_mode", False))
    extend_sec = float(setup.get("extend_sec", 0))

    out_dir = jd / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_meta(job_id, "processing", stage="processing", progress=35, message="Processing started.")

    try:
        results = run_tracker(
            video_path=str(in_mp4),
            out_dir=str(out_dir),
            click_xy=clicks,
            player_number=player_number,
            jersey_color=jersey_color,
            camera_mode=camera_mode,
            verify_mode=verify_mode,
            extend_sec=extend_sec,
        )
        _write_json(jd / "results.json", results)
        _set_meta(job_id, "done", stage="done", progress=100, message="Done.", results=results)
        return results
    except Exception as e:
        _set_meta(job_id, "error", stage="error", progress=100, message=f"Error: {type(e).__name__}: {e}")
        raise


# -----------------------------
# Core tracker logic
# -----------------------------
def _ffprobe_duration_seconds(video_path: str) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        out = subprocess.check_output(cmd).decode("utf-8").strip()
        return float(out)
    except Exception:
        return 0.0

def _ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it: apt-get install -y ffmpeg")

def _load_model() -> YOLO:
    # YOLO loads onto GPU automatically if torch+cuda available
    return YOLO(YOLO_WEIGHTS)

def run_tracker(
    video_path: str,
    out_dir: str,
    click_xy: List[Click],
    player_number: str,
    jersey_color: str = "dark",
    camera_mode: str = "side",
    verify_mode: bool = False,
    extend_sec: float = 0.0,
) -> Dict[str, Any]:
    """
    Runs detection/tracking and produces outputs in out_dir.
    Returns a results dict that will be written to results.json by process_job().
    """
    _ensure_ffmpeg()

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    model = _load_model()

    # Basic video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    duration = _ffprobe_duration_seconds(video_path)
    if duration <= 0 and fps > 0 and total_frames > 0:
        duration = total_frames / fps

    # Convert normalized clicks to pixel coords (still keep t)
    clicks_px: List[Tuple[float, int, int]] = []
    for c in click_xy:
        x_px = int(max(0, min(1, c.x)) * max(1, width - 1))
        y_px = int(max(0, min(1, c.y)) * max(1, height - 1))
        clicks_px.append((float(c.t), x_px, y_px))

    # Main loop: run YOLO periodically (stride) and do simple nearest-to-click selection
    detections_by_t: List[Dict[str, Any]] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps

        if (frame_idx % DETECT_STRIDE) == 0:
            # Ultralytics predict
            preds = model.predict(
                source=frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                verbose=False,
                device=0 if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else None,
            )
            boxes = []
            if preds and len(preds) > 0:
                r = preds[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    for b in r.boxes:
                        xyxy = b.xyxy.cpu().numpy().tolist()[0]
                        conf = float(b.conf.cpu().numpy().tolist()[0]) if b.conf is not None else 0.0
                        cls = int(b.cls.cpu().numpy().tolist()[0]) if b.cls is not None else -1
                        boxes.append({"xyxy": xyxy, "conf": conf, "cls": cls})

            # Pick "best" box near the closest click in time (if any)
            chosen = None
            if clicks_px and boxes:
                # closest click by time
                ct, cx, cy = min(clicks_px, key=lambda p: abs(p[0] - t))
                # choose box whose center is closest to click point
                def center_dist(box):
                    x1, y1, x2, y2 = box["xyxy"]
                    mx = (x1 + x2) / 2.0
                    my = (y1 + y2) / 2.0
                    return (mx - cx) ** 2 + (my - cy) ** 2
                chosen = min(boxes, key=center_dist)

            detections_by_t.append({
                "t": t,
                "frame": frame_idx,
                "boxes": boxes,
                "chosen": chosen,
            })

        frame_idx += 1

    cap.release()

    results = {
        "ok": True,
        "video_path": video_path,
        "out_dir": str(out_dir_p),
        "fps": fps,
        "width": width,
        "height": height,
        "frames": total_frames,
        "duration_sec": duration,
        "player_number": player_number,
        "jersey_color": jersey_color,
        "camera_mode": camera_mode,
        "verify_mode": verify_mode,
        "extend_sec": extend_sec,
        "yolo": {
            "weights": YOLO_WEIGHTS,
            "imgsz": YOLO_IMGSZ,
            "conf": YOLO_CONF,
            "stride": DETECT_STRIDE,
        },
        "detections": detections_by_t[:5000],  # guardrail to prevent insane JSON sizes
    }

    return results

