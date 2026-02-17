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


@dataclass
class TrackingParams:
    detect_stride: int = 1
    ocr_min_conf: float = 0.22
    lock_seconds_after_confirm: float = 4.0
    gap_merge_seconds: float = 2.5
    lost_timeout: float = 1.5
    min_track_seconds: float = 0.75
    post_roll: float = 2.0
    color_weight: float = 0.35
    motion_weight: float = 0.30
    ocr_weight: float = 0.35
    identity_weight: float = 0.5
    color_tolerance: int = 26
    ocr_confirm_m: int = 2
    ocr_confirm_k: int = 5
    bench_zone_ratio: float = 0.8
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


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{err}")


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


def track_presence(video_path: str, setup: Dict[str, Any], heartbeat=None, cancel_check=None):
    if cv2 is None or DeepSort is None or YOLO is None:
        raise RuntimeError("tracking dependencies unavailable")

    params = TrackingParams(**{k: v for k, v in setup.items() if k in TrackingParams.__annotations__})
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov8s.pt")
    device = resolve_device()
    ocr, ocr_gpu = _build_ocr_reader(device)
    tracker = DeepSort(max_age=40, n_init=2, embedder="mobilenet", embedder_gpu=device.startswith("cuda"), bgr=True, half=True)

    jersey_hsv = _hex_to_hsv(setup.get("jersey_color", "#203524"))
    player_num = re.sub(r"\D+", "", str(setup.get("player_number") or ""))

    state = "SEARCHING"
    ocr_window = deque(maxlen=params.ocr_confirm_k)
    locked_until = 0.0
    lost_since = None
    low_color_frames = 0
    locked_track = None
    last_box = None
    last_seen = 0.0

    segments = []
    seg_start = None
    shifts = []
    shift_start = None
    shift_state = "OFF_ICE"

    timeline = []
    overlay_path = None
    writer = None
    if params.debug_overlay:
        overlay_path = str(Path(video_path).with_name("debug_overlay.mp4"))
        writer = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    last_hb = 0.0
    while True:
        if cancel_check and cancel_check():
            raise RuntimeError("cancelled")
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, params.detect_stride) != 0:
            frame_idx += 1
            continue

        t = frame_idx / fps
        preds = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False, device=device)
        p = preds[0]
        dets = []
        for xyxy, conf, cls in zip(p.boxes.xyxy.cpu().numpy(), p.boxes.conf.cpu().numpy(), p.boxes.cls.cpu().numpy()):
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = _clip(xyxy, w, h)
            dets.append(([x1, y1, x2 - x1, y2 - y1], float(conf), "person"))
        tracks = tracker.update_tracks(dets, frame=frame)

        best = None
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            box = _clip(tr.to_ltrb(), w, h)
            cscore = _color_score(frame, box, jersey_hsv, params.color_tolerance)
            motion = _iou(box, last_box) if last_box else 0.0
            ocr_match = 0.0
            ocr_txt = None
            ocr_conf = 0.0
            if ocr is not None and (frame_idx % 4 == 0):
                x1, y1, x2, y2 = box
                crop = frame[y1 + int((y2-y1)*0.2):y1 + int((y2-y1)*0.75), x1:x2]
                try:
                    rr = ocr.readtext(crop, detail=1, paragraph=False)
                    for _, txt, cf in rr:
                        d = _parse_digits(str(txt))
                        if d:
                            ocr_txt, ocr_conf = d, float(cf)
                            if player_num and d == player_num and ocr_conf >= params.ocr_min_conf:
                                ocr_match = 1.0
                                break
                except Exception as exc:
                    if ocr_gpu and device.startswith("cuda") and _is_cuda_fork_error(exc):
                        log.warning("Falling back to CPU OCR due to CUDA fork constraint")
                        ocr = easyocr.Reader(["en"], gpu=False)
                        ocr_gpu = False
                    else:
                        pass
            identity_bonus = params.identity_weight if (locked_track is not None and tr.track_id == locked_track) else 0.0
            score = params.color_weight * cscore + params.motion_weight * motion + params.ocr_weight * ocr_match + identity_bonus
            if best is None or score > best["score"]:
                best = {"track_id": tr.track_id, "box": box, "score": score, "color": cscore, "ocr_txt": ocr_txt, "ocr_conf": ocr_conf, "ocr_match": ocr_match}

        prev_state = state
        locked = False
        reason = None
        if best:
            last_box = best["box"]
            last_seen = t
            locked_track = best["track_id"] if locked_track is None else locked_track
            ocr_window.append(1 if best["ocr_match"] > 0 else 0)
            if state == "SEARCHING" and sum(ocr_window) >= params.ocr_confirm_m:
                state = "CONFIRMED"
                timeline.append({"t": t, "event": "state", "from": "SEARCHING", "to": "CONFIRMED"})
            if state in {"CONFIRMED", "LOCKED"}:
                state = "LOCKED"
                locked = True
                locked_until = max(locked_until, t + params.lock_seconds_after_confirm)
            if best["color"] < 0.05:
                low_color_frames += 1
            else:
                low_color_frames = 0
            lost_since = None
        else:
            if state == "LOCKED" and t <= locked_until:
                locked = True
            else:
                if lost_since is None:
                    lost_since = t
                if lost_since is not None and (t - lost_since) > params.lost_timeout:
                    reason = "lost_timeout"
                    state = "SEARCHING"
                    ocr_window.clear()
                    locked_track = None
            if low_color_frames > 8:
                reason = "low_color"
                state = "SEARCHING"
                locked = False

        if state == "LOCKED" and locked:
            if seg_start is None:
                seg_start = t
            if shift_state == "OFF_ICE":
                shift_state = "ON_ICE"
                timeline.append({"t": t, "event": "shift_state", "to": shift_state})
        else:
            if seg_start is not None:
                segments.append((seg_start, t, reason or "unlock"))
                seg_start = None
            if shift_state == "ON_ICE":
                bench = False
                if last_box is not None:
                    _, _, _, y2 = last_box
                    bench = y2 >= int(h * params.bench_zone_ratio)
                if bench and (t - last_seen) > params.lost_timeout:
                    shift_state = "EXITING"
                    timeline.append({"t": t, "event": "shift_state", "to": shift_state})
                    if shift_start is not None:
                        shifts.append((shift_start, t))
                    shift_state = "OFF_ICE"
                    shift_start = None

        if shift_state == "ON_ICE" and shift_start is None:
            shift_start = t

        if prev_state != state:
            timeline.append({"t": t, "event": "state", "from": prev_state, "to": state, "reason": reason})

        if writer is not None:
            draw = frame.copy()
            if best:
                x1, y1, x2, y2 = best["box"]
                cv2.rectangle(draw, (x1, y1), (x2, y2), (40, 220, 40), 2)
                cv2.putText(draw, f"track:{best['track_id']} color:{best['color']:.2f}", (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(draw, f"ocr:{best['ocr_txt']} conf:{best['ocr_conf']:.2f}", (x1, max(38, y1+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(draw, f"state:{state} lock_left:{max(0.0, locked_until-t):.1f}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
            cv2.putText(draw, f"mode:{params.tracking_mode} cam:{setup.get('camera_mode')}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
            writer.write(draw)

        if heartbeat and (time.time() - last_hb) >= HEARTBEAT_SECONDS:
            heartbeat(frame_idx, total, t)
            last_hb = time.time()

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if seg_start is not None:
        segments.append((seg_start, frame_idx / fps, "eof"))
    if shift_start is not None:
        shifts.append((shift_start, frame_idx / fps))

    merged = []
    for a, b, reason in segments:
        if (b - a) < params.min_track_seconds:
            continue
        if merged and a - merged[-1][1] <= params.gap_merge_seconds:
            prev_end = merged[-1][1]
            merged[-1] = (merged[-1][0], b, "merged")
            timeline.append({"t": a, "event": "merge", "gap": a - prev_end})
        else:
            merged.append((a, b, reason))

    return {
        "segments": merged,
        "shifts": shifts,
        "timeline": timeline,
        "debug_overlay_path": overlay_path,
        "fps": fps,
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
            cur.meta = {**(cur.meta or {}), "stage": stage_name, "progress": int(progress), "message": message}
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

        def hb(frame_idx, total, t):
            if cancel_check():
                raise RuntimeError("cancelled")
            if stage["stalled"]:
                raise RuntimeError(f"Stalled in {stage['name']}")
            pct = int(min(70, 10 + (frame_idx / max(1, total)) * 60))
            write_status("processing", "tracking", pct, f"Tracking in progress ({t:.1f}s)")

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

        if tracking_mode == "shift":
            if not shifts_json:
                raise RuntimeError("No shifts detected; see debug overlay/timeline")
        elif not clips and not combined_path:
            raise RuntimeError("No clips created; see debug overlay/timeline (checked clips and combined outputs)")

        if stage["stalled"]:
            raise RuntimeError(f"Stalled in {stage['name']}")

        write_status("done", "done", 100, "Processing complete")
        meta.update({
            "clips": clips,
            "combined_path": combined_path,
            "combined_url": combined_url,
            "artifacts": artifacts,
            "shifts": shifts_json,
            "total_toi": total_toi,
            "shift_count": len(shifts_json),
            "status": "done",
            "stage": "done",
            "progress": 100,
            "message": "Processing complete",
            "updated_at": time.time(),
        })
        meta_path.write_text(json.dumps(meta, indent=2))
        (job_dir / "results.json").write_text(json.dumps(meta, indent=2))
        return meta
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
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
        meta.update({"status": "failed", "stage": "failed", "progress": 100, "message": err, "error": traceback.format_exc(), "updated_at": time.time()})
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
