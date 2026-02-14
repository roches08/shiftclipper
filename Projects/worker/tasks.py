import os
import json
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from deep_sort_realtime.deepsort_tracker import DeepSort

"""
Worker: YOLO + DeepSORT tracker (accurate identity persistence)

Key improvements:
- Start processing near the first click (avoids wrong lock in early intro shots)
- Lock target track using click point (not just seed-center)
- ROI fallback: if ROI detects 0 boxes, fall back to full-frame YOLO
- Better defaults so short-but-real presences still produce clips
"""

BASE_DIR = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
JOBS_DIR = BASE_DIR / "data" / "jobs"

_YOLO_MODEL: Optional[YOLO] = None


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _set_status(job_id: str, **fields: Any) -> None:
    mp = _job_dir(job_id) / "meta.json"
    meta = _read_json(mp, {})
    meta.update(fields)
    meta["job_id"] = job_id
    meta["updated_at"] = time.time()
    _write_json(mp, meta)


def _ffmpeg_cut(in_path: str, out_path: str, start: float, end: float) -> None:
    dur = max(0.05, end - start)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", in_path,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=False)


def _ffmpeg_concat(file_list_path: str, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", file_list_path,
        "-c", "copy",
        out_path,
    ]
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0 or (not os.path.exists(out_path)) or os.path.getsize(out_path) < 1024:
        cmd2 = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", file_list_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            out_path,
        ]
        subprocess.run(cmd2, check=False)


def _load_model(yolo_weights: str = "yolov8s.pt") -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(yolo_weights)
    return _YOLO_MODEL


def _yolo_person_boxes(
    frame_bgr: np.ndarray,
    conf: float = 0.30,
    yolo_weights: str = "yolov8s.pt"
) -> List[Tuple[int, int, int, int, float]]:
    model = _load_model(yolo_weights=yolo_weights)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = model.predict(rgb, imgsz=640, conf=conf, verbose=False)[0]

    out: List[Tuple[int, int, int, int, float]] = []
    if res.boxes is None:
        return out

    # class 0 = person
    for b in res.boxes:
        cls = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])
        if cls != 0:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        c = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
        out.append((int(x1), int(y1), int(x2), int(y2), c))
    return out


def _pick_nearest_box_to_click(
    boxes: List[Tuple[int, int, int, int, float]],
    click_xy_norm: Tuple[float, float],
    frame_w: int,
    frame_h: int
) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None

    cx = float(click_xy_norm[0]) * frame_w
    cy = float(click_xy_norm[1]) * frame_h

    best = None
    best_d = 1e18

    for (x1, y1, x2, y2, _conf) in boxes:
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        area_frac = (bw * bh) / float(frame_w * frame_h + 1e-6)
        if area_frac < 0.003 or area_frac > 0.25:
            continue

        ar = bh / float(bw + 1e-6)
        if ar < 1.2 or ar > 6.0:
            continue

        mx = (x1 + x2) * 0.5
        my = (y1 + y2) * 0.5
        d = (mx - cx) ** 2 + (my - cy) ** 2

        inside = (cx >= x1 and cx <= x2 and cy >= y1 and cy <= y2)
        if inside:
            d *= 0.25

        if d < best_d:
            best_d = d
            best = (x1, y1, x2, y2)

    return best


def _seed_from_clicks(video_path: str, clicks: List[Dict[str, Any]], conf: float = 0.30, yolo_weights: str = "yolov8s.pt") -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": "Could not open video"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    picked: List[Tuple[int, int, int, int]] = []
    seed_frames_debug: List[Dict[str, Any]] = []

    for c in clicks:
        t = float(c.get("t", 0.0))
        x = float(c.get("x", 0.5))
        y = float(c.get("y", 0.5))

        frame_idx = max(0, int(round(t * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        boxes = _yolo_person_boxes(frame, conf=conf, yolo_weights=yolo_weights)
        bb = _pick_nearest_box_to_click(boxes, (x, y), W, H)
        seed_frames_debug.append({
            "t": t,
            "frame": frame_idx,
            "dets": len(boxes),
            "picked": list(bb) if bb else None
        })
        if bb:
            picked.append(bb)

    cap.release()

    if len(picked) < 1:
        return {
            "ok": False,
            "error": "No seed detections found from clicks",
            "seed_frames": seed_frames_debug,
            "fps": fps, "W": W, "H": H
        }

    # Median bbox of picked
    xs1 = int(np.median([b[0] for b in picked]))
    ys1 = int(np.median([b[1] for b in picked]))
    xs2 = int(np.median([b[2] for b in picked]))
    ys2 = int(np.median([b[3] for b in picked]))

    # Clamp overly huge seed
    bw = xs2 - xs1
    bh = ys2 - ys1
    max_w = int(0.50 * W)
    max_h = int(0.70 * H)

    if bw > max_w:
        cx = (xs1 + xs2) // 2
        xs1 = max(0, cx - max_w // 2)
        xs2 = min(W - 1, cx + max_w // 2)

    if bh > max_h:
        cy = (ys1 + ys2) // 2
        ys1 = max(0, cy - max_h // 2)
        ys2 = min(H - 1, cy + max_h // 2)

    return {
        "ok": True,
        "seed_bbox": (xs1, ys1, xs2, ys2),
        "seed_frames": seed_frames_debug,
        "seed_count": len(picked),
        "fps": fps, "W": W, "H": H
    }


def _build_presence_spans(
    times_present: List[float],
    gap_merge: float,
    pre_roll: float,
    post_roll: float,
    min_len: float,
    max_clip_seconds: float = 18.0
) -> List[Tuple[float, float]]:
    if not times_present:
        return []

    times_present = sorted(times_present)

    spans: List[Tuple[float, float]] = []
    s = times_present[0]
    e = times_present[0]
    for t in times_present[1:]:
        if (t - e) <= gap_merge:
            e = t
        else:
            spans.append((max(0.0, s - pre_roll), e + post_roll))
            s = t
            e = t
    spans.append((max(0.0, s - pre_roll), e + post_roll))

    final: List[Tuple[float, float]] = []
    for s, e in spans:
        length = e - s
        if length < min_len:
            continue
        if length > max_clip_seconds:
            cur = s
            while cur < e:
                nxt = min(cur + max_clip_seconds, e)
                if (nxt - cur) >= min_len:
                    final.append((cur, nxt))
                cur = nxt
        else:
            final.append((s, e))

    return final


def process_job(job_id: str) -> Dict[str, Any]:
    jd = _job_dir(job_id)
    setup = _read_json(jd / "setup.json", {})
    meta = _read_json(jd / "meta.json", {})

    video_path = meta.get("video_path")
    if not video_path or not os.path.exists(video_path):
        _set_status(job_id, status="error", stage="error", progress=0, error="Missing video. Upload first.")
        return {"status": "error", "error": "Missing video"}

    clicks = setup.get("clicks") or setup.get("seeds") or []
    if not clicks:
        _set_status(job_id, status="error", stage="error", progress=0, error="No clicks found. Click the player at least 3 times.")
        return {"status": "error", "error": "No clicks"}

    # Accuracy-first defaults
    yolo_weights = str(setup.get("yolo_weights", "yolov8s.pt"))
    conf = float(setup.get("yolo_conf", 0.30))
    detect_stride = int(setup.get("detect_stride", 2))

    # Clip shaping defaults (better for real hockey shifts)
    min_clip_len = float(setup.get("min_clip_len", 3.0))          # WAS 5.0 (too strict)
    gap_merge = float(setup.get("gap_merge", 1.25))
    pre_roll = float(setup.get("pre_roll", 1.25))
    post_roll = float(setup.get("post_roll", 0.90))
    sticky_seconds = float(setup.get("sticky_seconds", 1.20))     # WAS 0.55 (too short for occlusion)
    max_clip_seconds = float(setup.get("max_clip_seconds", 18.0))

    # DeepSORT tuning
    ds_max_age = int(setup.get("ds_max_age", 90))                 # WAS 45
    ds_n_init = int(setup.get("ds_n_init", 2))
    ds_max_iou = float(setup.get("ds_max_iou", 0.8))
    ds_max_cos = float(setup.get("ds_max_cos", 0.15))
    ds_nn_budget = int(setup.get("ds_nn_budget", 100))

    tracker = DeepSort(
        max_age=ds_max_age,
        n_init=ds_n_init,
        max_iou_distance=ds_max_iou,
        max_cosine_distance=ds_max_cos,
        nn_budget=ds_nn_budget,
    )

    # ROI detection after lock
    roi_enable = bool(setup.get("roi_enable", True))
    roi_pad_frac = float(setup.get("roi_pad_frac", 0.55))

    # NEW: start near first click so we don't lock onto an intro shot player
    first_click_t = min(float(c.get("t", 0.0)) for c in clicks)
    start_t = max(0.0, first_click_t - float(setup.get("start_before_click_sec", 2.0)))

    _set_status(job_id, status="processing", stage="seed", progress=35, message="Seeding player from clicks…")

    seed_info = _seed_from_clicks(video_path, clicks, conf=conf, yolo_weights=yolo_weights)
    if not seed_info.get("ok"):
        _set_status(job_id, status="error", stage="error", progress=0, error=seed_info.get("error", "Seed failed"))
        results = {"status": "error", "job_id": job_id, "error": seed_info.get("error", "Seed failed"), "debug": seed_info}
        _write_json(jd / "results.json", results)
        return results

    seed_bbox = seed_info["seed_bbox"]
    fps = float(seed_info["fps"] or 30.0)
    W = int(seed_info["W"] or 0)
    H = int(seed_info["H"] or 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _set_status(job_id, status="error", stage="error", progress=0, error="Could not open video")
        return {"status": "error", "error": "Could not open video"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (total_frames / fps) if (total_frames > 0 and fps > 0) else None

    # Jump to start frame
    start_frame = int(start_t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    times_present: List[float] = []
    last_center = ((seed_bbox[0] + seed_bbox[2]) * 0.5, (seed_bbox[1] + seed_bbox[3]) * 0.5)
    last_present_t: Optional[float] = None
    lost_since_t: Optional[float] = None

    target_track_id: Optional[int] = None

    debug_samples: List[Dict[str, Any]] = []
    max_debug = 160

    _set_status(job_id, status="processing", stage="tracking", progress=45, message="Tracking (YOLO + DeepSORT)…")

    # Click lock window (seconds around any click where we force-lock by click point)
    click_lock_window = float(setup.get("click_lock_window_sec", 0.30))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps if fps > 0 else 0.0
        do_detect = (frame_idx % max(1, detect_stride) == 0)

        present_now = False
        chosen_bb = None
        boxes_count = None
        used_roi = False

        if do_detect:
            # ROI crop around last_center once we have a target and are not heavily lost
            frame_for_det = frame
            x_off, y_off = 0, 0

            if roi_enable and target_track_id is not None and (lost_since_t is None or (t - lost_since_t) < 1.0):
                pad_w = int(roi_pad_frac * W)
                pad_h = int(roi_pad_frac * H)
                cx, cy = int(last_center[0]), int(last_center[1])

                x1r = max(0, cx - pad_w // 2)
                x2r = min(W, cx + pad_w // 2)
                y1r = max(0, cy - pad_h // 2)
                y2r = min(H, cy + pad_h // 2)

                if (x2r - x1r) > 320 and (y2r - y1r) > 180:
                    frame_for_det = frame[y1r:y2r, x1r:x2r]
                    x_off, y_off = x1r, y1r
                    used_roi = True

            boxes = _yolo_person_boxes(frame_for_det, conf=conf, yolo_weights=yolo_weights)
            boxes_count = len(boxes)

            # NEW: ROI fallback — if ROI produces 0 detections, immediately run full-frame
            if used_roi and boxes_count == 0:
                used_roi = False
                x_off, y_off = 0, 0
                boxes = _yolo_person_boxes(frame, conf=conf, yolo_weights=yolo_weights)
                boxes_count = len(boxes)

            detections = []
            for (x1, y1, x2, y2, c) in boxes:
                x1 += x_off
                x2 += x_off
                y1 += y_off
                y2 += y_off

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                area_frac = (bw * bh) / float(W * H + 1e-6)
                if area_frac < 0.003 or area_frac > 0.25:
                    continue

                ar = bh / float(bw + 1e-6)
                if ar < 1.2 or ar > 6.0:
                    continue

                detections.append(([x1, y1, bw, bh], float(c), "person"))

            tracks = tracker.update_tracks(detections, frame=frame)

            # ---------- NEW: LOCK USING CLICK POINT ----------
            if target_track_id is None:
                # If we're near any click time, lock to the track that contains the click (best),
                # otherwise nearest center to click.
                lock_click = None
                for ck in clicks:
                    if abs(t - float(ck.get("t", 0.0))) <= click_lock_window:
                        lock_click = ck
                        break

                if lock_click is not None:
                    px = float(lock_click.get("x", 0.5)) * W
                    py = float(lock_click.get("y", 0.5)) * H

                    best = None
                    best_d = 1e18
                    for tr in tracks:
                        if not tr.is_confirmed():
                            continue
                        x1, y1, x2, y2 = map(int, tr.to_ltrb())
                        inside = (px >= x1 and px <= x2 and py >= y1 and py <= y2)
                        if inside:
                            best = (tr.track_id, (x1, y1, x2, y2))
                            break

                        cx = (x1 + x2) * 0.5
                        cy = (y1 + y2) * 0.5
                        d = (cx - px) ** 2 + (cy - py) ** 2
                        if d < best_d:
                            best_d = d
                            best = (tr.track_id, (x1, y1, x2, y2))

                    if best is not None:
                        target_track_id, chosen_bb = best
                        present_now = True

            # If already locked, try to find that track id
            if target_track_id is not None and not present_now:
                for tr in tracks:
                    if not tr.is_confirmed():
                        continue
                    if tr.track_id == target_track_id:
                        x1, y1, x2, y2 = map(int, tr.to_ltrb())
                        chosen_bb = (x1, y1, x2, y2)
                        present_now = True
                        break

            # Fallback: if still not locked (no click window hit yet), use seed center
            if target_track_id is None and not present_now:
                seed_cx = (seed_bbox[0] + seed_bbox[2]) * 0.5
                seed_cy = (seed_bbox[1] + seed_bbox[3]) * 0.5
                best_lock = None
                best_d = 1e18
                for tr in tracks:
                    if not tr.is_confirmed():
                        continue
                    x1, y1, x2, y2 = map(int, tr.to_ltrb())
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    d = (cx - seed_cx) ** 2 + (cy - seed_cy) ** 2
                    if d < best_d:
                        best_d = d
                        best_lock = (tr.track_id, (x1, y1, x2, y2))
                if best_lock is not None:
                    target_track_id, chosen_bb = best_lock
                    present_now = True
            # ---------- END LOCK ----------

        # sticky presence accounting
        if present_now and chosen_bb is not None:
            last_center = ((chosen_bb[0] + chosen_bb[2]) * 0.5, (chosen_bb[1] + chosen_bb[3]) * 0.5)
            last_present_t = t
            lost_since_t = None
            times_present.append(t)
        else:
            if lost_since_t is None:
                lost_since_t = t
            if last_present_t is not None and (t - last_present_t) <= sticky_seconds:
                times_present.append(t)

        if do_detect and len(debug_samples) < max_debug:
            debug_samples.append({
                "t": t,
                "present": present_now,
                "chosen": list(chosen_bb) if chosen_bb else None,
                "boxes": boxes_count,
                "roi": used_roi,
                "track_id": int(target_track_id) if target_track_id is not None else None,
            })

        if do_detect and (frame_idx - start_frame) % int(max(1, detect_stride) * 200) == 0 and total_frames > 0:
            prog = 45 + int(45.0 * (frame_idx / float(total_frames)))
            _set_status(job_id, progress=min(90, prog), message=f"Tracking… {t:.1f}s")

        frame_idx += 1

    cap.release()

    if not times_present:
        _set_status(
            job_id,
            status="error",
            stage="error",
            progress=100,
            error="No track matched. Try clicks with the player clearly visible."
        )
        results = {
            "status": "error",
            "job_id": job_id,
            "error": "No matches",
            "debug": {"seed": seed_info, "debug_samples": debug_samples}
        }
        _write_json(jd / "results.json", results)
        return results

    spans = _build_presence_spans(
        times_present,
        gap_merge=gap_merge,
        pre_roll=pre_roll,
        post_roll=post_roll,
        min_len=min_clip_len,
        max_clip_seconds=max_clip_seconds,
    )

    # NEW: if spans empty, report error (don’t pretend “done”)
    if not spans:
        _set_status(
            job_id,
            status="error",
            stage="done",
            progress=100,
            error="No spans >= min_clip_len. Lower min_clip_len or improve tracking lock."
        )
        results = {
            "status": "error",
            "job_id": job_id,
            "error": "No spans",
            "clips": [],
            "combined_path": str(jd / "combined.mp4"),
            "combined_url": f"/data/jobs/{job_id}/combined.mp4",
            "debug": {
                "seed": {
                    "seed_bbox": list(seed_bbox),
                    "seed_frames": seed_info.get("seed_frames", []),
                    "seed_count": seed_info.get("seed_count", 0),
                },
                "fps": fps, "W": W, "H": H,
                "total_frames": total_frames,
                "duration_s": duration_s,
                "start_t": start_t,
                "min_clip_len": min_clip_len,
                "gap_merge": gap_merge,
                "sticky_seconds": sticky_seconds,
                "pre_roll": pre_roll,
                "post_roll": post_roll,
                "detect_stride": detect_stride,
                "yolo_conf": conf,
                "yolo_weights": yolo_weights,
                "ds": {
                    "max_age": ds_max_age,
                    "n_init": ds_n_init,
                    "max_iou": ds_max_iou,
                    "max_cos": ds_max_cos,
                    "nn_budget": ds_nn_budget,
                },
                "roi_enable": roi_enable,
                "roi_pad_frac": roi_pad_frac,
                "debug_samples": debug_samples,
                "spans": spans,
                "target_track_id": int(target_track_id) if target_track_id is not None else None,
            }
        }
        _write_json(jd / "results.json", results)
        return results

    _set_status(job_id, status="processing", stage="cutting", progress=92, message=f"Cutting {len(spans)} clip(s)…")

    clips_dir = jd / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    for i, (s, e) in enumerate(spans, start=1):
        out_path = str(clips_dir / f"clip_{i:03d}.mp4")
        _ffmpeg_cut(video_path, out_path, s, e)
        clips.append({
            "start": float(s),
            "end": float(e),
            "path": out_path,
            "url": f"/data/jobs/{job_id}/clips/clip_{i:03d}.mp4",
        })

    list_path = str(jd / "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{c['path']}'\n")

    combined_path = str(jd / "combined.mp4")
    _ffmpeg_concat(list_path, combined_path)

    results = {
        "status": "done",
        "job_id": job_id,
        "camera_mode": setup.get("camera_mode", "broadcast"),
        "player_number": setup.get("player_number", ""),
        "jersey_color": setup.get("jersey_color", ""),
        "clicks_count": len(clicks),
        "clicks": clicks,
        "clips": clips,
        "combined_path": combined_path,
        "combined_url": f"/data/jobs/{job_id}/combined.mp4",
        "debug": {
            "seed": {
                "seed_bbox": list(seed_bbox),
                "seed_frames": seed_info.get("seed_frames", []),
                "seed_count": seed_info.get("seed_count", 0),
            },
            "fps": fps, "W": W, "H": H,
            "total_frames": total_frames,
            "duration_s": duration_s,
            "start_t": start_t,
            "min_clip_len": min_clip_len,
            "gap_merge": gap_merge,
            "sticky_seconds": sticky_seconds,
            "pre_roll": pre_roll,
            "post_roll": post_roll,
            "detect_stride": detect_stride,
            "yolo_conf": conf,
            "yolo_weights": yolo_weights,
            "ds": {
                "max_age": ds_max_age,
                "n_init": ds_n_init,
                "max_iou": ds_max_iou,
                "max_cos": ds_max_cos,
                "nn_budget": ds_nn_budget,
            },
            "roi_enable": roi_enable,
            "roi_pad_frac": roi_pad_frac,
            "debug_samples": debug_samples,
            "spans": spans,
            "target_track_id": int(target_track_id) if target_track_id is not None else None,
        }
    }

    _write_json(jd / "results.json", results)
    _set_status(
        job_id,
        status="done",
        stage="done",
        progress=100,
        message="Done.",
        clips=clips,
        combined_path=combined_path,
        combined_url=results["combined_url"],
    )
    return results

