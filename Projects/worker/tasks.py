import os
import json
import time
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# NEW: DeepSORT tracker
from deep_sort_realtime.deepsort_tracker import DeepSort

"""
Worker: YOLO + DeepSORT tracker (accurate identity persistence)

Pipeline:
1) Use clicks to seed who the player is (seed bbox).
2) Run YOLO "person" detection at a stride.
3) Feed detections to DeepSORT to get stable track IDs.
4) Lock onto a target track ID based on click/seed proximity.
5) Build "present spans" (sticky + merge gaps + pre/post roll + min clip)
6) Cut clips and concat into combined.mp4
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


def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def _load_model(yolo_weights: str = "yolov8s.pt") -> YOLO:
    """
    Accuracy-first default: yolov8s.pt (better than v8n on hockey broadcast).
    You can override via setup.json: "yolo_weights": "yolov8n.pt" or "yolov8m.pt"
    """
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(yolo_weights)
    return _YOLO_MODEL


def _yolo_person_boxes(frame_bgr: np.ndarray, conf: float = 0.30, yolo_weights: str = "yolov8s.pt") -> List[Tuple[int, int, int, int, float]]:
    """Run YOLO person detection on a single frame (BGR ndarray).

    Ultralytics input handling has changed across versions; to avoid "Unsupported image type"
    errors we:
      - convert to RGB uint8 contiguous
      - call predict with `source=...` and fall back to `[source]` if needed
    """
    model = _load_model(yolo_weights=yolo_weights)

    if frame_bgr is None:
        return []

    # BGR -> RGB, make sure it's uint8 contiguous (Ultralytics can be picky)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)

    try:
        res = model.predict(source=rgb, conf=conf, classes=[0], verbose=False)
    except TypeError:
        # Some versions only accept list/iterables of images
        res = model.predict(source=[rgb], conf=conf, classes=[0], verbose=False)
    except Exception:
        # Last resort: also try list form
        res = model.predict(source=[rgb], conf=conf, classes=[0], verbose=False)

    if not res:
        return []
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    out: List[Tuple[int, int, int, int, float]] = []
    xyxy = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        out.append((int(x1), int(y1), int(x2), int(y2), float(c)))
    return out


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

    # Filter outliers by area
    areas = np.array([(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in picked], dtype=np.float32)
    med_area = float(np.median(areas))
    keep: List[Tuple[int, int, int, int]] = []
    for bb, a in zip(picked, areas):
        if a < med_area * 0.35 or a > med_area * 2.8:
            continue
        keep.append(bb)
    if keep:
        picked = keep

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

    merged: List[List[float]] = []
    for (s, e) in spans:
        if not merged:
            merged.append([s, e])
            continue
        if (e - s) >= min_len:
            merged.append([s, e])
            continue
        ps, pe = merged[-1]
        if s - pe <= gap_merge:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])

    final: List[Tuple[float, float]] = []
    for s, e in merged:
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
    try:
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
        detect_stride = int(setup.get("detect_stride", 2))  # YOLO every 2 frames is accurate and manageable on L4

        # Pro hockey clip shaping defaults
        min_clip_len = float(setup.get("min_clip_len", 5.0))
        gap_merge = float(setup.get("gap_merge", 1.25))
        pre_roll = float(setup.get("pre_roll", 1.25))
        post_roll = float(setup.get("post_roll", 0.90))
        sticky_seconds = float(setup.get("sticky_seconds", 0.55))
        max_clip_seconds = float(setup.get("max_clip_seconds", 18.0))

        # DeepSORT tuning (accuracy-focused)
        tracker = DeepSort(
            max_age=int(setup.get("ds_max_age", 45)),
            n_init=int(setup.get("ds_n_init", 2)),
            max_iou_distance=float(setup.get("ds_max_iou", 0.8)),
            max_cosine_distance=float(setup.get("ds_max_cos", 0.15)),
            nn_budget=int(setup.get("ds_nn_budget", 100)),
        )

        # ROI detection after lock (reduces false candidates + speeds up)
        roi_enable = bool(setup.get("roi_enable", True))
        roi_pad_frac = float(setup.get("roi_pad_frac", 0.55))  # 0.4–0.7 reasonable

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

        # tracking loop
        times_present: List[float] = []
        last_center = ((seed_bbox[0] + seed_bbox[2]) * 0.5, (seed_bbox[1] + seed_bbox[3]) * 0.5)
        last_present_t: Optional[float] = None
        lost_since_t: Optional[float] = None

        target_track_id: Optional[int] = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        sampled = 0

        debug_samples: List[Dict[str, Any]] = []
        max_debug = 120

        _set_status(job_id, status="processing", stage="tracking", progress=45, message="Tracking (YOLO + DeepSORT)…")

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
                sampled += 1

                # ROI crop around last_center once we have a target and are not heavily lost
                frame_for_det = frame
                x_off, y_off = 0, 0

                if roi_enable and target_track_id is not None and (lost_since_t is None or (t - lost_since_t) < 1.0):
                    pad_w = int(roi_pad_frac * W)
                    pad_h = int(roi_pad_frac * H)
                    cx, cy = int(last_center[0]), int(last_center[1])

                    x1 = max(0, cx - pad_w // 2)
                    x2 = min(W, cx + pad_w // 2)
                    y1 = max(0, cy - pad_h // 2)
                    y2 = min(H, cy + pad_h // 2)

                    # avoid tiny ROI
                    if (x2 - x1) > 320 and (y2 - y1) > 180:
                        frame_for_det = frame[y1:y2, x1:x2]
                        x_off, y_off = x1, y1
                        used_roi = True

                boxes = _yolo_person_boxes(frame_for_det, conf=conf, yolo_weights=yolo_weights)
                boxes_count = len(boxes)

                detections = []
                for (x1, y1, x2, y2, c) in boxes:
                    # map back to full frame coords if ROI
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

                seed_cx = (seed_bbox[0] + seed_bbox[2]) * 0.5
                seed_cy = (seed_bbox[1] + seed_bbox[3]) * 0.5

                best_lock = None
                best_d = 1e18

                for tr in tracks:
                    if not tr.is_confirmed():
                        continue

                    ltrb = tr.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5

                    if target_track_id is not None:
                        if tr.track_id == target_track_id:
                            chosen_bb = (x1, y1, x2, y2)
                            present_now = True
                            break
                        continue

                    # Before locking: pick closest to seed center (clicks drove seed bbox)
                    d = (cx - seed_cx) ** 2 + (cy - seed_cy) ** 2
                    if d < best_d:
                        best_d = d
                        best_lock = (tr.track_id, (x1, y1, x2, y2))

                if target_track_id is None and best_lock is not None:
                    target_track_id, chosen_bb = best_lock
                    present_now = True

            # update sticky / lost tracking
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

            # debug sample
            if do_detect and len(debug_samples) < max_debug:
                debug_samples.append({
                    "t": t,
                    "present": present_now,
                    "chosen": list(chosen_bb) if chosen_bb else None,
                    "boxes": boxes_count,
                    "roi": used_roi,
                    "track_id": target_track_id,
                })

            # progress updates
            if do_detect and sampled % 200 == 0 and total_frames > 0:
                prog = 45 + int(45.0 * (frame_idx / float(total_frames)))
                _set_status(job_id, progress=min(90, prog), message=f"Tracking… {t:.1f}s")

            frame_idx += 1

        cap.release()

        if not times_present:
            _set_status(
                job_id,
                status="error",
                stage="error",
                progress=0,
                error="No track matched. Try clicks with the player clearly visible (not on bench / not occluded)."
            )
            results = {
                "status": "error",
                "job_id": job_id,
                "error": "No matches",
                "debug": {
                    "seed": seed_info,
                    "sampled": sampled,
                    "debug_samples": debug_samples,
                }
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
                "fps": fps,
                "W": W, "H": H,
                "total_frames": total_frames,
                "duration_s": duration_s,
                "min_clip_len": min_clip_len,
                "gap_merge": gap_merge,
                "sticky_seconds": sticky_seconds,
                "pre_roll": pre_roll,
                "post_roll": post_roll,
                "detect_stride": detect_stride,
                "yolo_conf": conf,
                "yolo_weights": yolo_weights,
                "ds": {
                    "max_age": int(setup.get("ds_max_age", 45)),
                    "n_init": int(setup.get("ds_n_init", 2)),
                    "max_iou": float(setup.get("ds_max_iou", 0.8)),
                    "max_cos": float(setup.get("ds_max_cos", 0.20)),
                    "nn_budget": int(setup.get("ds_nn_budget", 100)),
                },
                "roi_enable": roi_enable,
                "roi_pad_frac": roi_pad_frac,
                "sampled_detect_frames": sampled,
                "debug_samples": debug_samples,
                "spans": spans,
                "target_track_id": target_track_id,
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

    except Exception as e:
        # Make sure the UI does not sit on 'queued' forever if something blows up.
        try:
            _set_status(job_id, status='error', stage='error', progress=100, message=f'Error: {e}')
        except Exception:
            pass
        # Do not re-raise; keep worker alive and let UI show the error.
        return None