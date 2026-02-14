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

"""
Worker: detection-based tracker (no OpenCV contrib trackers)

Pipeline:
1) Use clicks to seed who the player is (seed bbox + seed color).
2) Run YOLO "person" detection at a stride across the video.
3) For each sampled frame pick the best candidate using:
   - jersey color match (dominant)
   - proximity to last known position (identity gating)
4) Build "present spans" (sticky + merge gaps + pre/post roll + min clip)
5) Cut clips and concat into combined.mp4
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


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    s = (hex_color or "").strip().lstrip("#")
    if len(s) != 6:
        return (0, 0, 0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def _bgr_to_hsv_color(bgr: Tuple[int, int, int]) -> np.ndarray:
    px = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)
    return hsv


def _hsv_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Hue wrap-around
    dh_raw = abs(float(a[0]) - float(b[0]))
    dh = min(dh_raw, 180.0 - dh_raw) / 90.0
    ds = abs(float(a[1]) - float(b[1])) / 255.0
    dv = abs(float(a[2]) - float(b[2])) / 255.0
    return float(2.0 * dh + 1.2 * ds + 0.6 * dv)


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


def _torso_crop(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    # torso-ish region
    cx1 = x1 + int(0.25 * bw)
    cx2 = x1 + int(0.75 * bw)
    cy1 = y1 + int(0.20 * bh)
    cy2 = y1 + int(0.65 * bh)
    H, W = frame.shape[:2]
    cx1, cy1, cx2, cy2 = _clip_box(cx1, cy1, cx2, cy2, W, H)
    return frame[cy1:cy2, cx1:cx2]


def _mean_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)


def _load_model() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO("yolov8n.pt")
    return _YOLO_MODEL


def _yolo_person_boxes(frame_bgr: np.ndarray, conf: float = 0.25) -> List[Tuple[int, int, int, int, float]]:
    model = _load_model()
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


def _seed_from_clicks(video_path: str, clicks: List[Dict[str, Any]], conf: float = 0.25) -> Dict[str, Any]:
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

        boxes = _yolo_person_boxes(frame, conf=conf)
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
    max_clip_seconds: float = 20.0
) -> List[Tuple[float, float]]:
    if not times_present:
        return []

    times_present = sorted(times_present)

    # raw spans from timestamps
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

    # merge / enforce minimum
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

    # drop tiny + split huge
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

    jersey_hex = setup.get("jersey_color", "") or ""
    jersey_hsv = _bgr_to_hsv_color(_hex_to_bgr(jersey_hex))

    # knobs
    min_clip_len = float(setup.get("min_clip_len", 5.0))
    gap_merge = float(setup.get("extend_sec", 1.25))      # merge window (auto extend)
    pre_roll = float(setup.get("pre_roll", 1.25))          # 3-5s requested; default 4
    post_roll = float(setup.get("post_roll", 1.0))
    sticky_seconds = float(setup.get("sticky_seconds", 0.55))

    detect_stride = int(setup.get("detect_stride", 2))
    conf = float(setup.get("yolo_conf", 0.30))

    color_threshold = float(setup.get("color_threshold", 0.95))  # lower=stricter
    dist_gate = float(setup.get("dist_gate_norm", 0.14))
    dist_gate2 = float(setup.get("dist_gate2_norm", 0.28))

    _set_status(job_id, status="processing", stage="seed", progress=35, message="Seeding player from clicks…")

    seed_info = _seed_from_clicks(video_path, clicks, conf=conf)
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

    # Build target color from seed bbox at earliest click frame (fallback to jersey picker)
    target_hsv = jersey_hsv.copy()
    first_click_t = float(min([float(c.get("t", 0.0)) for c in clicks]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(first_click_t * fps)))
    ok0, frame0 = cap.read()
    if ok0:
        torso0 = _torso_crop(frame0, seed_bbox)
        mh = _mean_hsv(torso0)
        if mh[1] > 25:  # saturation check
            target_hsv = mh

    # tracking loop
    times_present: List[float] = []
    last_center = ((seed_bbox[0] + seed_bbox[2]) * 0.5, (seed_bbox[1] + seed_bbox[3]) * 0.5)
    last_present_t: Optional[float] = None
    lost_since_t: Optional[float] = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    sampled = 0

    debug_samples: List[Dict[str, Any]] = []
    max_debug = 80

    _set_status(job_id, status="processing", stage="tracking", progress=45, message="Tracking (YOLO + jersey color)…")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps if fps > 0 else 0.0
        do_detect = (frame_idx % max(1, detect_stride) == 0)

        present_now = False
        chosen_bb = None
        chosen_score = None
        chosen_color = None
        chosen_dist = None
        boxes_count = None

        if do_detect:
            sampled += 1
            boxes = _yolo_person_boxes(frame, conf=conf)
            boxes_count = len(boxes)

            best_score = 1e18
            best_bb = None
            best_color = None
            best_dist = None

            for (x1, y1, x2, y2, _c) in boxes:
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

                torso = _torso_crop(frame, (x1, y1, x2, y2))
                mh = _mean_hsv(torso)
                sat_penalty = 0.25 if mh[1] < 20 else 0.0
                cd = _hsv_distance(mh, target_hsv) + sat_penalty

                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                dx = (cx - last_center[0]) / float(W + 1e-6)
                dy = (cy - last_center[1]) / float(H + 1e-6)
                dist = math.sqrt(dx * dx + dy * dy)

                # gating
                if cd > color_threshold:
                    continue
                if dist > dist_gate and cd > (color_threshold * 0.75):
                    continue
                if dist > dist_gate2:
                    continue

                score = (2.2 * cd) + (0.9 * dist)
                if score < best_score:
                    best_score = score
                    best_bb = (x1, y1, x2, y2)
                    best_color = cd
                    best_dist = dist

            if best_bb is not None:
                present_now = True
                chosen_bb = best_bb
                chosen_score = float(best_score)
                chosen_color = float(best_color) if best_color is not None else None
                chosen_dist = float(best_dist) if best_dist is not None else None

            # reacquire guard if we’ve been lost awhile
            if present_now and lost_since_t is not None:
                lost_dur = t - lost_since_t
                if lost_dur > 2.0:
                    # require stronger color to reacquire after a long loss
                    if chosen_color is None or chosen_color > (color_threshold * 0.6):
                        present_now = False
                        chosen_bb = None

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
                # sticky presence to avoid fragmenting
                times_present.append(t)

        # debug sample
        if do_detect and len(debug_samples) < max_debug:
            debug_samples.append({
                "t": t,
                "present": present_now,
                "chosen": list(chosen_bb) if chosen_bb else None,
                "score": chosen_score,
                "color": chosen_color,
                "dist": chosen_dist,
                "boxes": boxes_count,
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
            error="No detections matched jersey color/gating. Try clicks where the jersey is clearly visible, or loosen color_threshold slightly."
        )
        results = {
            "status": "error",
            "job_id": job_id,
            "error": "No matches",
            "debug": {
                "seed": seed_info,
                "sampled": sampled,
                "color_threshold": color_threshold,
                "dist_gate_norm": dist_gate,
                "dist_gate2_norm": dist_gate2,
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
        max_clip_seconds=float(setup.get("max_clip_seconds", 20.0)),
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
            "color_threshold": color_threshold,
            "dist_gate_norm": dist_gate,
            "dist_gate2_norm": dist_gate2,
            "sampled_detect_frames": sampled,
            "debug_samples": debug_samples,
            "spans": spans,
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

