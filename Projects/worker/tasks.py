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


def _set_status(job_id: str, **fields: Any) -> Dict[str, Any]:
    mp = _job_dir(job_id) / "meta.json"
    meta = _read_json(mp, {})
    meta.update(fields)
    meta["job_id"] = job_id
    meta["updated_at"] = time.time()
    _write_json(mp, meta)
    return meta


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    s = (hex_color or "").strip().lstrip("#")
    if len(s) != 6:
        return (0, 0, 0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def _bgr_to_hsv(bgr: Tuple[int, int, int]) -> np.ndarray:
    px = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)
    return hsv


def _hsv_dist(a: np.ndarray, b: np.ndarray) -> float:
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
    cx1 = x1 + int(0.25 * bw)
    cx2 = x1 + int(0.75 * bw)
    cy1 = y1 + int(0.20 * bh)
    cy2 = y1 + int(0.65 * bh)
    H, W = frame.shape[:2]
    cx1, cy1, cx2, cy2 = _clip_box(cx1, cy1, cx2, cy2, W, H)
    return frame[cy1:cy2, cx1:cx2]


def _mean_hsv(bgr_img: np.ndarray) -> np.ndarray:
    if bgr_img.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)


def _load_model() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        # use the small model you already downloaded (or ultralytics will auto-download)
        weights = "yolov8s.pt" if (BASE_DIR / "yolov8s.pt").exists() else "yolov8s.pt"
        _YOLO_MODEL = YOLO(weights)
    return _YOLO_MODEL


def _yolo_person_boxes(frame_bgr: np.ndarray, conf: float = 0.25) -> List[Tuple[int, int, int, int, float]]:
    """
    CRITICAL: pass a numpy image, NOT a VideoCapture, NOT an iterator, NOT a Path object.
    This avoids the 'Unsupported image type' crash.
    """
    model = _load_model()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = model.predict(rgb, imgsz=640, conf=conf, verbose=False)[0]

    out: List[Tuple[int, int, int, int, float]] = []
    if res.boxes is None:
        return out

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


def track_presence_spans(
    video_path: str,
    clicks: List[Dict[str, float]],
    jersey_hex: str,
    *,
    min_clip_len: float = 20.0,
    gap_merge: float = 2.0,
    sticky_seconds: float = 1.5,
    pre_roll: float = 4.0,
    post_roll: float = 1.5,
    detect_stride: int = 3,
    yolo_conf: float = 0.25,
    color_threshold: float = 1.05,
    dist_gate_norm: float = 0.18,
    dist_gate2_norm: float = 0.35,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if total_frames > 0 else 0.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    jersey_hsv = _bgr_to_hsv(_hex_to_bgr(jersey_hex))

    # --- build seed bbox + seed color from clicks ---
    seed_boxes: List[Tuple[int, int, int, int]] = []
    seed_samples: List[Dict[str, Any]] = []

    for c in (clicks or []):
        t = float(c.get("t", 0.0))
        x = float(c.get("x", 0.5))
        y = float(c.get("y", 0.5))
        frame_idx = int(round(t * fps))
        frame_idx = max(0, min(frame_idx, max(0, total_frames - 1)))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        dets = _yolo_person_boxes(frame, conf=yolo_conf)
        picked = _pick_nearest_box_to_click(dets, (x, y), W, H)
        seed_samples.append({"t": t, "frame": frame_idx, "dets": len(dets), "picked": picked})
        if picked is not None:
            seed_boxes.append(picked)

    if not seed_boxes:
        raise RuntimeError("No seed bbox found from clicks. Try clicking closer to the target player.")

    # seed color from median of torso HSV on seed boxes
    seed_hsvs = []
    for i, box in enumerate(seed_boxes):
        t = float(seed_samples[i].get("t", 0.0)) if i < len(seed_samples) else 0.0
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        torso = _torso_crop(frame, box)
        seed_hsvs.append(_mean_hsv(torso))

    if seed_hsvs:
        seed_hsv = np.median(np.stack(seed_hsvs, axis=0), axis=0).astype(np.float32)
    else:
        seed_hsv = jersey_hsv

    # --- detection loop ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    last_box: Optional[Tuple[int, int, int, int]] = None
    last_seen_t: float = -1e9
    present_times: List[float] = []

    debug_samples: List[Dict[str, Any]] = []

    sampled_frames = 0
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t = f / fps
        if f % max(1, detect_stride) != 0:
            f += 1
            continue

        sampled_frames += 1
        dets = _yolo_person_boxes(frame, conf=yolo_conf)

        best_score = -1e9
        best_box = None
        best_color = None
        best_dist = None

        for (x1, y1, x2, y2, conf) in dets:
            box = (x1, y1, x2, y2)
            torso = _torso_crop(frame, box)
            hsv = _mean_hsv(torso)

            # color score: closer to seed_hsv is better
            cdist = _hsv_dist(hsv, seed_hsv)
            color_score = max(0.0, 2.2 - cdist)  # ~[0..2.2]

            # distance gate: encourage continuity
            dist_norm = 0.0
            if last_box is not None:
                lx1, ly1, lx2, ly2 = last_box
                lcx = (lx1 + lx2) * 0.5
                lcy = (ly1 + ly2) * 0.5
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                dist = math.hypot(cx - lcx, cy - lcy)
                dist_norm = dist / max(1.0, math.hypot(W, H))

                # hard gate if it's way too far and we saw target recently
                if (t - last_seen_t) < sticky_seconds and dist_norm > dist_gate2_norm:
                    continue
                if dist_norm > dist_gate_norm and (t - last_seen_t) < sticky_seconds:
                    # penalty but not instant reject
                    color_score *= 0.80

            # include YOLO conf lightly (don’t let it dominate)
            score = color_score + 0.15 * conf

            if score > best_score:
                best_score = score
                best_box = box
                best_color = color_score
                best_dist = dist_norm

        present = False
        chosen = None
        if best_box is not None and (best_color is not None) and best_color >= color_threshold:
            present = True
            chosen = list(best_box)
            last_box = best_box
            last_seen_t = t
            present_times.append(t)
        else:
            # sticky: if we just saw him, keep presence briefly (helps occlusion)
            if (t - last_seen_t) <= sticky_seconds:
                present = True
                present_times.append(t)

        if len(debug_samples) < 250:
            debug_samples.append({
                "t": float(t),
                "present": bool(present),
                "chosen": chosen,
                "score": float(best_score) if best_score > -1e8 else None,
                "color": float(best_color) if best_color is not None else None,
                "dist": float(best_dist) if best_dist is not None else None,
                "boxes": len(dets),
            })

        f += 1

    cap.release()

    # --- convert present_times -> spans ---
    spans: List[Tuple[float, float]] = []
    if present_times:
        present_times.sort()
        start = present_times[0]
        prev = present_times[0]
        for t in present_times[1:]:
            if (t - prev) <= (detect_stride / fps) + gap_merge:
                prev = t
            else:
                spans.append((start, prev))
                start = t
                prev = t
        spans.append((start, prev))

    # expand with pre/post roll and merge again
    expanded: List[Tuple[float, float]] = []
    for (s, e) in spans:
        s2 = max(0.0, s - pre_roll)
        e2 = min(duration_s, e + post_roll)
        expanded.append((s2, e2))

    merged: List[Tuple[float, float]] = []
    for (s, e) in sorted(expanded):
        if not merged:
            merged.append((s, e))
        else:
            ls, le = merged[-1]
            if s <= le + gap_merge:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))

    # enforce min clip length
    final_spans = []
    for (s, e) in merged:
        if (e - s) >= min_clip_len:
            final_spans.append((float(s), float(e)))

    debug = {
        "seed": {
            "seed_count": len(seed_boxes),
            "seed_frames": seed_samples,
        },
        "fps": float(fps),
        "W": int(W),
        "H": int(H),
        "total_frames": int(total_frames),
        "duration_s": float(duration_s),
        "min_clip_len": float(min_clip_len),
        "gap_merge": float(gap_merge),
        "sticky_seconds": float(sticky_seconds),
        "pre_roll": float(pre_roll),
        "post_roll": float(post_roll),
        "detect_stride": int(detect_stride),
        "yolo_conf": float(yolo_conf),
        "color_threshold": float(color_threshold),
        "dist_gate_norm": float(dist_gate_norm),
        "dist_gate2_norm": float(dist_gate2_norm),
        "sampled_detect_frames": int(sampled_frames),
        "debug_samples": debug_samples,
        "spans": final_spans,
    }

    return final_spans, debug


def process_job(job_id: str) -> Dict[str, Any]:
    """
    This is what RQ runs: worker.tasks.process_job(job_id)
    Critical behavior:
    - Updates meta.json continuously
    - On ANY exception: status=error (no infinite "queued")
    """
    meta_path = _job_dir(job_id) / "meta.json"
    meta = _read_json(meta_path, {})
    video_path = meta.get("video_path") or str(_job_dir(job_id) / "in.mp4")
    setup = meta.get("setup") or {}

    clicks = setup.get("clicks") or []
    jersey_color = setup.get("jersey_color") or "#203524"

    # tuning knobs (safe defaults)
    extend_sec = float(setup.get("extend_sec", 2) or 2)

    try:
        _set_status(job_id, status="processing", stage="processing", progress=15, message="Starting tracking...")

        spans, debug = track_presence_spans(
            video_path=video_path,
            clicks=clicks,
            jersey_hex=jersey_color,
            min_clip_len=20.0,
            gap_merge=2.0,
            sticky_seconds=1.5,
            pre_roll=4.0,
            post_roll=1.5,
            detect_stride=3,
            yolo_conf=0.25,
            color_threshold=1.05,
            dist_gate_norm=0.18,
            dist_gate2_norm=0.35,
        )

        # extend spans by extend_sec on both sides
        spans2: List[Tuple[float, float]] = []
        for s, e in spans:
            spans2.append((max(0.0, s - extend_sec), max(0.0, e + extend_sec)))

        _set_status(job_id, progress=60, message="Cutting clips...")

        clips_dir = _job_dir(job_id) / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        clips = []
        list_file = _job_dir(job_id) / "concat_list.txt"
        lines = []

        for i, (s, e) in enumerate(spans2, start=1):
            out_path = clips_dir / f"clip_{i:03d}.mp4"
            _ffmpeg_cut(video_path, str(out_path), float(s), float(e))
            clips.append({
                "start": float(s),
                "end": float(e),
                "path": str(out_path),
                "url": f"/data/jobs/{job_id}/clips/{out_path.name}",
            })
            lines.append(f"file '{out_path.as_posix()}'\n")

        list_file.write_text("".join(lines), encoding="utf-8")

        combined_path = _job_dir(job_id) / "combined.mp4"
        if clips:
            _ffmpeg_concat(str(list_file), str(combined_path))

        _set_status(
            job_id,
            status="done",
            stage="done",
            progress=100,
            message="Done.",
            clips=clips,
            combined_path=str(combined_path),
            combined_url=f"/data/jobs/{job_id}/combined.mp4",
            debug=debug,
        )
        return _read_json(meta_path, {})

    except Exception as e:
        _set_status(
            job_id,
            status="error",
            stage="error",
            progress=100,
            message=f"ERROR: {type(e).__name__}: {e}",
        )
        raise

