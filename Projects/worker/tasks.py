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

# DeepSORT tracker
from deep_sort_realtime.deepsort_tracker import DeepSort

"""
Worker: YOLO + DeepSORT tracker (identity persistence)

Pipeline:
1) Use clicks (2-3) to seed who the player is (seed bbox).
2) Run YOLO "person" detection at a stride on frames (supported source type).
3) Feed detections to DeepSORT to get stable track IDs.
4) Lock onto a target track ID based on click/seed proximity.
5) Build presence spans (sticky + merge gaps + pre/post roll + min clip)
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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    return float(out)


def _ffprobe_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    if "/" in out:
        a, b = out.split("/")
        a = float(a)
        b = float(b)
        return a / b if b else 30.0
    return float(out)


def _ensure_yolo(weights: str = "yolov8s.pt") -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(weights)
    return _YOLO_MODEL


def _norm_click_to_px(click: Dict[str, float], W: int, H: int) -> Tuple[int, int]:
    x = int(round(float(click["x"]) * W))
    y = int(round(float(click["y"]) * H))
    return x, y


def _pick_seed_bbox_from_clicks(
    cap: cv2.VideoCapture,
    yolo: YOLO,
    clicks: List[Dict[str, float]],
    fps: float,
    W: int,
    H: int,
    yolo_conf: float,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    For each click: run YOLO on that frame, pick the person box containing/closest to the click.
    Then fuse into a single seed bbox (median corners).
    """
    seed_frames_debug: List[Dict[str, Any]] = []
    picked_boxes: List[List[int]] = []

    for c in clicks:
        t = float(c["t"])
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            seed_frames_debug.append({"t": t, "frame": frame_idx, "dets": 0, "picked": None})
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = yolo.predict(rgb, verbose=False, conf=yolo_conf, classes=[0])  # class 0 = person
        boxes = []
        if res and len(res) > 0 and res[0].boxes is not None:
            for b in res[0].boxes.xyxy.cpu().numpy().tolist():
                x1, y1, x2, y2 = map(int, b[:4])
                boxes.append([x1, y1, x2, y2])

        cx, cy = _norm_click_to_px(c, W, H)

        picked = None
        best = 1e18
        for (x1, y1, x2, y2) in boxes:
            inside = (x1 <= cx <= x2) and (y1 <= cy <= y2)
            mx = (x1 + x2) * 0.5
            my = (y1 + y2) * 0.5
            d = (mx - cx) ** 2 + (my - cy) ** 2
            score = d * (0.25 if inside else 1.0)
            if score < best:
                best = score
                picked = [x1, y1, x2, y2]

        seed_frames_debug.append({"t": t, "frame": frame_idx, "dets": len(boxes), "picked": picked})
        if picked is not None:
            picked_boxes.append(picked)

    if not picked_boxes:
        # fallback seed box (center-ish) so pipeline doesn't crash
        seed = [int(W * 0.45), int(H * 0.35), int(W * 0.55), int(H * 0.75)]
        return seed, seed_frames_debug

    arr = np.array(picked_boxes, dtype=np.float32)
    seed = np.median(arr, axis=0).astype(int).tolist()
    return seed, seed_frames_debug


def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = (a_area + b_area - inter)
    return float(inter) / float(denom) if denom else 0.0


def _mean_rgb_in_bbox(frame_bgr: np.ndarray, bb_xyxy: List[int]) -> np.ndarray:
    """Mean RGB (normalized 0..1) in the central torso region of a bbox."""
    x1, y1, x2, y2 = bb_xyxy
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)

    cx1 = x1 + int(0.20 * w)
    cx2 = x1 + int(0.80 * w)
    cy1 = y1 + int(0.25 * h)
    cy2 = y1 + int(0.70 * h)

    cx1 = max(0, cx1)
    cy1 = max(0, cy1)
    cx2 = min(frame_bgr.shape[1], cx2)
    cy2 = min(frame_bgr.shape[0], cy2)

    roi = frame_bgr[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    m = rgb.reshape(-1, 3).mean(axis=0).astype(np.float32) / 255.0
    return m


def _color_sim(a_rgb_norm: np.ndarray, b_rgb_norm: np.ndarray) -> float:
    """Cosine similarity in RGB space (0..1-ish)."""
    a = a_rgb_norm.astype(np.float32)
    b = b_rgb_norm.astype(np.float32)
    na = float(np.linalg.norm(a) + 1e-6)
    nb = float(np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b) / (na * nb))


def track_presence_spans(
    video_path: Path,
    clicks: List[Dict[str, float]],
    jersey_hex: Optional[str] = None,
    opponent_hex: Optional[str] = None,
    player_number: Optional[str] = None,
    detect_stride: int = 3,
    yolo_conf: float = 0.25,
    dist_gate_norm: float = 0.18,
    sticky_seconds: float = 1.5,
    gap_merge: float = 2.0,
    pre_roll: float = 4.0,
    post_roll: float = 1.5,
    min_clip_len: float = 20.0,
    color_sim_min: float = 0.86,
    ocr_stride_s: float = 1.0,
    device: Optional[str] = None,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    import torch
    import re
    try:
        import easyocr
    except Exception:
        easyocr = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = _ffprobe_fps(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = float(total_frames) / float(fps) if fps > 0 and total_frames > 0 else _ffprobe_duration(video_path)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    yolo = _ensure_yolo("yolov8s.pt")
    # Default: use GPU if available
    if device is None:
        device = 0 if bool(torch.cuda.is_available()) else 'cpu'

    # Jersey color signatures
    target_rgb = None
    opp_rgb = None
    if jersey_hex:
        h = jersey_hex.strip().lstrip('#')
        if len(h) == 6:
            target_rgb = np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float32) / 255.0
    if opponent_hex:
        h = opponent_hex.strip().lstrip('#')
        if len(h) == 6:
            opp_rgb = np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float32) / 255.0

    # OCR init (only if user provides a target number)
    ocr = None
    if player_number and easyocr is not None:
        try:
            ocr = easyocr.Reader(['en'], gpu=bool(torch.cuda.is_available()))
        except Exception:
            ocr = None

    # Build seed bbox using 2-3 clicks (recommended)
    seed_bbox, seed_frames_debug = _pick_seed_bbox_from_clicks(cap, yolo, clicks, fps, W, H, yolo_conf)

    # DeepSORT tracker
    tracker = DeepSort(
        max_age=30,
        n_init=2,
        max_iou_distance=0.7,
        max_cosine_distance=0.4,
        nn_budget=None,
        override_track_class=None,
    )

    # We will lock onto one track_id that overlaps seed_bbox early on
    target_track_id: Optional[int] = None

    present_flags = []
    debug_samples = []

    # rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    # distance gate in pixels
    dist_gate_px = dist_gate_norm * math.sqrt(W * W + H * H)

    last_good_bbox = seed_bbox[:]  # keep last bbox to help gating

    next_ocr_t = 0.0
    num_bad_streak = 0
    num_ok_streak = 0

    def _read_number(frame_bgr: np.ndarray, bb: List[int]) -> Optional[str]:
        if ocr is None:
            return None
        x1, y1, x2, y2 = bb
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        # torso crop
        cx1 = x1 + int(0.20 * w)
        cx2 = x1 + int(0.80 * w)
        cy1 = y1 + int(0.15 * h)
        cy2 = y1 + int(0.65 * h)
        cx1 = max(0, cx1); cy1 = max(0, cy1)
        cx2 = min(frame_bgr.shape[1], cx2); cy2 = min(frame_bgr.shape[0], cy2)
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        try:
            out = ocr.readtext(crop, detail=1, paragraph=False, allowlist='0123456789')
        except Exception:
            return None
        if not out:
            return None
        best = sorted(out, key=lambda x: float(x[2]), reverse=True)[0]
        txt = re.sub(r"\D+", "", str(best[1]))
        return txt or None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t = frame_idx / fps if fps > 0 else 0.0

        if frame_idx % detect_stride != 0:
            # just carry presence as unknown; we’ll fill with sticky later
            present_flags.append((t, None, None, None, None, None))
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = yolo.predict(
            rgb,
            verbose=False,
            conf=yolo_conf,
            classes=[0],
            device=device if device is not None else None,
        )

        dets = []
        boxes = []
        if res and len(res) > 0 and res[0].boxes is not None:
            for b, conf in zip(res[0].boxes.xyxy.cpu().numpy(), res[0].boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, b[:4].tolist())
                boxes.append([x1, y1, x2, y2])
                # DeepSORT wants: [x, y, w, h], confidence, class
                dets.append(([x1, y1, x2 - x1, y2 - y1], float(conf), "person"))

        tracks = tracker.update_tracks(dets, frame=frame)

        # Convert tracks to xyxy (with optional jersey-color gating)
        track_boxes = []
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            ltrb = trk.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            bb = [max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)]

            if target_rgb is not None:
                m = _mean_rgb_in_bbox(frame, bb)
                sim = _color_sim(m, target_rgb)
                if sim < float(color_sim_min):
                    continue
                if opp_rgb is not None:
                    sim_opp = _color_sim(m, opp_rgb)
                    if sim_opp > sim:
                        continue

            track_boxes.append((trk.track_id, bb))

        chosen = None
        chosen_id = None
        score = None
        dist = None

        # Acquire target id if not set (use IoU with seed_bbox)
        if target_track_id is None:
            best_iou = 0.0
            best = None
            for tid, bb in track_boxes:
                i = _iou(bb, seed_bbox)
                if i > best_iou:
                    best_iou = i
                    best = (tid, bb)
            if best is not None and best_iou >= 0.10:
                target_track_id, chosen = best[0], best[1]
                chosen_id = target_track_id
                last_good_bbox = chosen[:]

        # If we have a target id, find it
        if target_track_id is not None:
            found = None
            for tid, bb in track_boxes:
                if tid == target_track_id:
                    found = bb
                    break

            if found is not None:
                # gate by distance from last bbox center (helps avoid ID switches)
                lx1, ly1, lx2, ly2 = last_good_bbox
                lcx, lcy = (lx1 + lx2) * 0.5, (ly1 + ly2) * 0.5
                fx1, fy1, fx2, fy2 = found
                fcx, fcy = (fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5
                d = math.sqrt((fcx - lcx) ** 2 + (fcy - lcy) ** 2)
                dist = d

                if d <= dist_gate_px:
                    chosen = found
                    chosen_id = target_track_id
                    last_good_bbox = chosen[:]

        present = chosen is not None
        chosen_num = None

        # Sparse jersey-number validation (prevents ID swaps when tracking gets confused)
        if present and chosen is not None and player_number and (t >= next_ocr_t):
            next_ocr_t = t + max(0.25, float(ocr_stride_s))
            chosen_num = _read_number(frame, chosen)
            if chosen_num is None:
                pass
            elif str(chosen_num) == str(player_number):
                num_ok_streak += 1
                num_bad_streak = 0
            else:
                num_bad_streak += 1
                num_ok_streak = 0

        if player_number and num_bad_streak >= 2:
            present = False

        # score: use IoU with last_good_bbox (higher is better)
        if chosen is not None:
            score = _iou(chosen, last_good_bbox)

        present_flags.append((t, present, chosen, score, chosen_num, dist))

        # keep a small debug sample
        if len(debug_samples) < 400:
            debug_samples.append({
                "t": t,
                "present": bool(present),
                "chosen": chosen,
                "score": score,
                "dist": dist,
                "ocr": chosen_num,
                "boxes": len(boxes),
                "tracks": len(track_boxes),
                "target_id": target_track_id
            })

        frame_idx += 1

    cap.release()

    # ---- Convert sparse present_flags into spans with sticky/gap merge ----
    times = [x[0] for x in present_flags]
    pres = [x[1] for x in present_flags]

    # Treat None as False initially; sticky below will recover short misses
    pres_bool = [bool(p) for p in pres]

    # Sticky fill (if present within +/- sticky_seconds, keep present)
    if sticky_seconds > 0 and fps > 0:
        sticky_frames = int(round(sticky_seconds * fps))
        pres2 = pres_bool[:]
        for i in range(len(pres_bool)):
            if pres_bool[i]:
                lo = max(0, i - sticky_frames)
                hi = min(len(pres_bool) - 1, i + sticky_frames)
                for k in range(lo, hi + 1):
                    pres2[k] = True
        pres_bool = pres2

    spans: List[Tuple[float, float]] = []
    in_span = False
    s0 = 0.0

    for i, p in enumerate(pres_bool):
        t = times[i]
        if p and not in_span:
            in_span = True
            s0 = t
        elif (not p) and in_span:
            in_span = False
            spans.append((s0, t))

    if in_span:
        spans.append((s0, duration_s))

    # Merge gaps smaller than gap_merge
    if gap_merge > 0 and spans:
        merged = [spans[0]]
        for a, b in spans[1:]:
            pa, pb = merged[-1]
            if a - pb <= gap_merge:
                merged[-1] = (pa, b)
            else:
                merged.append((a, b))
        spans = merged

    # Apply pre/post roll and min clip length
    final_spans = []
    for a, b in spans:
        a2 = max(0.0, a - pre_roll)
        b2 = min(duration_s, b + post_roll)
        if (b2 - a2) >= min_clip_len:
            final_spans.append((a2, b2))

    debug = {
        "seed": {
            "seed_bbox": seed_bbox,
            "seed_frames": seed_frames_debug,
            "seed_count": len(clicks),
        },
        "fps": fps,
        "W": W,
        "H": H,
        "total_frames": total_frames,
        "duration_s": duration_s,
        "min_clip_len": min_clip_len,
        "gap_merge": gap_merge,
        "sticky_seconds": sticky_seconds,
        "pre_roll": pre_roll,
        "post_roll": post_roll,
        "detect_stride": detect_stride,
        "yolo_conf": yolo_conf,
        "device": device,
        "jersey_hex": jersey_hex,
        "opponent_hex": opponent_hex,
        "player_number": str(player_number) if player_number is not None else None,
        "color_sim_min": color_sim_min,
        "ocr_stride_s": ocr_stride_s,
        "dist_gate_norm": dist_gate_norm,
        "dist_gate_px": dist_gate_px,
        "debug_samples": debug_samples,
        "spans": final_spans,
        "target_track_id": target_track_id,
    }

    return final_spans, debug


def _run(cmd: List[str]) -> None:
    subprocess.check_call(cmd)


def _cut_clip(in_path: Path, out_path: Path, start: float, end: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(in_path),
        "-c", "copy",
        str(out_path),
    ]
    _run(cmd)


def _concat_clips(clip_paths: List[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.parent / "concat_list.txt"
    list_file.write_text("\n".join([f"file '{p.as_posix()}'" for p in clip_paths]), encoding="utf-8")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    _run(cmd)


def process_job(job_id: str) -> Dict[str, Any]:
    job_dir = _job_dir(job_id)
    in_path = job_dir / "in.mp4"
    setup_path = job_dir / "setup.json"
    status_path = job_dir / "status.json"

    setup = _read_json(setup_path, {})
    clicks = setup.get("clicks", []) or []

    # Recommend 2-3 clicks; but allow 1+
    if len(clicks) < 1:
        raise RuntimeError("No clicks provided. Provide at least 1 click to seed tracking.")

    # Parameters (safe defaults)
    detect_stride = int(setup.get("detect_stride", 3))
    yolo_conf = float(setup.get("yolo_conf", 0.25))
    dist_gate_norm = float(setup.get("dist_gate_norm", 0.18))

    sticky_seconds = float(setup.get("sticky_seconds", 1.5))
    gap_merge = float(setup.get("gap_merge", 2.0))
    pre_roll = float(setup.get("pre_roll", 4.0))
    post_roll = float(setup.get("post_roll", 1.5))
    min_clip_len = float(setup.get("min_clip_len", 20.0))

    # Update job status
    status = _read_json(status_path, {})
    status.update({
        "job_id": job_id,
        "status": "running",
        "stage": "tracking",
        "progress": 5,
        "message": "Tracking player…",
        "updated_at": time.time(),
        "setup": setup,
    })
    _write_json(status_path, status)

    spans, debug = track_presence_spans(
        in_path,
        clicks=clicks,
        detect_stride=detect_stride,
        yolo_conf=yolo_conf,
        dist_gate_norm=dist_gate_norm,
        sticky_seconds=sticky_seconds,
        gap_merge=gap_merge,
        pre_roll=pre_roll,
        post_roll=post_roll,
        min_clip_len=min_clip_len,
        device="0" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else None,
    )

    # Cut clips
    clips_dir = job_dir / "clips"
    clip_infos = []
    clip_paths = []

    status.update({"stage": "clipping", "progress": 60, "message": "Cutting clips…", "updated_at": time.time()})
    _write_json(status_path, status)

    for i, (a, b) in enumerate(spans, start=1):
        outp = clips_dir / f"clip_{i:03d}.mp4"
        _cut_clip(in_path, outp, a, b)
        clip_paths.append(outp)
        clip_infos.append({
            "start": a,
            "end": b,
            "path": str(outp),
            "url": f"/data/jobs/{job_id}/clips/{outp.name}",
        })

    combined_path = job_dir / "combined.mp4"
    if clip_paths:
        status.update({"stage": "combining", "progress": 85, "message": "Combining clips…", "updated_at": time.time()})
        _write_json(status_path, status)
        _concat_clips(clip_paths, combined_path)

    status.update({
        "status": "done",
        "stage": "done",
        "progress": 100,
        "message": "Done.",
        "updated_at": time.time(),
        "clips": clip_infos,
        "combined_path": str(combined_path),
        "combined_url": f"/data/jobs/{job_id}/combined.mp4",
        "debug": debug,
    })
    _write_json(status_path, status)
    return status
