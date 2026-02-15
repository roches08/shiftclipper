import os
import io
import math
import json
import time
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from rq import get_current_job

# ---------------------------
# MAX-ACCURACY DEFAULTS (override via env vars)
# ---------------------------
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8x.pt")   # biggest = best accuracy
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "960"))         # 960/1280 more accurate than 640
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))        # lower = more recall (can raise for precision)
DETECT_STRIDE_DEFAULT = int(os.getenv("DETECT_STRIDE", "1"))  # 1 = detect every frame (slowest, best)
USE_HALF = os.getenv("YOLO_HALF", "1") == "1"            # fp16 on GPU (faster), usually fine

# NOTE: if you want absolute max accuracy and don't care about speed:
# export YOLO_IMGSZ=1280
# export DETECT_STRIDE=1
# export YOLO_WEIGHTS=yolov8x.pt

# Lazy-load YOLO
_yolo_model = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run(cmd: List[str]) -> None:
    subprocess.check_call(cmd)


def _ffprobe_fps_wh(path: str) -> Tuple[float, int, int]:
    # fps + width/height from ffprobe
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,width,height",
        "-of", "json",
        path,
    ]
    out = subprocess.check_output(cmd).decode("utf-8")
    data = json.loads(out)
    st = data["streams"][0]
    w = int(st["width"])
    h = int(st["height"])
    fr = st["avg_frame_rate"]
    # fr like "60000/1001"
    if "/" in fr:
        a, b = fr.split("/")
        fps = float(a) / float(b)
    else:
        fps = float(fr)
    return fps, w, h


def _safe_hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        return (0, 255, 0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_WEIGHTS)
    return _yolo_model


def _yolo_person_boxes(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """
    Returns list of (x1,y1,x2,y2,conf) for person class.
    """
    model = _load_yolo()

    # Ultralytics expects RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # NOTE: device auto-selects CUDA in RunPod if available
    results = model.predict(
        rgb,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        verbose=False,
        half=USE_HALF,
        classes=[0],   # person
    )
    r0 = results[0]
    out = []
    if r0.boxes is None:
        return out

    boxes = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        out.append((int(x1), int(y1), int(x2), int(y2), float(c)))
    return out


def _bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def _crop(frame: np.ndarray, bb: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bb
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return frame[0:1, 0:1]
    return frame[y1:y2, x1:x2]


def _color_match_score(crop_bgr: np.ndarray, target_bgr: Tuple[int, int, int]) -> float:
    """
    Higher = closer to jersey color.
    Uses median in HSV (more stable vs lighting) and distance to target converted to HSV.
    """
    if crop_bgr.size == 0:
        return -1e9

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    med = np.median(hsv.reshape(-1, 3), axis=0)  # H,S,V

    # target BGR -> HSV
    tbgr = np.uint8([[list(target_bgr)]])
    thsv = cv2.cvtColor(tbgr, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)

    # hue wraps; handle hue circular distance
    dh = float(abs(med[0] - thsv[0]))
    dh = min(dh, 180.0 - dh)
    ds = float(abs(med[1] - thsv[1]))
    dv = float(abs(med[2] - thsv[2]))

    # weighted distance (tune weights)
    dist = (2.0 * dh) + (0.5 * ds) + (0.2 * dv)
    return -dist


@dataclass
class Click:
    t: float  # seconds
    x: float  # normalized 0-1
    y: float  # normalized 0-1


def _pick_box_from_click(
    boxes: List[Tuple[int, int, int, int, float]],
    click_xy_px: Tuple[float, float],
) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    cx, cy = click_xy_px
    best = None
    best_d = 1e18
    for x1, y1, x2, y2, conf in boxes:
        if cx >= x1 and cx <= x2 and cy >= y1 and cy <= y2:
            # inside box -> prefer smallest box that contains click
            area = (x2 - x1) * (y2 - y1)
            d = area
        else:
            bx, by = _bbox_center((x1, y1, x2, y2))
            d = (bx - cx) ** 2 + (by - cy) ** 2
        if d < best_d:
            best_d = d
            best = (x1, y1, x2, y2)
    return best


def run_tracker(
    video_path: str,
    out_dir: str,
    camera_mode: str,
    player_number: str,
    jersey_color: str,
    clicks: List[Dict],
    extend_sec: int = 20,
    verify_mode: bool = True,
) -> Dict:
    """
    Accuracy-first baseline tracker:
    - YOLO person detections (v8x + higher imgsz)
    - seed from 1-3 clicks (or more)
    - greedy tracking by IOU + jersey-color score (helps avoid ID switches)
    - clip span = when target present
    """
    _ensure_dir(out_dir)
    clips_dir = os.path.join(out_dir, "clips")
    _ensure_dir(clips_dir)

    fps, W, H = _ffprobe_fps_wh(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0

    target_bgr = _safe_hex_to_bgr(jersey_color)

    # Convert clicks
    click_objs = [Click(**c) for c in clicks]
    click_objs = sorted(click_objs, key=lambda c: c.t)

    # Seed: find best bboxes at click times
    seed_boxes = []
    seed_frames_debug = []

    for c in click_objs:
        frame_idx = max(0, min(total_frames - 1, int(round(c.t * fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        boxes = _yolo_person_boxes(frame)
        px = c.x * W
        py = c.y * H
        picked = _pick_box_from_click(boxes, (px, py))
        seed_frames_debug.append({
            "t": c.t,
            "frame": frame_idx,
            "dets": len(boxes),
            "picked": list(picked) if picked else None,
        })
        if picked:
            seed_boxes.append(picked)

    if not seed_boxes:
        return {
            "status": "error",
            "message": "No seed boxes found from clicks. Try clicking torso when player is visible.",
            "debug": {"seed_frames": seed_frames_debug},
        }

    # Combine seed into one initial bbox (median coordinates)
    sx1 = int(np.median([b[0] for b in seed_boxes]))
    sy1 = int(np.median([b[1] for b in seed_boxes]))
    sx2 = int(np.median([b[2] for b in seed_boxes]))
    sy2 = int(np.median([b[3] for b in seed_boxes]))
    current_bb = (sx1, sy1, sx2, sy2)

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detect_stride = DETECT_STRIDE_DEFAULT  # max accuracy default: 1
    last_dets = []
    present_flags = [False] * total_frames

    # tracking params
    min_iou = 0.10
    color_weight = 0.75  # higher = rely more on jersey color
    iou_weight = 1.0

    debug_samples = []
    sample_every = max(1, int(round(fps * 0.5)))  # ~2 per second

    for fi in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        if fi % detect_stride == 0:
            last_dets = _yolo_person_boxes(frame)

        best = None
        best_score = -1e18

        for x1, y1, x2, y2, conf in last_dets:
            bb = (x1, y1, x2, y2)

            iou = _bbox_iou(current_bb, bb)
            if iou <= 0 and fi > 0:
                # allow reacquire, but still needs some reason
                pass

            crop = _crop(frame, bb)
            cscore = _color_match_score(crop, target_bgr)

            score = (iou_weight * iou) + (color_weight * (cscore / 100.0)) + (0.05 * conf)

            if score > best_score:
                best_score = score
                best = bb

        if best is not None:
            # simple gate: accept if it isn't totally insane
            iou = _bbox_iou(current_bb, best)
            crop = _crop(frame, best)
            cscore = _color_match_score(crop, target_bgr)
            accept = (iou > min_iou) or (cscore > -40)  # -40 ~ “somewhat close” in HSV space

            if accept:
                current_bb = best
                present_flags[fi] = True

        if fi % sample_every == 0:
            debug_samples.append({
                "t": fi / fps,
                "present": bool(present_flags[fi]),
                "chosen": list(current_bb) if present_flags[fi] else None,
                "boxes": len(last_dets),
                "score": float(best_score) if best_score is not None else None,
                "detect_stride": detect_stride,
                "yolo_weights": YOLO_WEIGHTS,
                "imgsz": YOLO_IMGSZ,
                "conf": YOLO_CONF,
            })

        # Update progress in RQ (if running in worker)
        job = get_current_job()
        if job and fi % max(1, int(fps)) == 0:
            pct = int(100 * fi / max(1, total_frames))
            job.meta["progress"] = pct
            job.save_meta()

    cap.release()

    # Build spans from present_flags
    spans = []
    in_span = False
    s = 0
    for i, p in enumerate(present_flags):
        if p and not in_span:
            in_span = True
            s = i
        elif (not p) and in_span:
            in_span = False
            e = i - 1
            spans.append((s / fps, e / fps))
    if in_span:
        spans.append((s / fps, (len(present_flags) - 1) / fps))

    # Expand spans with preroll/postroll and merge small gaps
    pre_roll = 1.25
    post_roll = 0.9
    gap_merge = 1.25
    min_clip_len = 3.0

    expanded = []
    for a, b in spans:
        a2 = max(0.0, a - pre_roll)
        b2 = min(duration_s, b + post_roll)
        expanded.append((a2, b2))

    expanded.sort()
    merged = []
    for a, b in expanded:
        if not merged:
            merged.append([a, b])
        else:
            if a <= merged[-1][1] + gap_merge:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])

    final_spans = [(a, b) for a, b in merged if (b - a) >= min_clip_len]

    # Write clips
    clips = []
    for idx, (a, b) in enumerate(final_spans, start=1):
        outp = os.path.join(clips_dir, f"clip_{idx:03d}.mp4")
        _run([
            "ffmpeg", "-y",
            "-ss", f"{a:.3f}",
            "-to", f"{b:.3f}",
            "-i", video_path,
            "-c", "copy",
            outp
        ])
        clips.append({
            "start": a,
            "end": b,
            "path": outp,
            "url": outp,  # API may rewrite this elsewhere
        })

    # Combine
    combined_path = os.path.join(out_dir, "combined.mp4")
    if clips:
        concat_list = os.path.join(out_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for c in clips:
                f.write(f"file '{c['path']}'\n")
        _run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            combined_path
        ])
    else:
        combined_path = None

    return {
        "status": "done",
        "clips": clips,
        "combined_path": combined_path,
        "debug": {
            "seed": {
                "ok": True,
                "seed_bbox": [sx1, sy1, sx2, sy2],
                "seed_frames": seed_frames_debug,
                "seed_count": len(seed_boxes),
                "fps": fps,
                "W": W,
                "H": H,
            },
            "fps": fps,
            "W": W,
            "H": H,
            "total_frames": total_frames,
            "duration_s": duration_s,
            "min_clip_len": min_clip_len,
            "gap_merge": gap_merge,
            "pre_roll": pre_roll,
            "post_roll": post_roll,
            "detect_stride": detect_stride,
            "yolo_conf": YOLO_CONF,
            "yolo_weights": YOLO_WEIGHTS,
            "yolo_imgsz": YOLO_IMGSZ,
            "debug_samples": debug_samples,
            "spans": final_spans,
        }
    }

