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
from deep_sort_realtime.deepsort_tracker import DeepSort

"""
Phase 3 tracker (1–3 clicks):
- YOLO person detection
- DeepSORT for identity persistence
- Target appearance signature (HSV torso histogram) built from click frames
- Auto re-acquire when lost / after camera cuts using appearance + proximity + track stability
- ROI mode when locked, expands to global when lost

Outputs:
- clips/*.mp4 + combined.mp4
- results.json with debug info
"""

BASE_DIR = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
JOBS_DIR = BASE_DIR / "data" / "jobs"

_YOLO_MODEL: Optional[YOLO] = None


# -------------------------
# IO helpers
# -------------------------
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


# -------------------------
# ffmpeg helpers
# -------------------------
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


# -------------------------
# Vision helpers
# -------------------------
def _get_yolo(weights: str) -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None or getattr(_YOLO_MODEL, "ckpt_path", None) != weights:
        _YOLO_MODEL = YOLO(weights)
        try:
            _YOLO_MODEL.ckpt_path = weights
        except Exception:
            pass
    return _YOLO_MODEL


def _yolo_person_boxes(frame_bgr: np.ndarray, conf: float, yolo_weights: str) -> List[Tuple[int, int, int, int, float]]:
    model = _get_yolo(yolo_weights)
    # Ultralytics expects BGR OK; it handles conversion internally
    res = model.predict(frame_bgr, conf=conf, verbose=False)
    if not res:
        return []
    r0 = res[0]
    if r0.boxes is None:
        return []
    boxes = []
    for b in r0.boxes:
        cls = int(b.cls.item()) if b.cls is not None else -1
        if cls != 0:  # COCO person
            continue
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        c = float(b.conf.item()) if b.conf is not None else 0.0
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        boxes.append((x1, y1, x2, y2, c))
    return boxes


def _clamp_bb(bb: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bb
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _center(bb: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bb
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _torso_crop(frame_bgr: np.ndarray, bb: Tuple[int, int, int, int]) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = _clamp_bb(bb, W, H)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # torso: middle 40% vertically, middle 60% horizontally
    tx1 = x1 + int(0.20 * bw)
    tx2 = x2 - int(0.20 * bw)
    ty1 = y1 + int(0.20 * bh)
    ty2 = y1 + int(0.65 * bh)

    tx1, ty1, tx2, ty2 = _clamp_bb((tx1, ty1, tx2, ty2), W, H)
    crop = frame_bgr[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        crop = frame_bgr[y1:y2, x1:x2]
    return crop


def _hsv_hist_signature(frame_bgr: np.ndarray, bb: Tuple[int, int, int, int]) -> np.ndarray:
    crop = _torso_crop(frame_bgr, bb)
    if crop.size == 0:
        return np.zeros((16, 16), dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # hist over H,S (ignore V to reduce lighting sensitivity)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist = hist.astype(np.float32)
    s = float(hist.sum() + 1e-6)
    hist /= s
    return hist


def _hist_dist(a: np.ndarray, b: np.ndarray) -> float:
    # Bhattacharyya distance (0 is identical)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def _pick_nearest_box_to_click(
    boxes: List[Tuple[int, int, int, int, float]],
    click_xy_norm: Tuple[float, float],
    W: int,
    H: int
) -> Optional[Tuple[int, int, int, int]]:
    cx = float(click_xy_norm[0]) * W
    cy = float(click_xy_norm[1]) * H
    best = None
    best_d = 1e18
    for (x1, y1, x2, y2, _c) in boxes:
        ccx, ccy = _center((x1, y1, x2, y2))
        d = (ccx - cx) ** 2 + (ccy - cy) ** 2
        if d < best_d:
            best_d = d
            best = (x1, y1, x2, y2)
    return best


def _seed_from_clicks(
    video_path: str,
    clicks: List[Dict[str, Any]],
    conf: float,
    yolo_weights: str
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": "Could not open video"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    picked: List[Tuple[int, int, int, int]] = []
    seed_frames_debug: List[Dict[str, Any]] = []

    # Use click frames + a tiny ±window around each click to stabilize a 1-click setup
    win = 0.33  # seconds around click
    for c in clicks[:3]:  # we only need 1–3
        t0 = float(c.get("t", 0.0))
        x = float(c.get("x", 0.5))
        y = float(c.get("y", 0.5))

        for dt in (-win, 0.0, +win):
            t = max(0.0, t0 + dt)
            frame_idx = int(round(t * fps))
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
        return {"ok": False, "error": "No seed detections found from clicks", "seed_frames": seed_frames_debug, "fps": fps, "W": W, "H": H}

    # Median bbox
    areas = np.array([(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in picked], dtype=np.float32)
    med_area = float(np.median(areas))
    keep = []
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

    return {"ok": True, "seed_bbox": (xs1, ys1, xs2, ys2), "seed_frames": seed_frames_debug, "seed_count": len(picked), "fps": fps, "W": W, "H": H}


def _build_presence_spans(
    times_present: List[float],
    gap_merge: float,
    pre_roll: float,
    post_roll: float,
    min_len: float,
    max_clip_seconds: float = 120.0
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

    # merge tiny spans into neighbors if possible, then enforce min_len and split long
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


def _downsample_gray(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA)
    return g


def _scene_cut(prev_small: Optional[np.ndarray], cur_small: np.ndarray, thresh: float) -> bool:
    if prev_small is None:
        return False
    # normalized mean abs diff
    d = float(np.mean(np.abs(cur_small.astype(np.float32) - prev_small.astype(np.float32))) / 255.0)
    return d > thresh


# -------------------------
# main worker
# -------------------------
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
        _set_status(job_id, status="error", stage="error", progress=0, error="No clicks found. Click the player 1–3 times.")
        return {"status": "error", "error": "No clicks"}

    # -------------------------
    # Accuracy-first defaults (no JSON required)
    # -------------------------
    yolo_weights = str(setup.get("yolo_weights", "yolov8s.pt"))
    conf = float(setup.get("yolo_conf", 0.30))

    # "Best" on L4: stride=1 or 2
    detect_stride = int(setup.get("detect_stride", 1))  # 1 = max accuracy
    if detect_stride < 1:
        detect_stride = 1

    # Clip shaping
    min_clip_len = float(setup.get("min_clip_len", 3.0))
    gap_merge = float(setup.get("gap_merge", 1.25))
    pre_roll = float(setup.get("pre_roll", 1.25))
    post_roll = float(setup.get("post_roll", 0.90))
    sticky_seconds = float(setup.get("sticky_seconds", 1.20))
    max_clip_seconds = float(setup.get("max_clip_seconds", 120.0))  # allow long combined

    # DeepSORT tuning for hockey (more forgiving through occlusion)
    tracker = DeepSort(
        max_age=int(setup.get("ds_max_age", 120)),
        n_init=int(setup.get("ds_n_init", 2)),
        max_iou_distance=float(setup.get("ds_max_iou", 0.8)),
        max_cosine_distance=float(setup.get("ds_max_cos", 0.20)),
        nn_budget=int(setup.get("ds_nn_budget", 200)),
    )

    # ROI: fast when locked; auto-expands when lost
    roi_enable = bool(setup.get("roi_enable", True))
    roi_pad_frac_locked = float(setup.get("roi_pad_frac", 0.55))
    roi_pad_frac_lost = float(setup.get("roi_pad_frac_lost", 0.90))

    # Reacquire behavior
    lost_reacquire_after = float(setup.get("lost_reacquire_after", 0.60))  # seconds lost before global reacquire
    global_reacquire_every = float(setup.get("global_reacquire_every", 2.0))  # even when locked, sanity check
    scene_cut_thresh = float(setup.get("scene_cut_thresh", 0.22))  # broadcast cuts

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

    # Build target appearance signature from seed bbox around earliest click
    click_ts = [float(c.get("t", 0.0)) for c in clicks[:3]]
    if not click_ts:
        click_ts = [0.0]
    sample_ts = []
    for t0 in click_ts:
        for dt in (-0.33, 0.0, 0.33):
            sample_ts.append(max(0.0, t0 + dt))
    sigs = []
    for t0 in sample_ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)))
        ok0, frame0 = cap.read()
        if ok0:
            sigs.append(_hsv_hist_signature(frame0, seed_bbox))
    target_sig = np.mean(np.stack(sigs, axis=0), axis=0) if sigs else np.zeros((16, 16), dtype=np.float32)

    # tracking state
    times_present: List[float] = []
    last_center = _center(seed_bbox)
    last_present_t: Optional[float] = None
    lost_since_t: Optional[float] = None

    target_track_id: Optional[int] = None
    target_stable_hits = 0  # how many consecutive frames we see the target

    prev_small: Optional[np.ndarray] = None
    last_global_reacq_t = -1e9

    debug_samples: List[Dict[str, Any]] = []
    max_debug = 180

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    sampled = 0

    _set_status(job_id, status="processing", stage="tracking", progress=45, message="Tracking (Phase 3: DeepSORT + appearance)…")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps if fps > 0 else 0.0
        do_detect = (frame_idx % detect_stride == 0)

        present_now = False
        chosen_bb = None
        used_roi = False
        boxes_count = 0
        chosen_reason = ""
        chosen_score = None

        # scene cut detect (helps broadcast games a lot)
        small = _downsample_gray(frame)
        if _scene_cut(prev_small, small, scene_cut_thresh):
            target_track_id = None
            target_stable_hits = 0
            lost_since_t = t
        prev_small = small

        if do_detect:
            sampled += 1

            # Determine ROI mode
            frame_for_det = frame
            x_off, y_off = 0, 0
            pad_frac = roi_pad_frac_locked
            if lost_since_t is not None and (t - lost_since_t) >= lost_reacquire_after:
                pad_frac = roi_pad_frac_lost  # expand when lost

            if roi_enable and (target_track_id is not None or (lost_since_t is None or (t - lost_since_t) < lost_reacquire_after)):
                pad_w = int(pad_frac * W)
                pad_h = int(pad_frac * H)
                cx, cy = int(last_center[0]), int(last_center[1])
                x1 = max(0, cx - pad_w // 2)
                x2 = min(W, cx + pad_w // 2)
                y1 = max(0, cy - pad_h // 2)
                y2 = min(H, cy + pad_h // 2)
                if (x2 - x1) > 320 and (y2 - y1) > 180:
                    frame_for_det = frame[y1:y2, x1:x2]
                    x_off, y_off = x1, y1
                    used_roi = True

            # periodic global sanity reacquire (prevents drift)
            want_global_sanity = (t - last_global_reacq_t) >= global_reacquire_every
            if want_global_sanity and target_track_id is not None and (lost_since_t is None):
                # every N seconds, briefly use full frame to ensure we didn't drift onto wrong person
                frame_for_det = frame
                x_off, y_off = 0, 0
                used_roi = False
                last_global_reacq_t = t

            boxes = _yolo_person_boxes(frame_for_det, conf=conf, yolo_weights=yolo_weights)
            boxes_count = len(boxes)

            detections = []
            for (x1, y1, x2, y2, c) in boxes:
                # map ROI -> full
                x1 += x_off; x2 += x_off
                y1 += y_off; y2 += y_off

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                area_frac = (bw * bh) / float(W * H + 1e-6)
                if area_frac < 0.002 or area_frac > 0.28:
                    continue

                ar = bh / float(bw + 1e-6)
                if ar < 1.2 or ar > 6.5:
                    continue

                detections.append(([x1, y1, bw, bh], float(c), "person"))

            tracks = tracker.update_tracks(detections, frame=frame)

            # Candidate scoring:
            # - If target_track_id exists: prefer that track unless it looks wrong (appearance drift)
            # - Else: pick best by appearance + proximity
            best = None
            best_score = 1e18
            best_track_id = None

            # Normalization for proximity
            diag = math.sqrt(W * W + H * H) + 1e-6

            # First, collect track candidates
            candidates = []
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                ltrb = tr.to_ltrb()
                x1, y1, x2, y2 = [int(round(v)) for v in ltrb]
                bb = _clamp_bb((x1, y1, x2, y2), W, H)
                if (bb[2] - bb[0]) < 8 or (bb[3] - bb[1]) < 8:
                    continue
                sig = _hsv_hist_signature(frame, bb)
                ad = _hist_dist(sig, target_sig)  # 0..1-ish
                cx, cy = _center(bb)
                pd = math.sqrt((cx - last_center[0]) ** 2 + (cy - last_center[1]) ** 2) / diag  # 0..1
                candidates.append((tr.track_id, bb, ad, pd))

            # If we already have a target ID, try to stick to it, but validate appearance
            if target_track_id is not None:
                for tid, bb, ad, pd in candidates:
                    if tid == target_track_id:
                        # lock score: mostly appearance; proximity secondary
                        score = (1.60 * ad) + (0.60 * pd)
                        best = bb
                        best_score = score
                        best_track_id = tid
                        chosen_reason = "locked_id"
                        break

                # If the locked ID isn't present, or looks way off, allow re-acquire
                if best is None or best_score is None or best_score > 0.55:
                    target_track_id = None
                    target_stable_hits = 0

            # Acquire target if needed
            if target_track_id is None and candidates:
                # stronger proximity gating when we were just recently present, looser when lost
                lost_for = (t - lost_since_t) if lost_since_t is not None else 0.0
                prox_weight = 0.90 if lost_for < 0.5 else 0.40

                for tid, bb, ad, pd in candidates:
                    score = (1.50 * ad) + (prox_weight * pd)
                    if score < best_score:
                        best_score = score
                        best = bb
                        best_track_id = tid
                        chosen_reason = "reacquire"

                # accept if appearance is decent
                if best is not None and best_track_id is not None and best_score < 0.60:
                    target_track_id = int(best_track_id)
                    target_stable_hits = 0

            # Presence decision
            if best is not None and best_track_id is not None:
                present_now = True
                chosen_bb = best
                chosen_score = float(best_score)

        # sticky / lost tracking
        if present_now and chosen_bb is not None:
            last_center = _center(chosen_bb)
            last_present_t = t
            lost_since_t = None
            target_stable_hits += 1
            times_present.append(t)
        else:
            if lost_since_t is None:
                lost_since_t = t
            if last_present_t is not None and (t - last_present_t) <= sticky_seconds:
                times_present.append(t)

        # debug samples
        if do_detect and len(debug_samples) < max_debug:
            debug_samples.append({
                "t": t,
                "present": present_now,
                "chosen": list(chosen_bb) if chosen_bb else None,
                "track_id": target_track_id,
                "stable_hits": target_stable_hits,
                "score": chosen_score,
                "boxes": boxes_count,
                "roi": used_roi,
                "reason": chosen_reason,
                "lost_since": lost_since_t,
            })

        # progress
        if do_detect and sampled % 250 == 0 and total_frames > 0:
            prog = 45 + int(45.0 * (frame_idx / float(total_frames)))
            _set_status(job_id, progress=min(90, prog), message=f"Tracking… {t:.1f}s")

        frame_idx += 1

    cap.release()

    if not times_present:
        _set_status(job_id, status="error", stage="error", progress=0, error="No track found. Try 1–3 clicks on clear torso shots (not on bench).")
        results = {
            "status": "error",
            "job_id": job_id,
            "error": "No matches",
            "debug": {
                "seed": seed_info,
                "sampled_detect_frames": sampled,
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
        "player_number": setup.get("player_number"),
        "jersey_color": setup.get("jersey_color"),
        "clicks_count": len(clicks),
        "clicks": clicks,
        "clips": clips,
        "combined_path": combined_path,
        "combined_url": f"/data/jobs/{job_id}/combined.mp4",
        "debug": {
            "seed": seed_info,
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
            "yolo_conf": conf,
            "yolo_weights": yolo_weights,
            "ds": {
                "max_age": int(setup.get("ds_max_age", 120)),
                "n_init": int(setup.get("ds_n_init", 2)),
                "max_iou": float(setup.get("ds_max_iou", 0.8)),
                "max_cos": float(setup.get("ds_max_cos", 0.20)),
                "nn_budget": int(setup.get("ds_nn_budget", 200)),
            },
            "roi_enable": roi_enable,
            "roi_pad_frac_locked": roi_pad_frac_locked,
            "roi_pad_frac_lost": roi_pad_frac_lost,
            "lost_reacquire_after": lost_reacquire_after,
            "global_reacquire_every": global_reacquire_every,
            "scene_cut_thresh": scene_cut_thresh,
            "sampled_detect_frames": sampled,
            "debug_samples": debug_samples,
            "spans": spans,
            "target_track_id": None,  # final id can vary; debug_samples shows it over time
        }
    }

    _write_json(jd / "results.json", results)
    _set_status(job_id, status="done", stage="done", progress=100, message="Done.", clips=clips, combined_path=combined_path, combined_url=results["combined_url"])
    return results

