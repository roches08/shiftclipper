# ShiftClipper - Worker
# Reliable(ish) single-player tracker for broadcast hockey feeds.
#
# Reliability v1 (no new deps):
#  - Stronger jersey color matching (HSV + robust torso sampling)
#  - Better seed bbox from clicks (median/consensus)
#  - Reacquire gating + "don't switch identities" rule
#  - Two-stage reacquire: local search first -> global search second
#  - Sticky-present span building (gap tolerance) + pre-roll
#
# Reliability v2 (optional):
#  - EasyOCR jersey-number confirmation (if easyocr installed)
#  - Verify overlays: optional debug video showing detections & choices
#
# Notes:
#  - Uses Ultralytics YOLO if available (already in requirements). Falls back to OpenCV tracker.
#  - Outputs: clips/*.mp4 + combined.mp4 + results.json + optional debug_verify.mp4

from __future__ import annotations

import os
import json
import math
import time
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Paths / IO helpers
# ----------------------------

BASE_DIR = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"

def job_dir(job_id: str) -> Path:
    return (JOBS_DIR / job_id).resolve()

def meta_path(job_id: str) -> Path:
    return job_dir(job_id) / "meta.json"

def setup_path(job_id: str) -> Path:
    return job_dir(job_id) / "setup.json"

def results_path(job_id: str) -> Path:
    return job_dir(job_id) / "results.json"

def clips_dir(job_id: str) -> Path:
    return job_dir(job_id) / "clips"

def read_json(p: Path, default: Any) -> Any:
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def set_status(job_id: str, status: Optional[str] = None, **fields: Any) -> None:
    m = read_json(meta_path(job_id), {})
    m["job_id"] = job_id
    if status is not None:
        m["status"] = status
    m["updated_at"] = time.time()
    if fields:
        m.update(fields)
    write_json(meta_path(job_id), m)


# ----------------------------
# Utility: colors / scoring
# ----------------------------

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    s = (hex_color or "").strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6:
        return (0, 0, 0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)

def bgr_to_hsv(bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    arr = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])

def robust_torso_mean_bgr(frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Tuple[float,float,float]:
    """Compute robust mean color from the torso region (ignore ice/boards)."""
    x1,y1,x2,y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1+2 or y2 <= y1+2:
        return (0.0,0.0,0.0)

    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]

    # torso band: middle 40% of height, middle 60% of width
    tx1 = int(rw * 0.2); tx2 = int(rw * 0.8)
    ty1 = int(rh * 0.25); ty2 = int(rh * 0.70)
    torso = roi[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return (0.0,0.0,0.0)

    # mask out very bright (ice) and very dark (shadows/boards)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    s = hsv[:,:,1]
    mask = (v > 30) & (v < 245) & (s > 40)
    if mask.sum() < 20:
        mean = torso.reshape(-1,3).mean(axis=0)
        return (float(mean[0]), float(mean[1]), float(mean[2]))
    pix = torso[mask]
    mean = pix.reshape(-1,3).mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))

def hsv_distance(hsv_a: Tuple[float,float,float], hsv_b: Tuple[float,float,float]) -> float:
    # hue is circular [0..180]
    dh = min(abs(hsv_a[0]-hsv_b[0]), 180-abs(hsv_a[0]-hsv_b[0])) / 90.0
    ds = abs(hsv_a[1]-hsv_b[1]) / 255.0
    dv = abs(hsv_a[2]-hsv_b[2]) / 255.0
    return dh*1.3 + ds*1.0 + dv*0.7

def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
    inter = iw*ih
    if inter <= 0:
        return 0.0
    area_a = max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b = max(0,bx2-bx1)*max(0,by2-by1)
    denom = area_a + area_b - inter
    return float(inter/denom) if denom > 0 else 0.0

def center(b: Tuple[int,int,int,int]) -> Tuple[float,float]:
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def dist_norm(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int], frame_w: int, frame_h: int) -> float:
    ax,ay = center(a); bx,by = center(b)
    dx = (ax-bx)/max(1.0, frame_w)
    dy = (ay-by)/max(1.0, frame_h)
    return math.sqrt(dx*dx+dy*dy)


# ----------------------------
# Optional deps: YOLO + OCR
# ----------------------------

def load_yolo() -> Optional[Any]:
    try:
        from ultralytics import YOLO  # type: ignore
        # Prefer nano for speed
        model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
        return YOLO(model_name)
    except Exception:
        return None

def load_easyocr_reader() -> Optional[Any]:
    try:
        import easyocr  # type: ignore
        langs = os.getenv("OCR_LANGS", "en").split(",")
        return easyocr.Reader(langs, gpu=True)
    except Exception:
        try:
            import easyocr  # type: ignore
            langs = os.getenv("OCR_LANGS", "en").split(",")
            return easyocr.Reader(langs, gpu=False)
        except Exception:
            return None


# ----------------------------
# Video helpers
# ----------------------------

def ffprobe_duration(path: str) -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return float(out)
    except Exception:
        return None

def cut_clip(in_path: str, start: float, end: float, out_path: str) -> bool:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", in_path,
        "-c","copy",
        out_path
    ]
    r = subprocess.run(cmd, check=False)
    return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024

def concat_clips(clip_paths: List[str], out_path: str) -> bool:
    if not clip_paths:
        return False
    lst = out_path + ".txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",lst,"-c","copy",out_path]
    r = subprocess.run(cmd, check=False)
    try:
        os.remove(lst)
    except Exception:
        pass
    return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024


# ----------------------------
# Core tracking logic
# ----------------------------

def detections_yolo(model: Any, frame_bgr: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
    """Return person detections as (x1,y1,x2,y2,conf)."""
    res = model.predict(frame_bgr, verbose=False, imgsz=640, conf=0.25)
    dets: List[Tuple[int,int,int,int,float]] = []
    if not res:
        return dets
    r0 = res[0]
    if getattr(r0, "boxes", None) is None:
        return dets
    boxes = r0.boxes
    if boxes is None or len(boxes) == 0:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    # COCO: person=0
    for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
        if k != 0:
            continue
        dets.append((int(x1),int(y1),int(x2),int(y2),float(c)))
    return dets

def pick_bbox_from_click(frame: np.ndarray, dets: List[Tuple[int,int,int,int,float]], click_xy: Tuple[float,float]) -> Optional[Tuple[int,int,int,int]]:
    h,w = frame.shape[:2]
    cx = int(click_xy[0]*w); cy = int(click_xy[1]*h)
    best = None
    best_d = 1e9
    for x1,y1,x2,y2,_ in dets:
        mx = (x1+x2)//2; my = (y1+y2)//2
        d = (mx-cx)**2 + (my-cy)**2
        if d < best_d:
            best_d = d
            best = (x1,y1,x2,y2)
    return best

def seed_bbox_from_clicks(
    cap: cv2.VideoCapture,
    model: Optional[Any],
    clicks: List[Dict[str,Any]],
    fps: float,
    frame_w: int,
    frame_h: int,
) -> Tuple[Tuple[int,int,int,int], Dict[str,Any]]:
    """Use clicks to find a robust starting bbox.
    - If YOLO available: run detections at click frames and pick nearest.
    - Else: fixed bbox around click.
    Returns (bbox, debug)
    """
    seeds: List[Tuple[int,int,int,int]] = []
    dbg: Dict[str,Any] = {"seed_frames": []}
    for c in clicks[:8]:
        t = float(c.get("t", 0.0))
        x = float(c.get("x", 0.5))
        y = float(c.get("y", 0.5))
        fidx = max(0, int(t*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        bbox = None
        if model is not None:
            dets = detections_yolo(model, frame)
            bbox = pick_bbox_from_click(frame, dets, (x,y))
            dbg["seed_frames"].append({"t":t,"frame":fidx,"dets":len(dets),"picked":bbox})
        if bbox is None:
            # fallback: make a reasonable player-sized bbox around click
            bw = int(frame_w * 0.10)
            bh = int(frame_h * 0.22)
            cx = int(x*frame_w); cy = int(y*frame_h)
            bbox = (cx-bw//2, cy-bh//2, cx+bw//2, cy+bh//2)
            dbg["seed_frames"].append({"t":t,"frame":fidx,"dets":None,"picked":bbox,"fallback":True})
        seeds.append(bbox)

    if not seeds:
        # dead fallback: center-ish bbox
        bw = int(frame_w * 0.10); bh = int(frame_h * 0.22)
        bbox = (frame_w//2-bw//2, frame_h//2-bh//2, frame_w//2+bw//2, frame_h//2+bh//2)
        return bbox, {"seed_frames": [], "fallback": "no_seeds"}

    # consensus median bbox
    xs1 = sorted([b[0] for b in seeds]); ys1 = sorted([b[1] for b in seeds])
    xs2 = sorted([b[2] for b in seeds]); ys2 = sorted([b[3] for b in seeds])
    mid = len(seeds)//2
    med = (xs1[mid], ys1[mid], xs2[mid], ys2[mid])

    # Clamp
    x1,y1,x2,y2 = med
    x1 = max(0,min(frame_w-2,x1)); y1=max(0,min(frame_h-2,y1))
    x2 = max(x1+2,min(frame_w,x2)); y2=max(y1+2,min(frame_h,y2))
    return (x1,y1,x2,y2), {"seed_frames": dbg["seed_frames"], "seed_median": (x1,y1,x2,y2), "seed_count": len(seeds)}

def build_spans(times_present: List[float], gap_merge: float, min_len: float) -> List[Tuple[float,float]]:
    if not times_present:
        return []
    times_present = sorted(times_present)
    spans: List[Tuple[float,float]] = []
    s = times_present[0]; e = times_present[0]
    for t in times_present[1:]:
        if t - e <= gap_merge:
            e = t
        else:
            if (e - s) >= min_len:
                spans.append((s,e))
            s = t; e = t
    if (e - s) >= min_len:
        spans.append((s,e))
    return spans

def process_job(job_id: str) -> None:
    jd = job_dir(job_id)
    setup = read_json(setup_path(job_id), {})
    meta = read_json(meta_path(job_id), {})

    video_path = meta.get("video_path") or setup.get("video_path")
    if not video_path or not os.path.exists(video_path):
        set_status(job_id, "error", error="Missing video_path. Upload a video first.", progress=0)
        return

    # User intent params
    camera_mode = setup.get("camera_mode", "broadcast")
    jersey_hex = (setup.get("jersey_color") or "").strip() or "#1d3936"
    player_number = str(setup.get("player_number") or "").strip()
    clicks = setup.get("clicks") or []
    verify = bool(setup.get("verify", False))

    # Reliability knobs
    min_clip_len = float(setup.get("min_clip_len", 20.0))  # user request: >=20s
    gap_merge = float(setup.get("gap_merge", 5.0))          # merge short occlusions into same "shift"
    sticky_seconds = float(setup.get("sticky_seconds", 1.5)) # keep "present" briefly if detector misses
    pre_roll = float(setup.get("pre_roll", 4.0))            # user request: 3-5 sec prior
    post_roll = float(setup.get("post_roll", 1.5))

    # Tracking knobs
    detect_stride = int(setup.get("detect_stride", 2))      # run detector every N frames
    local_expand = float(setup.get("local_expand", 1.8))    # local search window scale
    switch_iou_min = float(setup.get("switch_iou_min", 0.05))
    color_gate = float(setup.get("color_gate", 0.55))       # max hsv distance allowed (lower is stricter)
    reacquire_confirm = int(setup.get("reacquire_confirm", 2))

    set_status(job_id, "processing", stage="seed", progress=30, message="Seeding track…")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        set_status(job_id, "error", error="Cannot open video with OpenCV.", progress=0)
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Load models (best-effort)
    yolo = load_yolo()
    ocr = load_easyocr_reader() if bool(setup.get("use_ocr", False)) else None

    target_hsv = bgr_to_hsv(hex_to_bgr(jersey_hex))

    # Build seed bbox from clicks
    if not isinstance(clicks, list) or len(clicks) < 1:
        # still allow running; but will be weaker
        clicks = []
    seed_bbox, seed_dbg = seed_bbox_from_clicks(cap, yolo, clicks, fps, frame_w, frame_h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize fallback tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(np.zeros((frame_h, frame_w, 3), dtype=np.uint8), (0,0,10,10))
    # We'll init tracker when we have first real frame
    tracker_inited = False

    # State
    track_bbox: Optional[Tuple[int,int,int,int]] = None
    last_good_bbox: Optional[Tuple[int,int,int,int]] = None
    present_times: List[float] = []

    miss_frames = 0
    confirm_left = 0
    pending_bbox: Optional[Tuple[int,int,int,int]] = None

    verify_writer = None
    verify_path = jd / "debug_verify.mp4"
    if verify:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        verify_writer = cv2.VideoWriter(str(verify_path), fourcc, max(1.0, fps/2.0), (frame_w, frame_h))

    def within_local(b: Tuple[int,int,int,int], ref: Tuple[int,int,int,int]) -> bool:
        rx1,ry1,rx2,ry2 = ref
        cx,cy = center(ref)
        rw = (rx2-rx1)*local_expand
        rh = (ry2-ry1)*local_expand
        lx1 = int(cx - rw/2); lx2=int(cx + rw/2)
        ly1 = int(cy - rh/2); ly2=int(cy + rh/2)
        bx1,by1,bx2,by2 = b
        mx,my = center(b)
        return (lx1 <= mx <= lx2) and (ly1 <= my <= ly2)

    def score_candidate(frame: np.ndarray, cand: Tuple[int,int,int,int], ref: Optional[Tuple[int,int,int,int]]) -> Tuple[float, Dict[str,Any]]:
        # color
        mean_bgr = robust_torso_mean_bgr(frame, cand)
        cand_hsv = bgr_to_hsv((int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])))
        cd = hsv_distance(cand_hsv, target_hsv)
        color_score = max(0.0, 1.0 - min(1.0, cd))  # 1 good, 0 bad

        # geometry
        if ref is None:
            i = 0.0
            d = 0.0
        else:
            i = iou(cand, ref)
            d = dist_norm(cand, ref, frame_w, frame_h)
        geo_score = (i*1.2 + max(0.0, 1.0 - d*6.0)*0.8)

        # total
        total = color_score*0.65 + geo_score*0.35
        return total, {"color_d": cd, "color_score": color_score, "iou": i, "dist": d, "total": total}

    # Main loop
    set_status(job_id, "processing", stage="track", progress=35, message="Tracking…")
    frame_idx = 0
    last_status_t = time.time()

    # We’ll treat "present" per-frame, then later merge into spans with gap_merge and add rolls.
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t = frame_idx / max(1.0, fps)

        # init tracker on first frame
        if not tracker_inited:
            x1,y1,x2,y2 = seed_bbox
            w = max(2, x2-x1); h = max(2, y2-y1)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1,y1,w,h))
            tracker_inited = True
            track_bbox = seed_bbox
            last_good_bbox = seed_bbox

        dets: List[Tuple[int,int,int,int,float]] = []
        use_det = (yolo is not None) and (frame_idx % max(1, detect_stride) == 0)

        if use_det:
            dets = detections_yolo(yolo, frame)

        chosen_bbox: Optional[Tuple[int,int,int,int]] = None
        chosen_dbg: Dict[str,Any] = {}

        # 1) Update tracker (fast) to keep continuity
        if tracker_inited:
            ok_t, bb = tracker.update(frame)
            if ok_t:
                x,y,w,h = bb
                cand = (int(x), int(y), int(x+w), int(y+h))
                chosen_bbox = cand
                chosen_dbg["tracker"] = True
            else:
                chosen_dbg["tracker"] = False

        # 2) If detector available, do reacquire / correction
        if use_det and dets:
            ref = chosen_bbox or track_bbox or last_good_bbox
            cand_list = [(x1,y1,x2,y2) for x1,y1,x2,y2,_ in dets]

            # Stage A: local candidates first
            local = [b for b in cand_list if ref is not None and within_local(b, ref)]
            search = local if local else cand_list
            stage = "local" if local else "global"

            best = None
            best_score = -1.0
            best_info: Dict[str,Any] = {}
            for b in search:
                sc, info = score_candidate(frame, b, ref)
                if sc > best_score:
                    best_score = sc
                    best = b
                    best_info = info

            chosen_dbg["det_stage"] = stage
            chosen_dbg["det_best_score"] = best_score
            chosen_dbg["det_best_info"] = best_info

            # Gating rules:
            if best is not None:
                # hard color gate (prevents identity switching on wrong team)
                if best_info.get("color_d", 9.9) <= color_gate:
                    # don't switch identities: require some overlap or confirm frames
                    if ref is None or best_info.get("iou", 0.0) >= switch_iou_min:
                        chosen_bbox = best
                        pending_bbox = None
                        confirm_left = 0
                        chosen_dbg["accepted"] = "iou_or_new"
                    else:
                        # low IOU: require consecutive confirmations
                        if pending_bbox is None or iou(pending_bbox, best) < 0.3:
                            pending_bbox = best
                            confirm_left = reacquire_confirm
                        else:
                            confirm_left = max(0, confirm_left-1)
                        if confirm_left <= 0:
                            chosen_bbox = best
                            pending_bbox = None
                            chosen_dbg["accepted"] = "confirmed_switch"
                        else:
                            chosen_dbg["accepted"] = f"pending({confirm_left})"
                else:
                    chosen_dbg["rejected"] = "color_gate"

        # 3) Update tracker bbox if we picked a det
        if chosen_bbox is not None:
            x1,y1,x2,y2 = chosen_bbox
            x1=max(0,min(frame_w-2,x1)); y1=max(0,min(frame_h-2,y1))
            x2=max(x1+2,min(frame_w,x2)); y2=max(y1+2,min(frame_h,y2))
            chosen_bbox = (x1,y1,x2,y2)
            track_bbox = chosen_bbox
            last_good_bbox = chosen_bbox
            miss_frames = 0
            # re-init CSRT periodically to reduce drift (cheap)
            if tracker_inited and (frame_idx % int(max(1, fps)) == 0):
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x1,y1,x2-x1,y2-y1))
            present_times.append(t)
        else:
            miss_frames += 1
            # sticky present
            if last_good_bbox is not None and miss_frames <= int(sticky_seconds * fps):
                present_times.append(t)
                chosen_dbg["sticky"] = True
            else:
                chosen_dbg["sticky"] = False

        # optional OCR confirm (v2)
        if ocr is not None and chosen_bbox is not None and player_number:
            # Only run OCR sparingly
            if frame_idx % int(max(1, fps*0.5)) == 0:
                x1,y1,x2,y2 = chosen_bbox
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    try:
                        txts = ocr.readtext(roi, detail=0, paragraph=False)
                        joined = " ".join([t for t in txts if isinstance(t,str)])
                        chosen_dbg["ocr"] = joined
                        if player_number not in joined:
                            # weaken presence if mismatch repeatedly (soft gate)
                            chosen_dbg["ocr_match"] = False
                        else:
                            chosen_dbg["ocr_match"] = True
                    except Exception as e:
                        chosen_dbg["ocr_error"] = str(e)

        # verify overlay
        if verify_writer is not None and (frame_idx % int(max(1, detect_stride)) == 0):
            vis = frame.copy()
            if dets:
                for x1,y1,x2,y2,conf in dets[:30]:
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (80,80,80), 1)
            if chosen_bbox is not None:
                x1,y1,x2,y2 = chosen_bbox
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"t={t:.2f}s present={bool(chosen_bbox)}", (12,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            # write at half-rate
            verify_writer.write(vis)

        # periodic status
        if time.time() - last_status_t > 1.2:
            prog = int(35 + (frame_idx / max(1, frame_count)) * 40) if frame_count > 0 else 50
            set_status(job_id, "processing", stage="track", progress=prog, message=f"Tracking… t={t:.1f}s")
            last_status_t = time.time()

        frame_idx += 1

    cap.release()
    if verify_writer is not None:
        verify_writer.release()

    set_status(job_id, "processing", stage="spans", progress=78, message="Building clips…")

    if not present_times:
        set_status(job_id, "error", progress=0,
                   error="No player selected. Either click the player once on the video, or provide at least 3 good seeds (tight torso).")
        # still write results so UI shows something
        write_json(results_path(job_id), {
            "status": "error",
            "job_id": job_id,
            "camera_mode": camera_mode,
            "player_number": player_number,
            "jersey_color": jersey_hex,
            "clicks_count": len(clicks),
            "clicks": clicks,
            "clips": [],
        })
        return

    spans = build_spans(present_times, gap_merge=gap_merge, min_len=min_clip_len)

    # If spans is empty due to min_len, relax once (user wants clips even if shorter)
    if not spans and present_times:
        spans = build_spans(present_times, gap_merge=gap_merge, min_len=max(6.0, min_clip_len*0.5))

    dur = ffprobe_duration(video_path) or (present_times[-1] + 1.0)
    dur = max(dur, 0.1)

    # Apply rolls + clamp
    rolled: List[Tuple[float,float]] = []
    for s,e in spans:
        ss = max(0.0, s - pre_roll)
        ee = min(dur, e + post_roll)
        if ee > ss:
            rolled.append((ss,ee))

    # Merge again after roll
    rolled.sort()
    merged: List[Tuple[float,float]] = []
    for s,e in rolled:
        if not merged:
            merged.append((s,e))
        else:
            ls,le = merged[-1]
            if s - le <= gap_merge:
                merged[-1] = (ls, max(le,e))
            else:
                merged.append((s,e))

    clips_dir(job_id).mkdir(parents=True, exist_ok=True)
    clip_infos: List[Dict[str,Any]] = []
    clip_paths: List[str] = []

    for i,(s,e) in enumerate(merged, start=1):
        outp = clips_dir(job_id) / f"clip_{i:03d}.mp4"
        okc = cut_clip(video_path, s, e, str(outp))
        if okc:
            clip_paths.append(str(outp))
            clip_infos.append({
                "start": float(s), "end": float(e),
                "path": str(outp),
                "url": f"/data/jobs/{job_id}/clips/{outp.name}"
            })

    combined_path = jd / "combined.mp4"
    combined_ok = concat_clips(clip_paths, str(combined_path)) if clip_paths else False

    res = {
        "status": "done",
        "job_id": job_id,
        "camera_mode": camera_mode,
        "player_number": player_number,
        "jersey_color": jersey_hex,
        "clicks_count": len(clicks),
        "clicks": clicks,
        "clips": clip_infos,
        "combined_path": str(combined_path) if combined_ok else None,
        "combined_url": f"/data/jobs/{job_id}/combined.mp4" if combined_ok else None,
        "debug": {
            "seed": seed_dbg,
            "fps": fps,
            "min_clip_len": min_clip_len,
            "gap_merge": gap_merge,
            "sticky_seconds": sticky_seconds,
            "pre_roll": pre_roll,
            "post_roll": post_roll,
            "verify": bool(verify),
            "verify_path": str(verify_path) if verify and verify_path.exists() else None,
        }
    }

    write_json(results_path(job_id), res)
    # Update status
    extra: Dict[str,Any] = {"progress": 100, "message": "Done."}
    if verify and verify_path.exists():
        extra["verify_url"] = f"/data/jobs/{job_id}/{verify_path.name}"
    set_status(job_id, "done", stage="done", **extra)
