import os
import json
import time
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np


# ----------------------------
# Paths / helpers
# ----------------------------

def _job_root() -> Path:
    # worker/tasks.py is Projects/worker/tasks.py → parents[1] is Projects/
    return Path(__file__).resolve().parents[1]

BASE_DIR = Path(os.getenv("APP_ROOT", _job_root())).resolve()
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def meta_path(job_id: str) -> Path:
    return job_dir(job_id) / "meta.json"


def setup_path(job_id: str) -> Path:
    return job_dir(job_id) / "setup.json"


def results_path(job_id: str) -> Path:
    return job_dir(job_id) / "results.json"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def set_status(job_id: str, status: str, **fields: Any) -> None:
    meta = read_json(meta_path(job_id), {})
    meta["job_id"] = job_id
    meta["status"] = status
    meta["updated_at"] = time.time()
    meta.update(fields)
    write_json(meta_path(job_id), meta)


def ffprobe_fps(path: str) -> float:
    # Try to get FPS reliably
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path
    ]
    try:
        out = subprocess.check_output(cmd).decode().strip()
        if "/" in out:
            a, b = out.split("/")
            a = float(a); b = float(b)
            if b != 0:
                return a / b
        return float(out)
    except Exception:
        return 30.0


def ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path
    ]
    try:
        out = subprocess.check_output(cmd).decode().strip()
        return float(out)
    except Exception:
        return 0.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def bbox_center(b: Tuple[int,int,int,int]) -> Tuple[float,float]:
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)


def safe_crop(img: np.ndarray, b: Tuple[int,int,int,int]) -> np.ndarray:
    h,w = img.shape[:2]
    x1,y1,x2,y2 = b
    x1 = int(clamp(x1, 0, w-1))
    x2 = int(clamp(x2, 0, w-1))
    y1 = int(clamp(y1, 0, h-1))
    y2 = int(clamp(y2, 0, h-1))
    if x2 <= x1 or y2 <= y1:
        return img[0:1,0:1].copy()
    return img[y1:y2, x1:x2].copy()


def jersey_hist(img_bgr: np.ndarray) -> np.ndarray:
    """
    Histogram in HSV for color matching.
    Uses center crop to reduce ice/boards influence.
    """
    h,w = img_bgr.shape[:2]
    cx1 = int(w*0.25); cx2 = int(w*0.75)
    cy1 = int(h*0.25); cy2 = int(h*0.75)
    patch = img_bgr[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [30,32], [0,180, 0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def hist_sim(h1: np.ndarray, h2: np.ndarray) -> float:
    if h1 is None or h2 is None:
        return 0.0
    # correlation in [ -1..1 ], normalize to [0..1]
    v = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)
    return float((v + 1.0) / 2.0)


def hex_to_bgr(hex_color: str) -> Tuple[int,int,int]:
    s = (hex_color or "").strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return (0,0,0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b,g,r)


def target_color_hist(hex_color: str) -> Optional[np.ndarray]:
    """
    Create a synthetic hist for the chosen jersey color (rough).
    Used as a weak prior when seed hist is noisy.
    """
    bgr = np.full((64,64,3), hex_to_bgr(hex_color), dtype=np.uint8)
    return jersey_hist(bgr)


# ----------------------------
# Detector (YOLO) with fallback
# ----------------------------

class PersonDetector:
    def __init__(self):
        self.model = None
        self.ok = False
        try:
            from ultralytics import YOLO
            # Use lightweight default; may auto-download weights
            self.model = YOLO("yolov8n.pt")
            self.ok = True
        except Exception:
            self.model = None
            self.ok = False

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        """
        Returns list of (x1,y1,x2,y2,conf) for persons.
        """
        if not self.ok or self.model is None:
            return []
        try:
            res = self.model.predict(frame_bgr, verbose=False, imgsz=640, conf=0.25)[0]
            boxes = []
            if res.boxes is None:
                return boxes
            for b in res.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                # class 0 is "person" in COCO
                if cls != 0:
                    continue
                x1,y1,x2,y2 = b.xyxy[0].tolist()
                boxes.append((int(x1),int(y1),int(x2),int(y2),conf))
            return boxes
        except Exception:
            return []


# ----------------------------
# Tracking logic
# ----------------------------

def pick_seed_bbox(
    cap: cv2.VideoCapture,
    detector: PersonDetector,
    clicks: List[Dict[str, float]],
    fps: float,
) -> Tuple[Optional[Tuple[int,int,int,int]], Optional[np.ndarray], Dict[str, Any]]:
    """
    Use click times (t, x, y normalized) to pick a bbox that best matches.
    Returns: (bbox, hist, debug)
    """
    debug = {"seed_pick": {"frames": []}}

    if not clicks:
        return None, None, debug

    best = None
    best_hist = None
    best_score = -1.0

    for c in clicks[:8]:
        t = float(c.get("t", 0.0))
        xn = float(c.get("x", 0.5))
        yn = float(c.get("y", 0.5))
        frame_idx = int(max(0, round(t * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h,w = frame.shape[:2]
        px = xn * w
        py = yn * h

        dets = detector.detect(frame)
        if not dets:
            continue

        # score by distance to click + conf
        for (x1,y1,x2,y2,conf) in dets:
            cx,cy = bbox_center((x1,y1,x2,y2))
            dist = math.hypot(cx - px, cy - py)
            dist_norm = dist / math.hypot(w, h)
            score = (conf * 0.7) + ((1.0 - dist_norm) * 0.3)

            debug["seed_pick"]["frames"].append({
                "t": t, "frame": frame_idx, "bbox": [x1,y1,x2,y2], "conf": conf,
                "dist_norm": dist_norm, "score": score
            })

            if score > best_score:
                best_score = score
                best = (x1,y1,x2,y2)

                crop = safe_crop(frame, best)
                best_hist = jersey_hist(crop)

    debug["seed_pick"]["best_score"] = best_score
    debug["seed_pick"]["best_bbox"] = list(best) if best else None
    return best, best_hist, debug


def create_tracker():
    # CSRT is best balance for broadcast + occlusion (slower but stable)
    try:
        return cv2.TrackerCSRT_create()
    except Exception:
        # fallback
        try:
            return cv2.TrackerKCF_create()
        except Exception:
            return None


def track_video(
    video_path: str,
    detector: PersonDetector,
    seed_bbox: Tuple[int,int,int,int],
    seed_hist: Optional[np.ndarray],
    jersey_prior_hist: Optional[np.ndarray],
    extend_sec: float,
    gap_sec: float,
    verify_mode: bool,
    jersey_color_hex: str,
) -> Dict[str, Any]:
    """
    Tracks the target; returns dict with:
      - spans: [[start,end], ...]
      - track_debug
      - verify_video_path (optional)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"spans": [], "track_debug": {"error": "Failed to open video"}}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = ffprobe_fps(video_path)
    dur = ffprobe_duration(video_path)

    # init tracker at first frame near seed
    tracker = create_tracker()
    if tracker is None:
        return {"spans": [], "track_debug": {"error": "OpenCV tracker not available"}}

    # Move to the seed frame for initialization
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Find a frame where bbox is valid: use the first click time (assume near there)
    init_frame = None
    init_idx = 0
    # attempt to init at time of first click
    init_t = 0.0
    init_t = float(max(0.0, min(dur, 0.0)))
    # If we have a bbox from click pick, just init at current pos (best effort)
    # We'll init at frame 0 but set bbox — tracker will lock on first read.
    ok, init_frame = cap.read()
    if not ok or init_frame is None:
        return {"spans": [], "track_debug": {"error": "Cannot read first frame"}}

    x1,y1,x2,y2 = seed_bbox
    w = max(2, x2-x1)
    h = max(2, y2-y1)
    bbox_cv = (float(x1), float(y1), float(w), float(h))

    tracker.init(init_frame, bbox_cv)

    # verify output
    out_verify_path = None
    writer = None
    if verify_mode:
        out_verify_path = str(Path(video_path).with_name("verify.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vh, vw = init_frame.shape[:2]
        writer = cv2.VideoWriter(out_verify_path, fourcc, fps, (vw, vh))

    # tracking params
    DET_EVERY_N = int(max(5, round(fps * 0.33)))  # run YOLO ~3x/sec
    REACQUIRE_RADIUS = 0.25  # as fraction of diagonal
    LOST_MAX = int(round(gap_sec * fps))          # gap tolerance
    PRESENT_MIN_CONF = 0.35

    last_good_bbox = seed_bbox
    last_good_hist = seed_hist
    last_seen_frame = 0
    present_flags = []

    # We'll rewind and process entire video for span detection
    cap.release()
    cap = cv2.VideoCapture(video_path)

    frame_idx = -1
    last_det_boxes = []
    diag = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1
        H,W = frame.shape[:2]
        if diag is None:
            diag = math.hypot(W, H)

        # tracker update
        ok_t, bb = tracker.update(frame)
        track_conf = 0.0
        cur_bbox = None

        if ok_t:
            x, y, w2, h2 = bb
            x1t = int(x); y1t = int(y); x2t = int(x+w2); y2t = int(y+h2)
            # sanity clamp
            x1t = int(clamp(x1t, 0, W-1)); x2t = int(clamp(x2t, 0, W-1))
            y1t = int(clamp(y1t, 0, H-1)); y2t = int(clamp(y2t, 0, H-1))
            if x2t > x1t and y2t > y1t:
                cur_bbox = (x1t,y1t,x2t,y2t)
                # color similarity
                crop = safe_crop(frame, cur_bbox)
                hcur = jersey_hist(crop)
                sim_seed = hist_sim(last_good_hist, hcur) if last_good_hist is not None else 0.0
                sim_prior = hist_sim(jersey_prior_hist, hcur) if jersey_prior_hist is not None else 0.0
                # combine
                track_conf = max(sim_seed, sim_prior) * 0.8 + 0.2  # keep non-zero
        else:
            cur_bbox = None
            track_conf = 0.0

        # periodic detection/reacquire
        if detector.ok and (frame_idx % DET_EVERY_N == 0 or cur_bbox is None or track_conf < PRESENT_MIN_CONF):
            dets = detector.detect(frame)
            last_det_boxes = dets

            # choose best detection based on proximity + color sim
            if dets:
                # reference center is last_good_bbox if we have it
                ref = last_good_bbox if last_good_bbox else seed_bbox
                rcx, rcy = bbox_center(ref)
                best = None
                best_score = -1.0

                for (x1d,y1d,x2d,y2d,conf) in dets:
                    cand = (x1d,y1d,x2d,y2d)
                    ccx, ccy = bbox_center(cand)
                    dist = math.hypot(ccx-rcx, ccy-rcy) / (diag if diag else 1.0)
                    dist_score = 1.0 - clamp(dist / REACQUIRE_RADIUS, 0.0, 1.0)

                    crop = safe_crop(frame, cand)
                    hcand = jersey_hist(crop)
                    sim_seed = hist_sim(last_good_hist, hcand) if last_good_hist is not None else 0.0
                    sim_prior = hist_sim(jersey_prior_hist, hcand) if jersey_prior_hist is not None else 0.0
                    sim = max(sim_seed, sim_prior)

                    # overall score
                    score = (conf * 0.35) + (dist_score * 0.25) + (sim * 0.40)
                    if score > best_score:
                        best_score = score
                        best = cand
                        best_hist2 = hcand

                # if reacquire is good enough, reset tracker
                if best is not None and best_score >= 0.35:
                    last_good_bbox = best
                    last_good_hist = best_hist2
                    x1r,y1r,x2r,y2r = best
                    w = max(2, x2r-x1r)
                    h = max(2, y2r-y1r)
                    tracker = create_tracker()
                    if tracker is not None:
                        tracker.init(frame, (float(x1r), float(y1r), float(w), float(h)))
                    cur_bbox = best
                    track_conf = max(track_conf, 0.55)

        # Determine "present" by confidence OR if within tolerated lost window
        present = (cur_bbox is not None and track_conf >= PRESENT_MIN_CONF)
        if present:
            last_seen_frame = frame_idx
            last_good_bbox = cur_bbox

        else:
            # still consider present if within gap tolerance (broadcast cut)
            if (frame_idx - last_seen_frame) <= LOST_MAX:
                present = True  # treat as continuous presence for span merging

        present_flags.append(present)

        # verify overlay output
        if writer is not None:
            # draw bbox
            if cur_bbox is not None:
                x1v,y1v,x2v,y2v = cur_bbox
                cv2.rectangle(frame, (x1v,y1v), (x2v,y2v), (0,255,0), 2)
                cv2.putText(frame, f"TRACK conf={track_conf:.2f}", (x1v, max(0,y1v-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # jersey color swatch
            bgr = hex_to_bgr(jersey_color_hex)
            cv2.rectangle(frame, (10,10), (70,40), bgr, -1)
            cv2.putText(frame, "JERSEY", (80,33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    # Convert present_flags into spans
    spans = []
    in_span = False
    start_idx = 0

    for i, p in enumerate(present_flags):
        if p and not in_span:
            in_span = True
            start_idx = i
        if not p and in_span:
            in_span = False
            end_idx = i - 1
            spans.append((start_idx / fps, end_idx / fps))

    if in_span:
        spans.append((start_idx / fps, (len(present_flags)-1) / fps))

    # Auto-extend + clamp + merge
    def extend_span(s,e):
        return (max(0.0, s - 0.0), min(dur if dur>0 else e+extend_sec, e + extend_sec))

    spans2 = [extend_span(s,e) for (s,e) in spans if e > s]
    spans2.sort(key=lambda x: x[0])

    merged = []
    for s,e in spans2:
        if not merged:
            merged.append([s,e])
        else:
            ps,pe = merged[-1]
            if s <= pe + gap_sec:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s,e])

    spans_final = [[float(s), float(e)] for s,e in merged if e - s >= 1.0]

    return {
        "spans": spans_final,
        "track_debug": {
            "fps": fps,
            "duration": dur,
            "detector_ok": detector.ok,
            "verify_mode": verify_mode,
            "present_frames": int(sum(1 for p in present_flags if p)),
            "total_frames": len(present_flags),
        },
        "verify_path": out_verify_path,
    }


# ----------------------------
# Click-span fallback (always works)
# ----------------------------

def spans_from_clicks(clicks: List[Dict[str,float]], extend_sec: float, gap_sec: float) -> List[List[float]]:
    times = sorted([float(c.get("t", 0.0)) for c in clicks if c is not None])
    if not times:
        return []
    # cluster into spans
    spans = []
    s = times[0]
    prev = times[0]
    for t in times[1:]:
        if (t - prev) <= gap_sec:
            prev = t
        else:
            spans.append([s, prev])
            s = t
            prev = t
    spans.append([s, prev])

    # extend
    spans = [[max(0.0, a), b + extend_sec] for a,b in spans]

    # merge any overlaps
    spans.sort(key=lambda x: x[0])
    merged = []
    for a,b in spans:
        if not merged:
            merged.append([a,b])
        else:
            pa,pb = merged[-1]
            if a <= pb:
                merged[-1][1] = max(pb,b)
            else:
                merged.append([a,b])
    # filter tiny
    return [x for x in merged if (x[1]-x[0]) >= 1.0]


# ----------------------------
# Clip cutting
# ----------------------------

def cut_clip(video_path: str, out_path: str, start: float, end: float) -> bool:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", video_path,
        "-c", "copy",
        out_path
    ]
    r = subprocess.run(cmd, check=False)
    return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024


def concat_clips(clip_paths: List[str], out_path: str) -> bool:
    if not clip_paths:
        return False
    list_file = os.path.join(os.path.dirname(out_path), "concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        out_path
    ]
    r = subprocess.run(cmd, check=False)
    return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024


# ----------------------------
# Main worker entry
# ----------------------------

def process_job(job_id: str) -> None:
    jd = job_dir(job_id)
    meta = read_json(meta_path(job_id), {})
    setup = read_json(setup_path(job_id), {})

    video_path = meta.get("video_path")
    if not video_path or not os.path.exists(video_path):
        set_status(job_id, "error", progress=0, message="Missing input video.", error="video_path not found")
        return

    clicks = setup.get("clicks", []) or []
    extend_sec = float(setup.get("extend_sec", 20) or 20)
    gap_sec = float(setup.get("gap_sec", 20) or 20)  # allow same shift across cuts
    verify_mode = bool(setup.get("verify_mode", False))
    jersey_color = setup.get("jersey_color", "#1d3936") or "#1d3936"
    player_number = (setup.get("player_number") or "").strip()
    camera_mode = setup.get("camera_mode", "broadcast")

    set_status(job_id, "processing", progress=35, message="Starting tracking…", stage="tracking")

    # Detector (YOLO) – can fail if weights cannot download; we fall back safely.
    detector = PersonDetector()

    # Pick seed bbox from clicks
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = ffprobe_fps(video_path)
    seed_bbox, seed_hist, seed_dbg = pick_seed_bbox(cap, detector, clicks, fps)
    cap.release()

    jersey_prior = target_color_hist(jersey_color)

    spans = []
    track_debug = {"seed": seed_dbg}

    if seed_bbox is not None:
        track_res = track_video(
            video_path=video_path,
            detector=detector,
            seed_bbox=seed_bbox,
            seed_hist=seed_hist,
            jersey_prior_hist=jersey_prior,
            extend_sec=extend_sec,
            gap_sec=gap_sec,
            verify_mode=verify_mode,
            jersey_color_hex=jersey_color,
        )
        spans = track_res.get("spans", []) or []
        track_debug["track"] = track_res.get("track_debug", {})
        verify_path = track_res.get("verify_path")
    else:
        verify_path = None
        track_debug["track"] = {"error": "No seed bbox found; falling back to click spans."}

    # fallback if tracker produced nothing
    if not spans:
        spans = spans_from_clicks(clicks, extend_sec=extend_sec, gap_sec=gap_sec)
        track_debug["fallback"] = "click_spans"

    if not spans:
        set_status(job_id, "error", progress=0, message="No spans found. Add 3–8 torso clicks.", error="no_spans")
        write_json(results_path(job_id), {
            "status": "error",
            "job_id": job_id,
            "error": "No spans found. Add 3–8 torso clicks.",
            "setup": setup,
            "debug": track_debug,
            "clips": [],
        })
        return

    set_status(job_id, "processing", progress=60, message="Cutting clips…", stage="clipping")

    clips_dir = jd / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    clip_paths = []
    for i, (s,e) in enumerate(spans, start=1):
        out = str(clips_dir / f"clip_{i:03d}.mp4")
        ok = cut_clip(video_path, out, s, e)
        if ok:
            clip_paths.append(out)
            clips.append({
                "start": float(s),
                "end": float(e),
                "path": out,
                "url": f"/data/jobs/{job_id}/clips/clip_{i:03d}.mp4"
            })

    combined_path = str(jd / "combined.mp4")
    combined_ok = concat_clips(clip_paths, combined_path)

    set_status(job_id, "done", progress=100, message="Done.", stage="done")

    out_obj = {
        "status": "done",
        "job_id": job_id,
        "camera_mode": camera_mode,
        "player_number": player_number,
        "jersey_color": jersey_color,
        "clicks_count": len(clicks),
        "clicks": clicks,
        "spans": spans,
        "clips": clips,
        "combined_path": combined_path if combined_ok else None,
        "combined_url": f"/data/jobs/{job_id}/combined.mp4" if combined_ok else None,
        "verify_path": verify_path,
        "verify_url": f"/data/jobs/{job_id}/verify.mp4" if verify_mode and verify_path else None,
        "debug": track_debug,
    }
    write_json(results_path(job_id), out_obj)

