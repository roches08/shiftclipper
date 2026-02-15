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


# ---------------------------
# Paths / storage helpers
# ---------------------------

ROOT = Path(__file__).resolve().parents[1]  # Projects/
DATA = ROOT / "data"
JOBS = DATA / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS = ROOT / "yolov8s.pt"


def _now() -> float:
    return time.time()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    jd = JOBS / job_id
    _ensure_dir(jd)
    _ensure_dir(jd / "clips")
    return jd


def read_json(p: Path, default: Any = None) -> Any:
    if not p.exists():
        return default
    return json.loads(p.read_text())


def write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2))


def status_path(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def results_path(job_id: str) -> Path:
    return job_dir(job_id) / "results.json"


def setup_path(job_id: str) -> Path:
    return job_dir(job_id) / "setup.json"


def input_path(job_id: str) -> Path:
    # upload endpoint stores "in.mp4"
    return job_dir(job_id) / "in.mp4"


def proxy_path(job_id: str) -> Path:
    return job_dir(job_id) / "input_proxy.mp4"


def clips_dir(job_id: str) -> Path:
    return job_dir(job_id) / "clips"


def combined_path(job_id: str) -> Path:
    return job_dir(job_id) / "combined.mp4"


def set_status(
    job_id: str,
    status: str,
    *,
    stage: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    setup: Optional[Dict[str, Any]] = None,
) -> None:
    st = read_json(status_path(job_id), {}) or {}
    st["job_id"] = job_id
    st["status"] = status
    st["stage"] = stage or st.get("stage") or status
    if progress is not None:
        st["progress"] = int(progress)
    if message is not None:
        st["message"] = message
    if setup is not None:
        st["setup"] = setup
    st["updated_at"] = _now()
    if extra:
        st.update(extra)
    write_json(status_path(job_id), st)


# ---------------------------
# ffmpeg helpers
# ---------------------------

def run(cmd: List[str]) -> None:
    subprocess.check_call(cmd)


def ffprobe_info(video_path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8")
    data = json.loads(out)
    stream = data["streams"][0]
    def _fps(fr: str) -> float:
        if not fr or "/" not in fr:
            return 0.0
        a, b = fr.split("/")
        b = float(b)
        return float(a) / b if b else 0.0

    fps = _fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    duration = float(stream.get("duration") or 0.0)
    w = int(stream.get("width") or 0)
    h = int(stream.get("height") or 0)
    nb_frames = stream.get("nb_frames")
    total_frames = int(nb_frames) if nb_frames and str(nb_frames).isdigit() else int(duration * fps) if fps and duration else 0
    return {"fps": fps, "duration": duration, "W": w, "H": h, "total_frames": total_frames}


def make_proxy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    # fast proxy for UI clicking
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        "scale=1280:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(dst),
    ]
    run(cmd)


def cut_clip(src: Path, dst: Path, start: float, end: float) -> None:
    # -ss before -i for speed, but accuracy is fine for now
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(src),
        "-c",
        "copy",
        str(dst),
    ]
    run(cmd)


def concat_clips(clip_paths: List[Path], out_path: Path) -> None:
    if not clip_paths:
        return
    lst = out_path.parent / "concat_list.txt"
    lst.write_text("\n".join([f"file '{p.as_posix()}'" for p in clip_paths]) + "\n")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(lst),
        "-c",
        "copy",
        str(out_path),
    ]
    run(cmd)


# ---------------------------
# Vision helpers (YOLO + color)
# ---------------------------

_YOLO_MODEL: Optional[YOLO] = None


def get_yolo() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        # Ensure weights exist (your runpod_start downloads them once)
        if not YOLO_WEIGHTS.exists():
            # fallback: let ultralytics auto-download if missing
            _YOLO_MODEL = YOLO("yolov8s.pt")
        else:
            _YOLO_MODEL = YOLO(str(YOLO_WEIGHTS))
    return _YOLO_MODEL


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    s = (hex_color or "").strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return (0, 0, 0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def color_score(bgr_patch: np.ndarray, jersey_bgr: Tuple[int, int, int]) -> float:
    """
    Very simple jersey-color affinity score. Returns >1.0 if close-ish.
    """
    if bgr_patch.size == 0:
        return 0.0
    mean = bgr_patch.reshape(-1, 3).mean(axis=0)
    jb = np.array(jersey_bgr, dtype=np.float32)
    d = np.linalg.norm(mean - jb)
    # Convert distance to score (smaller distance -> bigger score)
    return float(255.0 / (d + 1.0))


def crop_safe(img: np.ndarray, xyxy: List[int]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]
    return img[y1:y2, x1:x2]


def yolo_person_boxes(img_bgr: np.ndarray, conf: float = 0.25) -> List[List[int]]:
    """
    Returns person boxes [x1,y1,x2,y2] in pixel coords using ultralytics YOLO.
    """
    model = get_yolo()
    # IMPORTANT: we pass a numpy image (supported) NOT a generator
    res = model.predict(img_bgr, conf=conf, verbose=False)
    if not res:
        return []
    r0 = res[0]
    boxes = []
    if r0.boxes is None:
        return []
    for b in r0.boxes:
        cls = int(b.cls.item()) if hasattr(b, "cls") else -1
        if cls != 0:  # 0 = person
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes


def norm_center(box: List[int], W: int, H: int) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / max(W, 1)
    cy = (y1 + y2) / 2.0 / max(H, 1)
    return (cx, cy)


def dist_norm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


# ---------------------------
# Seed selection from clicks
# ---------------------------

def pick_seed_bbox_from_clicks(
    video_path: Path,
    clicks: List[Dict[str, Any]],
    *,
    yolo_conf: float,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Given 2-3 clicks (t,x,y normalized), find the person bbox at each click,
    then fuse into a representative bbox (median center/size).
    """
    cap = cv2.VideoCapture(str(video_path))
    info = ffprobe_info(video_path)
    fps = info["fps"] or cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(info["W"] or cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(info["H"] or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    picked = []
    debug_frames = []

    for c in clicks:
        t = float(c.get("t", 0.0))
        nx = float(c.get("x", 0.5))
        ny = float(c.get("y", 0.5))
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        boxes = yolo_person_boxes(frame, conf=yolo_conf)
        px = int(nx * W)
        py = int(ny * H)

        best = None
        best_d = 1e9
        for b in boxes:
            x1, y1, x2, y2 = b
            if px < x1 or px > x2 or py < y1 or py > y2:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            d = (cx - px) ** 2 + (cy - py) ** 2
            if d < best_d:
                best_d = d
                best = b

        debug_frames.append(
            {"t": t, "frame": frame_idx, "dets": len(boxes), "picked": best}
        )
        if best is not None:
            picked.append(best)

    cap.release()

    if not picked:
        # fallback: dummy bbox centered-ish
        seed = [int(W * 0.45), int(H * 0.35), int(W * 0.55), int(H * 0.70)]
        dbg = {"seed_bbox": seed, "seed_frames": debug_frames, "seed_count": 0}
        return seed, dbg

    # Fuse picked bboxes via medians
    xs1 = sorted([b[0] for b in picked])
    ys1 = sorted([b[1] for b in picked])
    xs2 = sorted([b[2] for b in picked])
    ys2 = sorted([b[3] for b in picked])
    mid = len(picked) // 2
    seed = [xs1[mid], ys1[mid], xs2[mid], ys2[mid]]

    dbg = {"seed_bbox": seed, "seed_frames": debug_frames, "seed_count": len(picked)}
    return seed, dbg


# ---------------------------
# Tracking / presence spans
# ---------------------------

def track_presence_spans(
    video_path: Path,
    seed_bbox: List[int],
    *,
    jersey_hex: str,
    yolo_conf: float = 0.25,
    detect_stride: int = 3,
    dist_gate_norm: float = 0.18,
    dist_gate2_norm: float = 0.35,
    color_threshold: float = 1.05,
    sticky_seconds: float = 1.5,
    min_clip_len: float = 20.0,
    gap_merge: float = 2.0,
    pre_roll: float = 4.0,
    post_roll: float = 1.5,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    Detect a target player by:
    - YOLO person detection on stride frames
    - nearest-to-last-center gating
    - jersey-color scoring to break ties / filter
    Produces time spans where player is "present".
    """
    info = ffprobe_info(video_path)
    fps = info["fps"] or 30.0
    W, H = info["W"], info["H"]
    duration = info["duration"]

    jersey_bgr = hex_to_bgr(jersey_hex)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or info["total_frames"] or (duration * fps))

    last_center = norm_center(seed_bbox, W, H)
    present_until_t = -1e9

    sampled = 0
    present_flags = []  # (t, present_bool)

    debug_samples = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t = frame_idx / max(fps, 1e-6)

        if frame_idx % max(1, detect_stride) != 0:
            frame_idx += 1
            continue

        sampled += 1
        boxes = yolo_person_boxes(frame, conf=yolo_conf)

        chosen = None
        chosen_score = None
        chosen_color = None
        chosen_dist = None

        # rank candidates by (distance gate then color)
        candidates = []
        for b in boxes:
            c = norm_center(b, W, H)
            d = dist_norm(c, last_center)
            if d > dist_gate2_norm:
                continue

            patch = crop_safe(frame, b)
            cs = color_score(patch, jersey_bgr)

            candidates.append((d, cs, b))

        candidates.sort(key=lambda x: (x[0], -x[1]))

        if candidates:
            d, cs, b = candidates[0]
            if d <= dist_gate_norm or cs >= color_threshold:
                chosen = b
                chosen_score = float(cs)
                chosen_dist = float(d)
                chosen_color = float(cs)

        present = False
        if chosen is not None:
            last_center = norm_center(chosen, W, H)
            present_until_t = t + sticky_seconds
            present = True
        else:
            present = t <= present_until_t

        present_flags.append((t, present))

        if len(debug_samples) < 200:
            debug_samples.append(
                {
                    "t": t,
                    "present": present,
                    "chosen": chosen,
                    "score": chosen_score,
                    "color": chosen_color,
                    "dist": chosen_dist,
                    "boxes": len(boxes),
                }
            )

        frame_idx += 1

    cap.release()

    # build spans from present_flags
    spans: List[Tuple[float, float]] = []
    in_span = False
    s0 = 0.0

    for t, p in present_flags:
        if p and not in_span:
            in_span = True
            s0 = t
        if (not p) and in_span:
            in_span = False
            spans.append((s0, t))

    if in_span:
        spans.append((s0, duration))

    # apply pre/post roll, merge gaps, min length
    rolled = []
    for a, b in spans:
        a2 = max(0.0, a - pre_roll)
        b2 = min(duration, b + post_roll)
        rolled.append((a2, b2))

    rolled.sort()
    merged = []
    for a, b in rolled:
        if not merged:
            merged.append([a, b])
            continue
        pa, pb = merged[-1]
        if a <= pb + gap_merge:
            merged[-1][1] = max(pb, b)
        else:
            merged.append([a, b])

    final_spans = []
    for a, b in merged:
        if (b - a) >= min_clip_len:
            final_spans.append((float(a), float(b)))

    dbg = {
        "fps": fps,
        "W": W,
        "H": H,
        "total_frames": total_frames,
        "duration_s": duration,
        "min_clip_len": min_clip_len,
        "gap_merge": gap_merge,
        "sticky_seconds": sticky_seconds,
        "pre_roll": pre_roll,
        "post_roll": post_roll,
        "detect_stride": detect_stride,
        "yolo_conf": yolo_conf,
        "color_threshold": color_threshold,
        "dist_gate_norm": dist_gate_norm,
        "dist_gate2_norm": dist_gate2_norm,
        "sampled_detect_frames": sampled,
        "debug_samples": debug_samples,
        "spans": final_spans,
    }
    return final_spans, dbg


# ---------------------------
# Main worker entrypoint
# ---------------------------

def process_job(job_id: str) -> Dict[str, Any]:
    """
    RQ worker entrypoint.
    """
    jd = job_dir(job_id)
    vid = input_path(job_id)
    if not vid.exists():
        set_status(job_id, "failed", stage="failed", progress=100, message="Missing input video.")
        return {"ok": False, "error": "missing video"}

    setup = read_json(setup_path(job_id), {}) or {}
    clicks = setup.get("clicks") or []
    jersey_hex = setup.get("jersey_color") or "#203524"
    extend_sec = float(setup.get("extend_sec") or 2.0)
    verify_mode = bool(setup.get("verify_mode") or False)

    # TUNABLES (keep sane defaults)
    yolo_conf = float(setup.get("yolo_conf") or 0.25)
    detect_stride = int(setup.get("detect_stride") or 3)
    color_threshold = float(setup.get("color_threshold") or 1.05)
    dist_gate_norm = float(setup.get("dist_gate_norm") or 0.18)
    dist_gate2_norm = float(setup.get("dist_gate2_norm") or 0.35)

    min_clip_len = float(setup.get("min_clip_len") or 8.0)  # better for shifts than 20s
    gap_merge = float(setup.get("gap_merge") or 1.0)
    sticky_seconds = float(setup.get("sticky_seconds") or 1.2)
    pre_roll = float(setup.get("pre_roll") or 2.0)
    post_roll = float(setup.get("post_roll") or 1.0)

    set_status(job_id, "processing", stage="processing", progress=35, message="Starting tracking...")

    try:
        # proxy for UI
        make_proxy(vid, proxy_path(job_id))

        # seed bbox from clicks (2-3 clicks is enough; 0 clicks uses fallback bbox)
        seed_bbox, seed_dbg = pick_seed_bbox_from_clicks(
            vid,
            clicks,
            yolo_conf=yolo_conf,
        )

        spans, track_dbg = track_presence_spans(
            vid,
            seed_bbox,
            jersey_hex=jersey_hex,
            yolo_conf=yolo_conf,
            detect_stride=detect_stride,
            dist_gate_norm=dist_gate_norm,
            dist_gate2_norm=dist_gate2_norm,
            color_threshold=color_threshold,
            sticky_seconds=sticky_seconds,
            min_clip_len=min_clip_len,
            gap_merge=gap_merge,
            pre_roll=pre_roll,
            post_roll=post_roll,
        )

        set_status(job_id, "processing", stage="clipping", progress=70, message="Cutting clips...")

        clip_paths: List[Path] = []
        clips_out = []
        for i, (a, b) in enumerate(spans, start=1):
            a2 = max(0.0, a - extend_sec)
            b2 = b + extend_sec
            outp = clips_dir(job_id) / f"clip_{i:03d}.mp4"
            cut_clip(vid, outp, a2, b2)
            clip_paths.append(outp)
            clips_out.append(
                {
                    "start": a2,
                    "end": b2,
                    "path": str(outp),
                    "url": f"/data/jobs/{job_id}/clips/{outp.name}",
                }
            )

        comb = combined_path(job_id)
        if clip_paths:
            concat_clips(clip_paths, comb)

        res = {
            "video_path": str(vid),
            "job_id": job_id,
            "status": "done",
            "progress": 100,
            "message": "Done.",
            "proxy_ready": True,
            "proxy_path": str(proxy_path(job_id)),
            "proxy_url": f"/data/jobs/{job_id}/{proxy_path(job_id).name}",
            "clips": clips_out,
            "combined_path": str(comb) if comb.exists() else None,
            "combined_url": f"/data/jobs/{job_id}/{comb.name}" if comb.exists() else None,
            "setup": setup,
            "stage": "done",
            "updated_at": _now(),
            "debug": {"seed": seed_dbg, **track_dbg},
        }
        write_json(results_path(job_id), res)
        set_status(job_id, "done", stage="done", progress=100, message="Done.", extra=res)
        return res

    except Exception as e:
        set_status(job_id, "failed", stage="failed", progress=100, message=f"Worker crashed: {e}")
        raise

