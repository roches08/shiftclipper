import os
import re
import json
import traceback
import math
import time
import threading
import shutil
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
import numpy as np

from rq import get_current_job

# --- Optional heavy deps (installed in requirements.runpod_pro.txt) ---
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:  # pragma: no cover
    DeepSort = None

log = logging.getLogger("worker")
log.setLevel(logging.INFO)

DEBUG_MODE = os.getenv("WORKER_DEBUG", "0") == "1"
HEARTBEAT_SECONDS = float(os.getenv("WORKER_HEARTBEAT_SECONDS", "5"))
STALL_TIMEOUT_MINUTES = float(os.getenv("WORKER_STALL_TIMEOUT_MINUTES", "5"))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
JOBS_DIR = os.path.abspath(os.getenv("JOBS_DIR", os.path.join(DATA_DIR, "jobs")))


def _ensure_dirs() -> None:
    os.makedirs(JOBS_DIR, exist_ok=True)


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = (hex_color or "").strip().lstrip("#")
    if len(h) != 6:
        return (0, 0, 0)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def _mean_color_bgr(img: np.ndarray) -> Tuple[float, float, float]:
    if img.size == 0:
        return (0.0, 0.0, 0.0)
    m = img.reshape(-1, 3).mean(axis=0)
    return float(m[0]), float(m[1]), float(m[2])


def _color_sim(bgr_a: Tuple[float, float, float], bgr_b: Tuple[float, float, float]) -> float:
    da = np.array(bgr_a, dtype=np.float32)
    db = np.array(bgr_b, dtype=np.float32)
    dist = float(np.linalg.norm(da - db))
    return max(0.0, 1.0 - dist / 441.0)


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def _parse_digits(text: str) -> Optional[str]:
    if not text:
        return None
    digits = re.sub(r"\D+", "", text)
    return digits or None


@dataclass
class TrackingParams:
    yolo_conf: float = 0.25
    detect_stride: int = 2
    max_age: int = 40
    n_init: int = 2
    pre_roll: float = 2.0
    post_roll: float = 2.0
    gap_merge: float = 1.0
    min_clip_len: float = 6.0
    ocr_stride_s: float = 0.8
    ocr_min_conf: float = 0.35
    verify_mode: bool = True


def _device() -> str:
    if torch is None:
        return "cpu"
    try:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_yolo(model_path: str) -> Any:
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")
    return YOLO(model_path)


def _load_ocr(enabled: bool = True) -> Optional[Any]:
    if not enabled:
        return None
    if easyocr is None:
        return None
    try:
        return easyocr.Reader(["en"], gpu=_device().startswith("cuda"))
    except Exception:
        return easyocr.Reader(["en"], gpu=False)


def _extract_person_dets(yolo_res: Any) -> List[Tuple[Tuple[int, int, int, int], float]]:
    dets: List[Tuple[Tuple[int, int, int, int], float]] = []
    boxes = getattr(yolo_res, "boxes", None)
    if boxes is None:
        return dets
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        if int(k) != 0:
            continue
        dets.append(((int(x1), int(y1), int(x2), int(y2)), float(c)))
    return dets


def _best_track_for_click(tracks: List[Any], click_xy: Tuple[float, float], W: int, H: int) -> Optional[int]:
    cx, cy = click_xy
    px = int(cx * W)
    py = int(cy * H)
    best_id = None
    best_dist = 1e18
    for tr in tracks:
        if not tr.is_confirmed():
            continue
        l, t, r, b = tr.to_ltrb()
        l, t, r, b = int(l), int(t), int(r), int(b)
        if px < l or px > r or py < t or py > b:
            continue
        mx = (l + r) / 2.0
        my = (t + b) / 2.0
        d = (mx - px) ** 2 + (my - py) ** 2
        if d < best_dist:
            best_dist = d
            best_id = tr.track_id
    return best_id


def track_presence_spans_pro(
    video_path: str,
    clicks: List[Dict[str, float]],
    player_number: str,
    jersey_color_hex: str,
    opponent_color_hex: Optional[str],
    params: TrackingParams,
    cancel_check: Optional[Any] = None,
    progress_cb: Optional[Any] = None,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:

    if DeepSort is None:
        raise RuntimeError("deep-sort-realtime is not installed")

    model_path = os.path.join(BASE_DIR, "yolov8s.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8s.pt"

    yolo = _load_yolo(model_path)

    ocr = _load_ocr(params.verify_mode)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = (total_frames / fps) if total_frames and fps else 0.0

    player_bgr = _hex_to_bgr(jersey_color_hex)
    opp_bgr = _hex_to_bgr(opponent_color_hex) if opponent_color_hex else None
    target_digits = re.sub(r"\D+", "", (player_number or ""))
    target_digits = target_digits.lstrip("0") or target_digits

    tracker = DeepSort(
        max_age=params.max_age,
        n_init=params.n_init,
        nms_max_overlap=1.0,
        max_iou_distance=0.7,
        max_cosine_distance=0.4,
        nn_budget=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=_device().startswith("cuda"),
    )

    clicks_sorted = sorted((clicks or []), key=lambda c: float(c.get("t", 0.0)))
    debug_clicks = [{"t": c.get("t"), "x": c.get("x"), "y": c.get("y")} for c in clicks_sorted]

    chosen_track_id: Optional[int] = None
    chosen_votes: Dict[int, int] = {}
    last_box: Optional[Tuple[int, int, int, int]] = None
    last_seen_t: Optional[float] = None
    present_samples: List[Tuple[float, bool]] = []
    ocr_cache: Dict[int, Tuple[float, Optional[str], float]] = {}

    def maybe_ocr(frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[Optional[str], float]:
        if ocr is None:
            return None, 0.0
        x1, y1, x2, y2 = box
        y_top = y1 + int(0.20 * (y2 - y1))
        y_bot = y1 + int(0.75 * (y2 - y1))
        crop = frame_bgr[y_top:y_bot, x1:x2]
        if crop.size == 0:
            return None, 0.0
        try:
            results = ocr.readtext(crop, detail=1, paragraph=False)
        except Exception:
            return None, 0.0
        best_digits = None
        best_conf = 0.0
        for _bbox, text, conf in results:
            d = _parse_digits(str(text))
            if not d:
                continue
            if float(conf) > best_conf:
                best_conf = float(conf)
                best_digits = d
        if best_conf < params.ocr_min_conf:
            return None, float(best_conf)
        return best_digits, float(best_conf)

    next_click_idx = 0
    next_click_t = float(clicks_sorted[0]["t"]) if clicks_sorted else math.inf
    next_ocr_t = 0.0

    frame_idx = 0
    last_heartbeat = 0.0
    while True:
        if cancel_check and cancel_check():
            raise RuntimeError("Job cancelled by user.")
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx % params.detect_stride) != 0:
            frame_idx += 1
            continue

        t_s = frame_idx / fps

        pred = yolo.predict(source=frame, conf=params.yolo_conf, verbose=False, imgsz=640,
                            device=0 if _device().startswith("cuda") else "cpu")
        yres = pred[0] if isinstance(pred, (list, tuple)) and pred else pred
        dets = _extract_person_dets(yres)

        ds_in = []
        for (x1, y1, x2, y2), conf in dets:
            x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, W, H)
            ds_in.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks = tracker.update_tracks(ds_in, frame=frame)

        while t_s + (0.5 / fps) >= next_click_t and next_click_idx < len(clicks_sorted):
            c = clicks_sorted[next_click_idx]
            tid = _best_track_for_click(tracks, (float(c["x"]), float(c["y"])), W, H)
            if tid is not None:
                chosen_votes[tid] = chosen_votes.get(tid, 0) + 1
                chosen_track_id = max(chosen_votes.items(), key=lambda kv: kv[1])[0]
            next_click_idx += 1
            next_click_t = float(clicks_sorted[next_click_idx]["t"]) if next_click_idx < len(clicks_sorted) else math.inf

        present = False
        best_box = None
        best_score = None
        best_id = None

        candidates: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            l, t, r, b = tr.to_ltrb()
            box = _clip_box(int(l), int(t), int(r), int(b), W, H)
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            cmean = _mean_color_bgr(crop)
            sim_player = _color_sim(cmean, player_bgr)
            sim_opp = _color_sim(cmean, opp_bgr) if opp_bgr else 0.0
            color_term = sim_player - 0.6 * sim_opp

            dist_term = 0.0
            if last_box is not None:
                lx1, ly1, lx2, ly2 = last_box
                cx0 = (lx1 + lx2) / 2.0
                cy0 = (ly1 + ly2) / 2.0
                cx1 = (x1 + x2) / 2.0
                cy1 = (y1 + y2) / 2.0
                d = math.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)
                dist_term = 1.0 - min(1.0, d / (0.35 * math.sqrt(W * W + H * H)))

            ocr_digits = None
            ocr_conf = 0.0
            if ocr is not None:
                cached = ocr_cache.get(tr.track_id)
                if cached and (t_s - cached[0]) < params.ocr_stride_s:
                    ocr_digits, ocr_conf = cached[1], cached[2]
                elif t_s >= next_ocr_t and (chosen_track_id is None or tr.track_id == chosen_track_id):
                    ocr_digits, ocr_conf = maybe_ocr(frame, box)
                    ocr_cache[tr.track_id] = (t_s, ocr_digits, ocr_conf)
                    next_ocr_t = t_s + params.ocr_stride_s

            num_term = 0.0
            if ocr_digits and target_digits:
                if ocr_digits == target_digits or ocr_digits.lstrip("0") == target_digits.lstrip("0"):
                    num_term = 1.0
                elif target_digits in ocr_digits or ocr_digits in target_digits:
                    num_term = 0.6

            id_term = 0.0
            if chosen_track_id is not None:
                id_term = 1.0 if tr.track_id == chosen_track_id else -0.25

            score = 0.55 * id_term + 0.25 * color_term + 0.15 * dist_term + 0.35 * num_term
            candidates.append((score, tr.track_id, box))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_id, best_box = candidates[0]
            if chosen_track_id is not None:
                present = best_id == chosen_track_id
            else:
                present = (best_score or 0.0) > 0.15
                if present and best_id is not None and chosen_votes.get(best_id, 0) >= 1:
                    chosen_track_id = best_id

        if present and best_box is not None:
            last_box = best_box
            last_seen_t = t_s
        else:
            if last_seen_t is not None and (t_s - last_seen_t) <= 0.4:
                pass

        present_samples.append((t_s, bool(present)))

        if progress_cb is not None and total_frames:
            now = time.time()
            if (now - last_heartbeat) >= HEARTBEAT_SECONDS:
                progress_cb(frame_idx=frame_idx, total_frames=total_frames, t_s=t_s)
                last_heartbeat = now

        frame_idx += 1

    cap.release()

    spans: List[Tuple[float, float]] = []
    if present_samples:
        in_span = False
        start_t = 0.0
        for t_s, pr in present_samples:
            if pr and not in_span:
                in_span = True
                start_t = t_s
            if not pr and in_span:
                in_span = False
                spans.append((start_t, t_s))
        if in_span:
            spans.append((start_t, present_samples[-1][0]))

    rolled: List[Tuple[float, float]] = []
    for a, b in spans:
        aa = max(0.0, a - params.pre_roll)
        bb = min(duration_s if duration_s else b + params.post_roll, b + params.post_roll)
        rolled.append((aa, bb))
    rolled.sort(key=lambda x: x[0])

    merged: List[Tuple[float, float]] = []
    for a, b in rolled:
        if not merged:
            merged.append((a, b))
            continue
        pa, pb = merged[-1]
        if a <= pb + params.gap_merge:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))

    final_spans = [(a, b) for a, b in merged if (b - a) >= params.min_clip_len]

    debug = {
        "fps": fps,
        "W": W,
        "H": H,
        "total_frames": total_frames,
        "duration_s": duration_s,
        "detect_stride": params.detect_stride,
        "yolo_conf": params.yolo_conf,
        "ocr_enabled": bool(ocr is not None),
        "clicks": debug_clicks,
        "chosen_track_id": chosen_track_id,
        "votes": chosen_votes,
        "spans": final_spans,
    }

    return final_spans, debug


def _ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(cmd: List[str], *, job_id: Optional[str] = None, debug: bool = False) -> None:
    import subprocess

    if debug:
        log.info("job_id=%s ffmpeg_cmd=%s", job_id or "-", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def make_proxy(in_path: str, out_path: str, *, job_id: Optional[str] = None, debug: bool = False) -> None:
    if os.path.exists(out_path):
        return
    if not _ffmpeg_exists():
        raise RuntimeError("ffmpeg not found")
    _run([
        "ffmpeg", "-y", "-i", in_path,
        "-vf", "scale=-2:720",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-b:a", "96k",
        out_path,
    ], job_id=job_id, debug=debug)


def cut_clip(in_path: str, start: float, end: float, out_path: str, *, job_id: Optional[str] = None, debug: bool = False) -> None:
    if os.path.exists(out_path):
        return
    if not _ffmpeg_exists():
        raise RuntimeError("ffmpeg not found")
    dur = max(0.01, end - start)
    _run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", in_path,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ], job_id=job_id, debug=debug)


def concat_clips(clip_paths: List[str], out_path: str, *, job_id: Optional[str] = None, debug: bool = False) -> None:
    if os.path.exists(out_path):
        return
    if not clip_paths:
        return
    if not _ffmpeg_exists():
        raise RuntimeError("ffmpeg not found")
    lst = out_path + ".txt"
    with open(lst, "w", encoding="utf-8") as f:
        for pth in clip_paths:
            f.write(f"file '{pth}'\n")
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out_path], job_id=job_id, debug=debug)
    try:
        os.remove(lst)
    except Exception:
        pass


def process_job(job_id: str) -> Dict[str, Any]:

  class JobCancelled(Exception):
      pass

  def is_cancel_requested(path: str) -> bool:
      try:
          with open(path, "r", encoding="utf-8") as f:
              meta = json.load(f)
          return bool(meta.get("cancel_requested"))
      except Exception:
          return False

  watchdog_stop = None
  try:
      _ensure_dirs()
      cur = get_current_job()
      debug_mode = DEBUG_MODE
      stage_state = {"stage": "starting", "last_update": time.time(), "stalled": False}
      stall_timeout_s = max(60.0, STALL_TIMEOUT_MINUTES * 60.0)

      def check_stalled() -> None:
          if stage_state.get("stalled"):
              raise RuntimeError(f"stalled during {stage_state['stage']}")
          if (time.time() - stage_state["last_update"]) > stall_timeout_s:
              raise RuntimeError(f"stalled during {stage_state['stage']}")

      def update_status(stage: str, progress: int, message: str, status: str = "processing") -> None:
          stage_state["stage"] = stage
          stage_state["last_update"] = time.time()
          jobmeta.update({
              "status": status,
              "stage": stage,
              "progress": int(progress),
              "message": message,
              "updated_at": time.time(),
          })
          with open(meta_path, "w", encoding="utf-8") as f:
              json.dump(jobmeta, f, indent=2)
          if cur:
              cur.meta = {**(cur.meta or {}), "stage": stage, "progress": int(progress), "message": message}
              cur.save_meta()
          log.info("job_id=%s stage=%s progress=%s message=%s", job_id, stage, progress, message)

      job_dir = os.path.join(JOBS_DIR, job_id)
      meta_path = os.path.join(job_dir, "job.json")
      alt_meta_path = os.path.join(job_dir, "meta.json")
      if (not os.path.exists(meta_path)) and os.path.exists(alt_meta_path):
          meta_path = alt_meta_path

      if not os.path.exists(meta_path):
          raise RuntimeError(f"Missing job meta: {meta_path}")

      jobmeta = None
      for _ in range(10):
          try:
              with open(meta_path, "r", encoding="utf-8") as f:
                  jobmeta = json.load(f)
              break
          except Exception:
              time.sleep(0.05)

      if not isinstance(jobmeta, dict):
          raise RuntimeError("Could not read job meta JSON")
      debug_mode = debug_mode or bool(jobmeta.get("debug_mode"))

      watchdog_stop = False
      def _watchdog() -> None:
          while not watchdog_stop:
              time.sleep(2.0)
              if (time.time() - stage_state["last_update"]) > stall_timeout_s:
                  stage_state["stalled"] = True
                  log.error("job_id=%s stage=%s stalled watchdog timeout reached", job_id, stage_state["stage"])
                  break

      threading.Thread(target=_watchdog, daemon=True).start()

      in_path = jobmeta.get("video_path") or os.path.join(job_dir, "in.mp4")
      if not os.path.exists(in_path):
          raise RuntimeError(f"Missing input video: {in_path}")

      if is_cancel_requested(meta_path):
          raise JobCancelled("Job cancelled before processing started.")

      proxy_path = os.path.join(job_dir, "input_proxy.mp4")
      update_status("proxy", 2, "Preparing proxy")
      make_proxy(in_path, proxy_path, job_id=job_id, debug=debug_mode)

      jobmeta["proxy_path"] = proxy_path
      jobmeta["proxy_url"] = f"/data/jobs/{job_id}/input_proxy.mp4"
      jobmeta["proxy_ready"] = True

      setup_path = os.path.join(job_dir, "setup.json")
      setup = jobmeta.get("setup") or {}
      if (not setup) and os.path.exists(setup_path):
          with open(setup_path, "r", encoding="utf-8") as f:
              setup = json.load(f)
      verify_only = bool(setup.get("verify_mode", False)) or os.getenv("WORKER_VERIFY_ONLY", "0") == "1"
      if cv2 is None and not verify_only:
          raise RuntimeError("opencv-python is not installed; set verify_mode=true or WORKER_VERIFY_ONLY=1 for wiring checks")

      if verify_only:
          jobmeta["clips"] = []
          jobmeta["combined_path"] = None
          jobmeta["combined_url"] = None
          jobmeta["debug"] = {
              "verify_only": True,
              "reason": "verify_mode enabled" if bool(setup.get("verify_mode", False)) else "WORKER_VERIFY_ONLY=1",
              "cv2_available": bool(cv2 is not None),
          }
          jobmeta["status"] = "done"
          jobmeta["progress"] = 100
          jobmeta["stage"] = "done"
          jobmeta["message"] = "Verify-only mode completed (queue + status wiring)."
          jobmeta["updated_at"] = time.time()
          with open(meta_path, "w", encoding="utf-8") as f:
              json.dump(jobmeta, f, indent=2)
          results_path = os.path.join(job_dir, "results.json")
          with open(results_path, "w", encoding="utf-8") as f:
              json.dump(jobmeta, f, indent=2)
          if cur:
              cur.meta = {**(cur.meta or {}), "stage": "done", "progress": 100}
              cur.save_meta()
          return jobmeta

      clicks = setup.get("clicks") or []
      jersey_color = setup.get("jersey_color") or "#203524"
      opponent_color = setup.get("opponent_color") or None
      player_number = str(setup.get("player_number") or "")

      params = TrackingParams(
          detect_stride=int(setup.get("detect_stride") or 2),
          yolo_conf=float(setup.get("yolo_conf") or 0.25),
          pre_roll=float(setup.get("pre_roll") or 2.0),
          post_roll=float(setup.get("post_roll") or setup.get("extend_sec") or 2.0),
          gap_merge=float(setup.get("gap_merge") or 1.0),
          min_clip_len=float(setup.get("min_clip_len") or 6.0),
          ocr_min_conf=float(setup.get("ocr_min_conf") or 0.35),
          verify_mode=bool(setup.get("verify_mode", True)),
      )

      log.info("job_id=%s debug_mode=%s model_device=%s", job_id, debug_mode, _device())

      update_status("tracking", 10, "Starting tracking")

      def tracking_heartbeat(frame_idx: int, total_frames: int, t_s: float) -> None:
          pct = int(min(80, 10 + (frame_idx / max(1, total_frames)) * 65))
          update_status("tracking", pct, f"Tracking in progress ({t_s:.1f}s)")
          check_stalled()

      spans, debug = track_presence_spans_pro(
          video_path=in_path,
          clicks=clicks,
          player_number=player_number,
          jersey_color_hex=jersey_color,
          opponent_color_hex=opponent_color,
          params=params,
          cancel_check=lambda: is_cancel_requested(meta_path) or stage_state.get("stalled", False),
          progress_cb=tracking_heartbeat,
      )

      clips_dir = os.path.join(job_dir, "clips")
      os.makedirs(clips_dir, exist_ok=True)
      update_status("exporting", 82, "Exporting clips")

      clips: List[Dict[str, Any]] = []
      clip_paths: List[str] = []
      for i, (a, b) in enumerate(spans, start=1):
          check_stalled()
          if is_cancel_requested(meta_path):
              raise JobCancelled("Job cancelled while building clips.")
          outp = os.path.join(clips_dir, f"clip_{i:03d}.mp4")
          cut_clip(in_path, a, b, outp, job_id=job_id, debug=debug_mode)
          clip_paths.append(outp)
          clips.append({"start": float(a), "end": float(b), "path": outp, "url": f"/data/jobs/{job_id}/clips/clip_{i:03d}.mp4"})
          step_pct = 82 + int((i / max(1, len(spans))) * 12)
          update_status("exporting", min(94, step_pct), f"Exporting clips ({i}/{len(spans)})")

      combined_path = os.path.join(job_dir, "combined.mp4")
      update_status("exporting", 95, "Combining clips")
      concat_clips(clip_paths, combined_path, job_id=job_id, debug=debug_mode)
      update_status("exporting", 98, "Finished exporting clips")

      jobmeta["clips"] = clips
      jobmeta["combined_path"] = combined_path if os.path.exists(combined_path) else None
      jobmeta["combined_url"] = f"/data/jobs/{job_id}/combined.mp4" if os.path.exists(combined_path) else None
      jobmeta["debug"] = debug
      jobmeta["status"] = "done"
      jobmeta["progress"] = 100
      jobmeta["stage"] = "done"
      jobmeta["updated_at"] = time.time()
      jobmeta["message"] = "Processing complete"

      with open(meta_path, "w", encoding="utf-8") as f:
          json.dump(jobmeta, f, indent=2)

      results_path = os.path.join(job_dir, "results.json")
      with open(results_path, "w", encoding="utf-8") as f:
          json.dump(jobmeta, f, indent=2)

      if cur:
          cur.meta = {**(cur.meta or {}), "stage": "done", "progress": 100}
          cur.save_meta()
      log.info("job_id=%s stage=done progress=100 message=Processing complete", job_id)

      return jobmeta
  except JobCancelled as e:
    fail_meta = {
      "job_id": job_id,
      "status": "cancelled",
      "stage": "cancelled",
      "progress": 100,
      "message": str(e),
      "updated_at": time.time(),
    }
    os.makedirs(JOBS_DIR, exist_ok=True)
    failed_job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(failed_job_dir, exist_ok=True)
    fail_meta_path = os.path.join(failed_job_dir, "job.json")
    with open(fail_meta_path, "w", encoding="utf-8") as f:
      json.dump(fail_meta, f, indent=2)
    return fail_meta
  except Exception as e:
    tb = traceback.format_exc()
    try:
      os.makedirs(JOBS_DIR, exist_ok=True)
      failed_job_dir = os.path.join(JOBS_DIR, job_id)
      os.makedirs(failed_job_dir, exist_ok=True)
      fail_meta_path = os.path.join(failed_job_dir, "job.json")
      fail_meta = {
        "job_id": job_id,
        "status": "failed",
        "stage": "failed",
        "progress": 100,
        "message": f"Error: {e}",
        "error": {"type": type(e).__name__, "detail": str(e), "traceback": tb},
        "updated_at": time.time(),
      }
      with open(fail_meta_path, "w", encoding="utf-8") as f:
        json.dump(fail_meta, f, indent=2)
    except Exception:
      pass
    raise
  finally:
    watchdog_stop = True


def self_test_task(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "payload": payload or {},
        "worker_pid": os.getpid(),
        "cv2_available": bool(cv2 is not None),
        "updated_at": time.time(),
    }
