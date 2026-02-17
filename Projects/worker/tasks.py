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
    detect_stride: int = 1
    max_age: int = 40
    n_init: int = 2
    pre_roll: float = 1.0
    post_roll: float = 2.0
    gap_merge_seconds: float = 1.0
    min_track_seconds: float = 1.2
    ocr_stride_s: float = 0.4
    ocr_min_conf: float = 0.3
    lock_seconds_after_confirm: float = 1.5
    lost_timeout_seconds: float = 1.0
    jersey_color_tolerance: float = 95.0
    verify_mode: bool = False
    require_confirm_hits: int = 2
    confirm_window_frames: int = 6
    debug_overlay: bool = False
    device: str = "cpu"


def resolve_device() -> str:
    override = (os.getenv("SHIFTCLIPPER_DEVICE") or "").strip()
    if override:
        return override
    if torch is None:
        return "cpu"
    try:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _device_info(device: str) -> Dict[str, Any]:
    info = {"device": device, "torch_version": None, "cuda_available": False, "gpu_name": None}
    if torch is None:
        return info
    info["torch_version"] = getattr(torch, "__version__", None)
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"] and str(device).startswith("cuda"):
            idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
            info["gpu_name"] = torch.cuda.get_device_name(idx)
    except Exception:
        pass
    return info


def _load_yolo(model_path: str) -> Any:
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")
    return YOLO(model_path)


def _load_ocr(enabled: bool = True, device: str = "cpu") -> Optional[Any]:
    if not enabled:
        return None
    if easyocr is None:
        return None
    try:
        return easyocr.Reader(["en"], gpu=str(device).startswith("cuda"))
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
    debug_overlay_path: Optional[str] = None,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    if DeepSort is None:
        raise RuntimeError("deep-sort-realtime is not installed")

    model_path = os.path.join(BASE_DIR, "yolov8s.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8s.pt"
    yolo = _load_yolo(model_path)
    ocr = _load_ocr(True, params.device)

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
    target_digits = re.sub(r"\D+", "", (player_number or "")).lstrip("0")

    tracker = DeepSort(max_age=params.max_age, n_init=params.n_init, nms_max_overlap=1.0, max_iou_distance=0.7,
                       max_cosine_distance=0.4, nn_budget=None, embedder="mobilenet", half=True, bgr=True,
                       embedder_gpu=str(params.device).startswith("cuda"))

    writer = None
    if debug_overlay_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(debug_overlay_path, fourcc, fps, (W, H))

    clicks_sorted = sorted((clicks or []), key=lambda c: float(c.get("t", 0.0)))
    next_click_idx = 0
    next_click_t = float(clicks_sorted[0]["t"]) if clicks_sorted else math.inf
    chosen_votes: Dict[int, int] = {}
    chosen_track_id: Optional[int] = None

    state = "searching"
    state_events: List[Dict[str, Any]] = []
    ocr_hits: List[Tuple[int, bool]] = []
    last_confirm_t = -1e9
    last_seen_t = -1e9
    segment_start: Optional[float] = None
    raw_segments: List[Tuple[float, float]] = []

    frame_idx = 0
    last_heartbeat = 0.0
    ocr_cache: Dict[int, Tuple[float, Optional[str], float]] = {}

    def set_state(new_state: str, t_s: float, reason: str) -> None:
        nonlocal state
        if new_state != state:
            state_events.append({"t": round(t_s, 3), "from": state, "to": new_state, "reason": reason})
            state = new_state

    def maybe_ocr(frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[Optional[str], float]:
        if ocr is None:
            return None, 0.0
        x1, y1, x2, y2 = box
        crop = frame_bgr[y1 + int(0.2 * (y2 - y1)):y1 + int(0.75 * (y2 - y1)), x1:x2]
        if crop.size == 0:
            return None, 0.0
        try:
            out = ocr.readtext(crop, detail=1, paragraph=False)
        except Exception:
            return None, 0.0
        best_d, best_c = None, 0.0
        for _bb, txt, conf in out:
            d = _parse_digits(str(txt))
            if d and float(conf) > best_c:
                best_d, best_c = d, float(conf)
        if best_c < params.ocr_min_conf:
            return None, best_c
        return best_d, best_c

    while True:
        if cancel_check and cancel_check():
            raise RuntimeError("Job cancelled by user.")
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, params.detect_stride) != 0:
            frame_idx += 1
            continue

        t_s = frame_idx / fps
        pred = yolo.predict(source=frame, conf=params.yolo_conf, verbose=False, imgsz=640,
                            device=params.device if str(params.device).startswith("cuda") else "cpu")
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

        best = None
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            l, t, r, b = tr.to_ltrb()
            box = _clip_box(int(l), int(t), int(r), int(b), W, H)
            x1, y1, x2, y2 = box
            cmean = _mean_color_bgr(frame[y1:y2, x1:x2])
            sim_player = _color_sim(cmean, player_bgr)
            sim_opp = _color_sim(cmean, opp_bgr) if opp_bgr else 0.0
            color_score = sim_player - 0.65 * sim_opp
            color_gate = float(np.linalg.norm(np.array(cmean) - np.array(player_bgr))) <= params.jersey_color_tolerance
            ocr_digits, ocr_conf = ocr_cache.get(tr.track_id, (0.0, None, 0.0))[1:]
            cached_t = ocr_cache.get(tr.track_id, (0.0, None, 0.0))[0]
            if t_s - cached_t >= params.ocr_stride_s:
                ocr_digits, ocr_conf = maybe_ocr(frame, box)
                ocr_cache[tr.track_id] = (t_s, ocr_digits, ocr_conf)
            ocr_match = bool(target_digits and ocr_digits and (ocr_digits.lstrip("0") == target_digits))
            id_bonus = 0.6 if (chosen_track_id is not None and tr.track_id == chosen_track_id) else 0.0
            score = color_score + id_bonus + (0.9 if ocr_match else 0.0)
            item = {"track": tr.track_id, "box": box, "ocr": ocr_digits, "ocr_conf": ocr_conf,
                    "ocr_match": ocr_match, "color_score": color_score, "color_gate": color_gate, "score": score}
            if best is None or item["score"] > best["score"]:
                best = item

        active = best is not None and (best["color_gate"] or (state in {"confirmed", "locked"}))
        if active:
            last_seen_t = t_s

        ocr_ok = bool(best and best["ocr_match"] and best["ocr_conf"] >= params.ocr_min_conf)
        ocr_hits.append((frame_idx, ocr_ok))
        if len(ocr_hits) > params.confirm_window_frames:
            ocr_hits.pop(0)
        hit_count = sum(1 for _, h in ocr_hits if h)
        confirm_now = hit_count >= params.require_confirm_hits

        if confirm_now and active:
            last_confirm_t = t_s
            if segment_start is None:
                segment_start = t_s
            set_state("confirmed", t_s, "ocr_confirm")
        elif state in {"confirmed", "locked"} and (t_s - last_confirm_t) <= params.lock_seconds_after_confirm and active:
            if segment_start is None:
                segment_start = t_s
            set_state("locked", t_s, "lock_window")
        elif state in {"confirmed", "locked"} and (t_s - last_seen_t) > params.lost_timeout_seconds:
            if segment_start is not None:
                raw_segments.append((segment_start, t_s))
                segment_start = None
            set_state("searching", t_s, "lost_timeout")
        elif state == "searching" and active:
            set_state("searching", t_s, "candidate")

        if writer is not None:
            out = frame.copy()
            if best:
                x1, y1, x2, y2 = best["box"]
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, f"track={best['track']} state={state}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(out, f"ocr={best['ocr']} conf={best['ocr_conf']:.2f} color={best['color_score']:.2f}",
                            (x1, min(H - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(out, f"t={t_s:.2f}s", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            writer.write(out)

        if progress_cb is not None and total_frames:
            now = time.time()
            if (now - last_heartbeat) >= HEARTBEAT_SECONDS:
                progress_cb(frame_idx=frame_idx, total_frames=total_frames, t_s=t_s)
                last_heartbeat = now

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if segment_start is not None:
        raw_segments.append((segment_start, duration_s or segment_start))

    rolled = [(max(0.0, a - params.pre_roll), min(duration_s if duration_s else (b + params.post_roll), b + params.post_roll)) for a, b in raw_segments]
    rolled.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    merge_events: List[Dict[str, float]] = []
    for a, b in rolled:
        if not merged:
            merged.append((a, b))
            continue
        pa, pb = merged[-1]
        if a - pb <= params.gap_merge_seconds:
            merge_events.append({"prev_end": round(pb, 3), "next_start": round(a, 3), "gap": round(a - pb, 3)})
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    final_spans = [(a, b) for a, b in merged if (b - a) >= params.min_track_seconds]

    return final_spans, {
        "fps": fps,
        "W": W,
        "H": H,
        "duration_s": duration_s,
        "params": params.__dict__,
        "chosen_track_id": chosen_track_id,
        "votes": chosen_votes,
        "state_transitions": state_events,
        "raw_segments": raw_segments,
        "merged_segments": final_spans,
        "merge_events": merge_events,
        "debug_overlay_path": debug_overlay_path,
    }

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
              raise RuntimeError(f"stalled during {stage_state['stage']}; watchdog timeout hit, check model/device and debug timeline")
          if (time.time() - stage_state["last_update"]) > stall_timeout_s:
              raise RuntimeError(f"stalled during {stage_state['stage']}; watchdog timeout hit, check model/device and debug timeline")

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
          jobmeta["artifacts"] = []
          jobmeta["debug"] = {
              "verify_only": True,
              "reason": "verify_mode enabled" if bool(setup.get("verify_mode", False)) else "WORKER_VERIFY_ONLY=1",
              "cv2_available": bool(cv2 is not None),
          }
          jobmeta["status"] = "verified"
          jobmeta["progress"] = 100
          jobmeta["stage"] = "verified"
          jobmeta["message"] = "verify_mode=true: verified only; no clips/combined will be created."
          jobmeta["updated_at"] = time.time()
          with open(meta_path, "w", encoding="utf-8") as f:
              json.dump(jobmeta, f, indent=2)
          results_path = os.path.join(job_dir, "results.json")
          with open(results_path, "w", encoding="utf-8") as f:
              json.dump(jobmeta, f, indent=2)
          if cur:
              cur.meta = {**(cur.meta or {}), "stage": "verified", "progress": 100, "message": jobmeta["message"]}
              cur.save_meta()
          return jobmeta

      clicks = setup.get("clicks") or []
      jersey_color = setup.get("jersey_color") or "#203524"
      opponent_color = setup.get("opponent_color") or None
      player_number = str(setup.get("player_number") or "")

      device = resolve_device()
      params = TrackingParams(
          detect_stride=int(setup.get("detect_stride") or 1),
          yolo_conf=float(setup.get("yolo_conf") or 0.25),
          pre_roll=float(setup.get("pre_roll") or 0.5),
          post_roll=float(setup.get("post_roll") or setup.get("extend_sec") or 2.0),
          gap_merge_seconds=float(setup.get("gap_merge_seconds") or setup.get("gap_merge") or 1.0),
          min_track_seconds=float(setup.get("min_track_seconds") or setup.get("min_clip_len") or 1.0),
          ocr_min_conf=float(setup.get("ocr_min_conf") or 0.3),
          lock_seconds_after_confirm=float(setup.get("lock_seconds_after_confirm") or 1.5),
          lost_timeout_seconds=float(setup.get("lost_timeout_seconds") or 1.0),
          jersey_color_tolerance=float(setup.get("jersey_color_tolerance") or 95.0),
          verify_mode=bool(setup.get("verify_mode", False)),
          debug_overlay=bool(setup.get("debug_overlay", False)),
          device=device,
      )
      dlog = _device_info(device)
      log.info("job_id=%s stage=startup device=%s torch=%s cuda_available=%s gpu=%s", job_id, dlog.get("device"), dlog.get("torch_version"), dlog.get("cuda_available"), dlog.get("gpu_name"))

      update_status("tracking", 10, "Running advanced tracker")

      def tracking_heartbeat(frame_idx: int, total_frames: int, t_s: float) -> None:
          pct = int(min(80, 10 + (frame_idx / max(1, total_frames)) * 65))
          update_status("tracking", pct, f"Tracking in progress ({t_s:.1f}s)")
          check_stalled()

      debug_overlay_path = os.path.join(job_dir, "debug_overlay.mp4") if params.debug_overlay else None
      spans, debug = track_presence_spans_pro(
          video_path=in_path,
          clicks=clicks,
          player_number=player_number,
          jersey_color_hex=jersey_color,
          opponent_color_hex=opponent_color,
          params=params,
          cancel_check=lambda: is_cancel_requested(meta_path) or stage_state.get("stalled", False),
          progress_cb=tracking_heartbeat,
          debug_overlay_path=debug_overlay_path,
      )

      debug_timeline_path = os.path.join(job_dir, "debug.json")
      with open(debug_timeline_path, "w", encoding="utf-8") as f:
          json.dump(debug, f, indent=2)

      clips_dir = os.path.join(job_dir, "clips")
      os.makedirs(clips_dir, exist_ok=True)
      update_status("exporting", 82, "Exporting clips")

      clips: List[Dict[str, Any]] = []
      artifacts: List[Dict[str, Any]] = []
      clip_paths: List[str] = []
      for i, (a, b) in enumerate(spans, start=1):
          check_stalled()
          if is_cancel_requested(meta_path):
              raise JobCancelled("Job cancelled while building clips.")
          outp = os.path.join(clips_dir, f"clip_{i:03d}.mp4")
          cut_clip(in_path, a, b, outp, job_id=job_id, debug=debug_mode)
          if not os.path.exists(outp):
              raise RuntimeError(f"ffmpeg output missing clip: {outp}")
          clip_paths.append(outp)
          clip_url = f"/data/jobs/{job_id}/clips/clip_{i:03d}.mp4"
          clips.append({"start": float(a), "end": float(b), "path": outp, "url": clip_url})
          artifacts.append({"type": "clip", "path": outp, "url": clip_url})
          step_pct = 82 + int((i / max(1, len(spans))) * 12)
          update_status("exporting", min(94, step_pct), f"Exporting clips ({i}/{len(spans)})")

      combined_enabled = bool(setup.get("enable_combined", True))
      combined_path = os.path.join(job_dir, "combined.mp4")
      if combined_enabled and clip_paths:
          update_status("exporting", 95, "Combining clips")
          concat_clips(clip_paths, combined_path, job_id=job_id, debug=debug_mode)
      update_status("exporting", 98, "Finished exporting clips")

      if (not clip_paths) and not os.path.exists(combined_path):
          checked = [clips_dir, combined_path]
          raise RuntimeError(f"No clips created; see debug overlay/timeline. checked_paths={checked}")

      if os.path.exists(combined_path):
          artifacts.append({"type": "combined", "path": combined_path, "url": f"/data/jobs/{job_id}/combined.mp4"})
      artifacts.append({"type": "debug_timeline", "path": debug_timeline_path, "url": f"/data/jobs/{job_id}/debug.json"})
      if debug_overlay_path and os.path.exists(debug_overlay_path):
          artifacts.append({"type": "debug_overlay", "path": debug_overlay_path, "url": f"/data/jobs/{job_id}/debug_overlay.mp4"})

      jobmeta["clips"] = clips
      jobmeta["artifacts"] = artifacts
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
