import os
import time
import json
import csv
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2


def _read_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _set_meta(meta_path: str, status: str, extra: Optional[Dict[str, Any]] = None):
    meta = _read_json(meta_path, {})
    meta["status"] = status
    meta["updated_at"] = time.time()
    if extra:
        meta.update(extra)
    _write_json(meta_path, meta)


def _run(cmd: List[str]):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def _ffmpeg_clip(in_path: str, out_path: str, start: float, end: float) -> bool:
    # Try stream-copy for speed; if it fails, fallback to re-encode.
    dur = max(0.05, end - start)

    cmd_copy = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", in_path,
        "-t", f"{dur:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    _run(cmd_copy)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    cmd_enc = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", in_path,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]
    _run(cmd_enc)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def _ffmpeg_concat(clips_abs: List[str], out_path: str) -> bool:
    if not clips_abs:
        return False
    list_path = out_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clips_abs:
            f.write(f"file '{p}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path]
    _run(cmd)
    try:
        os.remove(list_path)
    except OSError:
        pass
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def _clamp_box_xyxy(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(w - 2, x1)))
    y1 = int(max(0, min(h - 2, y1)))
    x2 = int(max(1, min(w - 1, x2)))
    y2 = int(max(1, min(h - 1, y2)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 2)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 2)
    return x1, y1, x2, y2


def _norm_to_px(box_norm, w, h):
    x1, y1, x2, y2 = box_norm
    return _clamp_box_xyxy(int(round(x1 * w)), int(round(y1 * h)), int(round(x2 * w)), int(round(y2 * h)), w, h)


def _group_segments(times: List[float], min_gap: float, pad: float, min_len: float) -> List[Tuple[float, float]]:
    if not times:
        return []
    times = sorted(times)
    segs: List[Tuple[float, float]] = []
    s = times[0]
    prev = times[0]
    for t in times[1:]:
        if t - prev > min_gap:
            segs.append((max(0.0, s - pad), prev + pad))
            s = t
        prev = t
    segs.append((max(0.0, s - pad), prev + pad))

    merged: List[Tuple[float, float]] = []
    for a, b in segs:
        if not merged:
            merged.append((a, b))
        else:
            la, lb = merged[-1]
            if a <= lb + 0.25:
                merged[-1] = (la, max(lb, b))
            else:
                merged.append((a, b))

    out: List[Tuple[float, float]] = []
    for a, b in merged:
        if (b - a) < min_len:
            b = a + min_len
        out.append((a, b))
    return out


def _split_segments(segs: List[Tuple[float, float]], max_len: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in segs:
        if (b - a) <= max_len:
            out.append((a, b))
        else:
            t = a
            while t < b:
                out.append((t, min(b, t + max_len)))
                t += max_len
    return out


# ---------------------------
# ReID-lite signature (CPU)
# ---------------------------
def _sig_from_crop(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # hist: (256,) HSV (H 16 x S 16), tex: (1024,) edges 32x32
    if bgr.size == 0:
        return np.zeros((256,), np.float32), np.zeros((1024,), np.float32)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hch = hsv[:, :, 0]
    sch = hsv[:, :, 1]
    hist = cv2.calcHist([hch, sch], [0, 1], None, [16, 16], [0, 180, 0, 256]).astype(np.float32).reshape(-1)
    hist /= (float(hist.sum()) + 1e-6)

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    e = cv2.Canny(g, 60, 160)
    e = cv2.resize(e, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    tex = e.reshape(-1)
    tex /= (float(np.linalg.norm(tex)) + 1e-6)

    return hist, tex


def _hist_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - (float(np.dot(a, b)) / ((float(np.linalg.norm(a)) + 1e-6) * (float(np.linalg.norm(b)) + 1e-6))))


def _combine_dist(hist_d: float, tex_d: float) -> float:
    return 0.65 * hist_d + 0.35 * tex_d


def _avg_sigs(hists: List[np.ndarray], texs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    hist = np.mean(np.stack(hists, axis=0), axis=0).astype(np.float32) if hists else np.zeros((256,), np.float32)
    tex = np.mean(np.stack(texs, axis=0), axis=0).astype(np.float32) if texs else np.zeros((1024,), np.float32)
    hist = hist / (float(hist.sum()) + 1e-6)
    tex = tex / (float(np.linalg.norm(tex)) + 1e-6)
    return hist, tex


def _process_broadcast_dettrack_reid_lite(
    video_path: str,
    job_id: str,
    job_dir: str,
    setup: Dict[str, Any],
    seeds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Broadcast-grade baseline:
      - If setup.selected_click is provided -> choose the track that contains the click at that time (no seeds needed).
      - Else -> use ReID-lite scoring against player.seeds (requires >=3 seeds).
    """
    debug_csv = os.path.join(job_dir, "debug_tracks.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "error": "OpenCV could not open the video. Convert to MP4 (H.264) and retry."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    # Optional click-selection (preferred UX)
    selected_click = setup.get("selected_click") or {}
    click_t = selected_click.get("t", None)
    click_x = selected_click.get("x", None)
    click_y = selected_click.get("y", None)
    click_ready = (click_t is not None) and (click_x is not None) and (click_y is not None)

    # Optional manual override
    selected_track_id = setup.get("selected_track_id", None)

    # Seed signature (only needed if no click selection)
    seed_hist = None
    seed_tex = None
    seed_hists: List[np.ndarray] = []
    seed_texs: List[np.ndarray] = []

    if (not click_ready) and (selected_track_id is None):
        cap2 = cv2.VideoCapture(video_path)
        for s in sorted(seeds, key=lambda x: float(x.get("t", 0.0))):
            t = float(s.get("t", 0.0))
            cap2.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap2.read()
            if not ok:
                continue
            x1, y1, x2, y2 = _norm_to_px(s["box"], w, h)
            crop = frame[y1:y2, x1:x2]
            hist, tex = _sig_from_crop(crop)
            seed_hists.append(hist)
            seed_texs.append(tex)
        cap2.release()

        if len(seed_hists) < 3:
            return {
                "status": "error",
                "error": "No player selected. Either click the player once on the video, or provide at least 3 good seeds (tight torso).",
            }

        seed_hist, seed_tex = _avg_sigs(seed_hists, seed_texs)

    try:
        from ultralytics import YOLO
    except Exception as e:
        return {"status": "error", "error": f"Missing dependency ultralytics. Rebuild image. Details: {e}"}

    model_name = setup.get("det_model", "yolov8n.pt")
    conf = float(setup.get("det_conf", 0.28))
    iou = float(setup.get("det_iou", 0.45))
    imgsz = int(setup.get("imgsz", 640))
    det_stride = int(setup.get("det_stride", 4))
    max_w = int(setup.get("max_width", 960))
    sig_stride = int(setup.get("sig_stride", 6))

    # Track state
    track_times: Dict[int, List[float]] = {}
    track_hist_sum: Dict[int, np.ndarray] = {}
    track_tex_sum: Dict[int, np.ndarray] = {}
    track_sig_n: Dict[int, int] = {}
    track_last_sig_frame: Dict[int, int] = {}

    # For click-selection
    click_track_id: Optional[int] = int(selected_track_id) if selected_track_id is not None else None
    click_found_at: Optional[float] = None

    yolo = YOLO(model_name)

    try:
        results_iter = yolo.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device="cpu",
            verbose=False,
        )
    except Exception as e:
        return {"status": "error", "error": f"Ultralytics tracking failed (model download/network?). Details: {e}"}

    debug_rows: List[List[Any]] = []

    # Window (seconds) around click time to accept a selection
    click_window = float(setup.get("click_window", max(0.25, (det_stride / max(1e-6, fps)) * 2.0)))

    for frame_idx, r in enumerate(results_iter):
        if frame_idx % det_stride != 0:
            continue

        frame = getattr(r, "orig_img", None)
        if frame is None:
            continue

        fh, fw = frame.shape[:2]
        scale = 1.0
        if fw > max_w:
            scale = max_w / float(fw)
            frame = cv2.resize(
                frame,
                (int(round(fw * scale)), int(round(fh * scale))),
                interpolation=cv2.INTER_AREA
            )

        t = float(frame_idx / max(1e-6, fps))

        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.id is None:
            debug_rows.append([frame_idx, round(t, 3), 0])
            continue

        ids = boxes.id.cpu().numpy().astype(int).tolist()
        xyxy = boxes.xyxy.cpu().numpy().tolist()

        # If click selection is enabled and we haven't picked a track yet, try to pick here
        if click_ready and (click_track_id is None) and (abs(float(t) - float(click_t)) <= click_window):
            px = float(click_x) * float(frame.shape[1])
            py = float(click_y) * float(frame.shape[0])
            best_tid = None
            best_area = None
            for tid, b in zip(ids, xyxy):
                x1, y1, x2, y2 = b
                if scale != 1.0:
                    x1 *= scale; x2 *= scale; y1 *= scale; y2 *= scale
                x1, y1, x2, y2 = _clamp_box_xyxy(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
                if (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2):
                    area = (x2 - x1) * (y2 - y1)
                    if best_area is None or area < best_area:
                        best_area = area
                        best_tid = int(tid)
            if best_tid is not None:
                click_track_id = int(best_tid)
                click_found_at = float(t)

        # Collect track times and signatures
        for tid, b in zip(ids, xyxy):
            x1, y1, x2, y2 = b
            if scale != 1.0:
                x1 *= scale; x2 *= scale; y1 *= scale; y2 *= scale
            x1, y1, x2, y2 = _clamp_box_xyxy(x1, y1, x2, y2, frame.shape[1], frame.shape[0])

            track_times.setdefault(int(tid), []).append(t)

            # If we are using click-only mode, signature collection is optional. If seeds mode, we need sigs.
            if seed_hist is None:
                continue

            lf = track_last_sig_frame.get(int(tid), -10**9)
            if (frame_idx - lf) < sig_stride:
                continue
            track_last_sig_frame[int(tid)] = frame_idx

            crop = frame[y1:y2, x1:x2]
            hist, tex = _sig_from_crop(crop)

            if int(tid) not in track_hist_sum:
                track_hist_sum[int(tid)] = hist.copy()
                track_tex_sum[int(tid)] = tex.copy()
                track_sig_n[int(tid)] = 1
            else:
                track_hist_sum[int(tid)] += hist
                track_tex_sum[int(tid)] += tex
                track_sig_n[int(tid)] += 1

        debug_rows.append([frame_idx, round(t, 3), len(set(ids))])

    # Write debug
    try:
        with open(debug_csv, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["frame_idx", "t", "num_tracks"])
            wcsv.writerows(debug_rows)
    except Exception:
        pass

    # Choose track
    chosen_id = None
    chosen_d = None
    chosen_n = None
    scored: List[Tuple[int, float, int]] = []

    if click_ready or (selected_track_id is not None):
        if click_track_id is None:
            return {
                "status": "error",
                "error": "Could not select a player. Pause near the player, then click directly on their body (not ice/boards).",
                "debug": {"click_t": click_t, "click_window": click_window},
                "outputs": {"debug_tracks_csv": f"/data/jobs/{job_id}/debug_tracks.csv"},
            }
        chosen_id = int(click_track_id)
        chosen_times = track_times.get(chosen_id, [])
        if not chosen_times:
            return {
                "status": "error",
                "error": f"Selected track {chosen_id} had no timeline. Try clicking again later in the video.",
                "outputs": {"debug_tracks_csv": f"/data/jobs/{job_id}/debug_tracks.csv"},
            }
        chosen_d = 0.0
        chosen_n = 0
    else:
        # seeds + ReID-lite scoring
        for tid, ts in track_times.items():
            n = track_sig_n.get(tid, 0)
            if n < 2:
                continue
            th = track_hist_sum[tid] / float(n)
            tt = track_tex_sum[tid] / float(n)
            th = th / (float(th.sum()) + 1e-6)
            tt = tt / (float(np.linalg.norm(tt)) + 1e-6)

            d = _combine_dist(_hist_dist(seed_hist, th), _cos_dist(seed_tex, tt))
            scored.append((int(tid), float(d), int(n)))

        if not scored:
            return {"status": "error", "error": "No stable tracks found. Try lowering det_conf to 0.22 or clicking the player instead."}

        scored.sort(key=lambda x: x[1])
        chosen_id, chosen_d, chosen_n = scored[0]
        chosen_times = track_times.get(int(chosen_id), [])

    # Segments
    min_gap = float(setup.get("seg_gap", 1.2))
    pad = float(setup.get("seg_pad", 1.6))
    min_len = float(setup.get("seg_min_len", 7.0))
    max_len = float(setup.get("seg_max_len", 22.0))

    segs = _group_segments(chosen_times, min_gap=min_gap, pad=pad, min_len=min_len)
    segs = _split_segments(segs, max_len=max_len)

    return {
        "status": "done",
        "mode": "broadcast_dettrack_reid_lite",
        "fps": float(fps),
        "video_w": int(w),
        "video_h": int(h),
        "selection": "click" if (click_ready or (selected_track_id is not None)) else "seeds",
        "selected_click": {"t": float(click_t), "x": float(click_x), "y": float(click_y), "found_at": click_found_at} if click_ready else None,
        "chosen_track_id": int(chosen_id),
        "chosen_track_distance": float(chosen_d) if chosen_d is not None else None,
        "chosen_track_sig_samples": int(chosen_n) if chosen_n is not None else None,
        "tracks_scored": [{"track_id": int(tid), "dist": float(d), "sig_samples": int(n)} for tid, d, n in scored[:10]] if scored else [],
        "segments": [{"start": float(a), "end": float(b), "reason": "track_presence"} for a, b in segs],
        "outputs": {"debug_tracks_csv": f"/data/jobs/{job_id}/debug_tracks.csv"},
    }


def process_job(job_id: str):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    jobs_dir = os.path.join(base, "data", "jobs")
    job_dir = os.path.join(jobs_dir, job_id)

    meta_path = os.path.join(job_dir, "meta.json")
    setup_path = os.path.join(job_dir, "setup.json")
    results_path = os.path.join(job_dir, "results.json")
    clips_dir = os.path.join(job_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    meta = _read_json(meta_path, {})
    setup = _read_json(setup_path, {})

    video_path = meta.get("video_path")
    if not video_path or not os.path.exists(video_path):
        _set_meta(meta_path, "error", {"error": "video_path missing"})
        _write_json(results_path, {"status": "error", "error": "video_path missing"})
        return False

    mode = setup.get("mode", "broadcast_dettrack_reid_lite")

    _set_meta(meta_path, "processing")

    player = setup.get("player") or {}
    seeds = player.get("seeds") or []
    player_id = player.get("player_id", "1")

    if mode == "broadcast_dettrack_reid_lite":
        base_results = _process_broadcast_dettrack_reid_lite(video_path, job_id, job_dir, setup, seeds)
    else:
        base_results = {"status": "error", "error": f"Unknown mode '{mode}'."}

    if base_results.get("status") != "done":
        _set_meta(meta_path, "error", {"error": base_results.get("error", "failed")})
        _write_json(results_path, {"job_id": job_id, **base_results})
        return False

    segments = base_results.get("segments", [])
    clip_entries = []
    clip_paths = []
    for i, seg in enumerate(segments, start=1):
        s = float(seg["start"])
        e = float(seg["end"])
        out_file = f"clip_{i:03d}.mp4"
        out_abs = os.path.join(clips_dir, out_file)
        ok = _ffmpeg_clip(video_path, out_abs, s, e)
        if ok:
            clip_entries.append({"file": out_file, "url": f"/data/jobs/{job_id}/clips/{out_file}", "start": s, "end": e})
            clip_paths.append(out_abs)

    highlight_abs = os.path.join(job_dir, "highlight.mp4")
    highlight_ok = _ffmpeg_concat(clip_paths, highlight_abs)
    highlight_url = f"/data/jobs/{job_id}/highlight.mp4" if highlight_ok else None

    results = {
        "job_id": job_id,
        "status": "done",
        "player_id": player_id,
        "video_path": video_path,
        **base_results,
        "outputs": {
            **(base_results.get("outputs") or {}),
            "clips_dir": f"/data/jobs/{job_id}/clips",
            "clips": clip_entries,
            "clips_count": len(clip_entries),
            "highlight": {"file": "highlight.mp4", "url": highlight_url} if highlight_ok else None,
        }
    }

    _write_json(results_path, results)
    _set_meta(meta_path, "done", {"clips": len(clip_entries), "highlight": bool(highlight_ok), "mode": mode})
    return True
