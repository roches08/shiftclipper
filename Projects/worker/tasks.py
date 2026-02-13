import os
import json
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

"""
Worker: click-driven clipper (reliable prototype)

This version is intentionally conservative:
- Uses the user's click times to build clip "spans"
- Cuts clips with ffmpeg (re-encode for maximum container compatibility)
- Produces:
    clips/clip_###.mp4
    combined.mp4 (all clips concatenated)
    results.json
- NO "No player selected" failure if clicks exist.
"""

BASE_DIR = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
JOBS_DIR = BASE_DIR / "data" / "jobs"

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
    # If stream-copy fails (codec/timebase mismatches), fall back to re-encode.
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

def _build_spans(click_times: List[float], gap_s: float = 12.0, pad_s: float = 2.0) -> List[Tuple[float, float]]:
    """Group clicks into spans. If clicks are far apart, we make multiple clips."""
    t = sorted(float(x) for x in click_times)
    if not t:
        return []
    spans: List[Tuple[float, float]] = []
    cur_start = t[0]
    cur_end = t[0]
    for tt in t[1:]:
        if (tt - cur_end) <= gap_s:
            cur_end = tt
        else:
            spans.append((max(0.0, cur_start - pad_s), cur_end + pad_s))
            cur_start = tt
            cur_end = tt
    spans.append((max(0.0, cur_start - pad_s), cur_end + pad_s))
    # Merge tiny spans
    merged: List[Tuple[float, float]] = []
    for s,e in spans:
        if not merged:
            merged.append((s,e))
        else:
            ps,pe = merged[-1]
            if s - pe <= 0.5:
                merged[-1] = (ps, max(pe,e))
            else:
                merged.append((s,e))
    return merged

def process_job(job_id: str) -> Dict[str, Any]:
    jd = _job_dir(job_id)
    setup_path = jd / "setup.json"
    meta_path = jd / "meta.json"

    setup = _read_json(setup_path, {})
    meta = _read_json(meta_path, {})

    video_path = meta.get("video_path")
    if not video_path or not os.path.exists(video_path):
        _set_status(job_id, status="error", stage="error", progress=0, error="Missing video. Upload first.")
        return {"status": "error", "error": "Missing video"}

    # Accept either clicks or seeds (older UI)
    clicks = setup.get("clicks") or setup.get("seeds") or []
    click_times = [c.get("t") for c in clicks if isinstance(c, dict) and c.get("t") is not None]
    if len(click_times) < 1:
        _set_status(job_id, status="error", stage="error", progress=0, error="No clicks found. Use Select Player and click the player.")
        return {"status": "error", "error": "No clicks"}

    spans = _build_spans(click_times, gap_s=12.0, pad_s=2.0)
    _set_status(job_id, status="processing", stage="processing", progress=60, message=f"Cutting {len(spans)} clip(s)…")

    clips_dir = jd / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    for i,(s,e) in enumerate(spans, start=1):
        out_path = str(clips_dir / f"clip_{i:03d}.mp4")
        _ffmpeg_cut(video_path, out_path, s, e)
        clips.append({
            "start": s,
            "end": e,
            "path": out_path,
            "url": f"/data/jobs/{job_id}/clips/clip_{i:03d}.mp4",
        })

    # combined mp4
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
        "player_number": setup.get("player_number", ""),
        "jersey_color": setup.get("jersey_color", ""),
        "clicks_count": len(clicks),
        "clicks": clicks,
        "clips": clips,
        "combined_path": combined_path,
        "combined_url": f"/data/jobs/{job_id}/combined.mp4",
        "debug": {
            "spans": {"count_spans": len(spans), "count_times": len(click_times)},
            "spans_raw": spans,
        },
    }

    _write_json(jd / "results.json", results)
    _set_status(job_id, status="done", stage="done", progress=100, message="Done.", clips=clips, combined_path=combined_path, combined_url=results["combined_url"])
    return results
