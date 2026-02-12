import os
import json
import uuid
import time
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess

import redis
from rq import Queue
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

"""API for ShiftClipper MVP.

Key goals:
- Zero hardcoded paths. Works whether the repo lives at /workspace/Projects or /workspace.
- Web UI loads from / and JS/CSS from /static/*
- Redis defaults to local (127.0.0.1) which matches runpod_start.sh
"""

# Repo root can be overridden, otherwise infer relative to this file.
BASE_DIR = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
WEB_DIR = BASE_DIR / "web"

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
rconn = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
q = Queue("jobs", connection=rconn)

JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Serve the web UI.
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    # Keep this explicit so we never depend on StaticFiles(html=True) routing.
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>ShiftClipper</h1><p>Missing web/index.html</p>", status_code=500)
    return FileResponse(str(index_path), media_type="text/html")

def job_dir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, job_id)

def meta_path(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "meta.json")

def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def set_status(job_id: str, status: str, extra: Optional[Dict[str, Any]] = None, **fields: Any) -> None:
    """Update meta.json for a job.

    Backwards compatible:
      - older code passes a dict via `extra=...`
      - newer code may pass keyword fields like progress=..., message=...
    """
    meta = read_json(meta_path(job_id), {})
    meta["job_id"] = job_id
    meta["status"] = status
    meta["updated_at"] = time.time()
    if extra:
        meta.update(extra)
    if fields:
        meta.update(fields)
    write_json(meta_path(job_id), meta)


def make_proxy(in_path: str, out_path: str, max_h: int = 360, fps: int = 30) -> bool:
    """Create a fast-start MP4 proxy for browser playback.

    Returns True if the output file exists and is non-trivial in size.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # IMPORTANT:
    #   Do NOT use commas inside scale expressions here (e.g. min(360,ih)).
    #   Those have repeatedly caused ffmpeg expression parse failures in the pod.
    #   Use force_original_aspect_ratio=decrease instead.
    # Also: ensure even dimensions (H.264 requirement) via a small pad.
    vf = (
        f"scale=-2:{max_h}:force_original_aspect_ratio=decrease,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", in_path,
        "-map", "0:v:0", "-map", "0:a?",  # audio is optional
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-movflags", "+faststart",
        out_path,
    ]

    # Capture ffmpeg output for debugging (proxy failures were silent before).
    log_path = os.path.join(os.path.dirname(out_path), "proxy_ffmpeg.log")
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(p.stdout or "")
    except Exception as e:
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"Exception running ffmpeg: {e}\n")
        except Exception:
            pass
    try:
        return os.path.exists(out_path) and os.path.getsize(out_path) > 64 * 1024
    except Exception:
        return False

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/jobs")
def create_job(payload: Dict[str, Any] | None = None):
    job_id = uuid.uuid4().hex[:12]
    os.makedirs(job_dir(job_id), exist_ok=True)
    name = (payload or {}).get("name", "job")
    set_status(job_id, "created", {"name": name, "created_at": time.time()})
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    mp = meta_path(job_id)
    if not os.path.exists(mp):
        raise HTTPException(status_code=404, detail="Job not found")
    meta = read_json(mp, {})
    # Convenience fields the UI relies on
    proxy_path = meta.get("proxy_path")
    if proxy_path and os.path.exists(proxy_path):
        meta["proxy_ready"] = True
        meta["proxy_url"] = f"/data/jobs/{job_id}/input_proxy.mp4"
    return meta


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    # UI convenience endpoint
    return get_job(job_id)


@app.get("/data/jobs/{job_id}/{path:path}")
def serve_job_file(job_id: str, path: str):
    full = os.path.join(job_dir(job_id), path)
    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(full)


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    # Alias for compatibility with the web UI
    return get_job(job_id)

@app.post("/jobs/{job_id}/upload")
async def upload_video(job_id: str, file: UploadFile = File(...)):
    if not os.path.exists(job_dir(job_id)):
        raise HTTPException(status_code=404, detail="Job not found")
    ext = os.path.splitext(file.filename or "")[1] or ".mp4"
    in_path = os.path.join(job_dir(job_id), f"input{ext}")
    proxy_path = os.path.join(job_dir(job_id), "input_proxy.mp4")

    set_status(job_id, "uploading", {"progress": 5, "message": "Uploading…"})

    # Stream upload to disk (don't hold big videos in RAM)
    with open(in_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    set_status(job_id, "uploaded", {
        "video_path": in_path,
        "uploaded_at": time.time(),
        "progress": 10,
        "message": "Upload complete. Building proxy…",
        "proxy_ready": False,
    })

    # Build a small proxy mp4 for fast browser playback
    ok = make_proxy(in_path, proxy_path, max_h=360)
    if ok:
        set_status(job_id, "ready", {
            "proxy_path": proxy_path,
            "proxy_ready": True,
            "progress": 18,
            "message": "Proxy ready.",
        })
    else:
        set_status(job_id, "error", {
            "progress": 0,
            "message": "Proxy creation failed (ffmpeg error).",
        })
        raise HTTPException(status_code=500, detail="Proxy creation failed")

    return {"ok": True, "video_path": in_path, "proxy_path": proxy_path}

@app.put("/jobs/{job_id}/setup")
def setup_job(job_id: str, payload: Dict[str, Any]):
    if not os.path.exists(job_dir(job_id)):
        raise HTTPException(status_code=404, detail="Job not found")
    write_json(os.path.join(job_dir(job_id), "setup.json"), payload)
    set_status(job_id, "ready", progress=25, message="Setup saved.")
    return {"ok": True}

@app.post("/jobs/{job_id}/run")
def run_job(job_id: str):
    if not os.path.exists(job_dir(job_id)):
        raise HTTPException(status_code=404, detail="Job not found")
    rq_job = q.enqueue("worker.tasks.process_job", job_id)
    set_status(job_id, "queued", {"rq_id": rq_job.get_id()})
    return {"rq_id": rq_job.get_id()}

@app.get("/jobs/{job_id}/results")
def results(job_id: str):
    rp = os.path.join(job_dir(job_id), "results.json")
    if not os.path.exists(rp):
        raise HTTPException(status_code=404, detail="No results yet")
    return JSONResponse(read_json(rp, {}))

# IMPORTANT: mount web LAST so API routes work
