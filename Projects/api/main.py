import os
import json
import uuid
import time
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess

import redis
from rq import Queue
from rq.job import Job
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

"""
ShiftClipper API (RunPod friendly)
- No hardcoded /workspace/Projects assumptions.
- Repo root inferred from this file location: Projects/
- Web UI: GET /
- Static: /static/app.js
- Job data served from /data/jobs/...
"""

BASE_DIR = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1])).resolve()  # Projects/
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
WEB_DIR = BASE_DIR / "web"

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
rconn = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
q = Queue("jobs", connection=rconn)

JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Static JS/CSS
if not WEB_DIR.exists():
    WEB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def meta_path(job_id: str) -> Path:
    return job_dir(job_id) / "meta.json"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def set_status(
    job_id: str,
    status: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    *,
    stage: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    **more: Any,
) -> None:
    """
    Flexible status updater.

    Supports BOTH calling styles:
      set_status(job_id, "created", {"name": "job"})
    and:
      set_status(job_id, "ready", progress=25, message="Setup saved.")
    """
    mp = meta_path(job_id)
    meta = read_json(mp, {})
    meta["job_id"] = job_id

    if status is not None:
        meta["status"] = status
    if stage is not None:
        meta["stage"] = stage
    if progress is not None:
        try:
            meta["progress"] = int(progress)
        except Exception:
            meta["progress"] = progress
    if message is not None:
        meta["message"] = message
    if error is not None:
        meta["error"] = error

    if extra:
        meta.update(extra)
    if more:
        meta.update(more)

    meta["updated_at"] = time.time()
    write_json(mp, meta)


def make_proxy(in_path: str, out_path: str, max_h: int = 360, fps: int = 30) -> bool:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-vf", f"scale=-2:min({max_h}\\,ih)",
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=False)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 64 * 1024


@app.get("/api/health")
def health():
    return {"status": "ok", "root": str(BASE_DIR)}


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>ShiftClipper</h1><p>Missing Projects/web/index.html</p>", status_code=500)
    return FileResponse(str(index_path), media_type="text/html")


@app.post("/jobs")
def create_job(payload: Dict[str, Any] | None = None):
    job_id = uuid.uuid4().hex[:12]
    jd = job_dir(job_id)
    jd.mkdir(parents=True, exist_ok=True)
    name = (payload or {}).get("name", "job")
    set_status(job_id, "created", {"name": name, "created_at": time.time(), "progress": 0, "stage": "created"})
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    mp = meta_path(job_id)
    if not mp.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    meta = read_json(mp, {})

    rq_id = meta.get("rq_id")
    if rq_id:
        try:
            rq_job = Job.fetch(rq_id, connection=rconn)
            if rq_job.is_failed:
                meta["status"] = "error"
                meta["stage"] = "error"
                meta["progress"] = 0
                if rq_job.exc_info:
                    meta["error"] = rq_job.exc_info.splitlines()[-1]
            elif rq_job.is_finished:
                meta["status"] = meta.get("status") or "done"
                meta["stage"] = meta.get("stage") or "done"
                meta["progress"] = int(meta.get("progress", 100))
            else:
                rq_meta = rq_job.meta or {}
                if "progress" in rq_meta:
                    meta["progress"] = int(rq_meta.get("progress", meta.get("progress", 0)))
                if "stage" in rq_meta:
                    meta["stage"] = rq_meta.get("stage")
                if meta.get("status") not in {"done", "error"}:
                    meta["status"] = "processing"
        except Exception:
            pass

    # Convenience fields UI expects
    proxy_path = meta.get("proxy_path")
    if proxy_path and os.path.exists(proxy_path):
        meta["proxy_ready"] = True
        meta["proxy_url"] = f"/data/jobs/{job_id}/input_proxy.mp4"
    return meta


@app.get("/data/jobs/{job_id}/{path:path}")
def serve_job_file(job_id: str, path: str):
    full = job_dir(job_id) / path
    if not full.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(full))


@app.post("/jobs/{job_id}/upload")
async def upload_video(job_id: str, file: UploadFile = File(...)):
    jd = job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    # Always save to a single predictable filename so the worker/UI agree.
    in_path = str(jd / "in.mp4")
    proxy_path = str(jd / "input_proxy.mp4")

    set_status(job_id, "uploading", extra={"progress": 5, "message": "Uploading…", "proxy_ready": False})

    with open(in_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    # Persist metadata for the worker (this was missing before)
    write_json(jd / "meta.json", {
        "video_path": in_path,
        "orig_filename": file.filename,
        "uploaded_at": time.time(),
    })

    set_status(job_id, "uploaded", extra={
        "video_path": in_path,
        "uploaded_at": time.time(),
        "progress": 10,
        "message": "Upload complete. Building proxy…",
        "proxy_ready": False,
    })

    ok = make_proxy(in_path, proxy_path, max_h=360, fps=30)
    if not ok:
        set_status(job_id, "error", extra={"progress": 0, "message": "Proxy creation failed (ffmpeg)."})
        raise HTTPException(status_code=500, detail="Proxy creation failed")

    # Store proxy path too (useful for UI)
    meta = read_json(jd / "meta.json", {})
    meta["proxy_path"] = proxy_path
    write_json(jd / "meta.json", meta)

    set_status(job_id, "ready", extra={
        "proxy_path": proxy_path,
        "proxy_ready": True,
        "progress": 18,
        "message": "Proxy ready.",
        "stage": "ready",
    })

    return {"ok": True, "video_path": in_path, "proxy_path": proxy_path}


@app.put("/jobs/{job_id}/setup")
def setup_job(job_id: str, payload: Dict[str, Any]):
    jd = job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    write_json(jd / "setup.json", payload)

    set_status(
        job_id,
        "ready",
        stage="ready",
        progress=25,
        message="Setup saved.",
        setup=payload,
    )
    return {"ok": True, "status": "ready", "job_id": job_id}


@app.post("/jobs/{job_id}/run")
def run_job(job_id: str):
    jd = job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    # Make sure we have an input video before we queue work
    meta = read_json(jd / "meta.json", {})
    video_path = meta.get("video_path")
    if not video_path or not Path(video_path).exists():
        # Fallback: accept common filenames if meta.json is missing/old
        for cand in [jd / "in.mp4", jd / "input.mp4", jd / "input.mov", jd / "input.mkv"]:
            if cand.exists():
                video_path = str(cand)
                meta["video_path"] = video_path
                write_json(jd / "meta.json", meta)
                break

    if not video_path or not Path(video_path).exists():
        set_status(
            job_id,
            "error",
            stage="error",
            progress=0,
            message="Missing input video (in.mp4). Upload a video first.",
        )
        raise HTTPException(status_code=400, detail="Missing input video")

    # Long videos + detection can exceed the default RQ timeout (often 180s).
    from worker.tasks import process_job
    rq_job = q.enqueue(process_job, job_id, job_timeout=3600)

    set_status(
        job_id,
        "queued",
        stage="queued",
        progress=30,
        message="Queued for processing.",
        rq_id=rq_job.get_id(),
    )
    return {"rq_id": rq_job.get_id(), "job_id": job_id}


@app.get("/jobs/{job_id}/results")
def results(job_id: str):
    rp = job_dir(job_id) / "results.json"
    if not rp.exists():
        raise HTTPException(status_code=404, detail="No results yet")
    return JSONResponse(read_json(rp, {}))
