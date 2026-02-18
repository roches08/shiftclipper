import os
import json
import uuid
import time
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess
import shutil

import redis
from rq import Queue
from rq.job import Job
try:
    from rq.command import send_stop_job_command
except Exception:  # pragma: no cover
    send_stop_job_command = None
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from common.config import normalize_setup
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
JOBS_DIR = Path(os.getenv("JOBS_DIR", str(DATA_DIR / "jobs"))).resolve()
WEB_DIR = BASE_DIR / "web"

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_NAMES = [q.strip() for q in os.getenv("RQ_QUEUES", "jobs").split(",") if q.strip()]
rconn = redis.from_url(REDIS_URL, decode_responses=True)
q = Queue(QUEUE_NAMES[0], connection=rconn)
print(f"API starting | redis={REDIS_URL} | queues={QUEUE_NAMES} | jobs_dir={JOBS_DIR}")

JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Static JS/CSS
if not WEB_DIR.exists():
    WEB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def meta_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def read_json(path: Path, default: Any) -> Any:
    """Read JSON safely (handles empty/partial files)."""
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write to avoid partial files seen by other threads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


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

    rq_state = None

    # Reflect actual RQ state so the UI never sits "queued" forever
    rq_id = meta.get("rq_id")
    if rq_id:
        try:
            rconn = redis.from_url(REDIS_URL)
            rq_job = Job.fetch(rq_id, connection=rconn)
            rq_state = rq_job.get_status(refresh=True)
            rq_meta = rq_job.meta or {}

            if rq_state == "failed" or rq_job.is_failed:
                exc = (rq_job.exc_info or "").strip()
                msg = "Worker failed."
                if exc:
                    # show last line of stacktrace for quick debugging
                    msg = exc.splitlines()[-1][:500]
                cancelled = meta.get("status") == "cancelled" or "cancelled" in msg.lower()
                meta.update({
                    "rq_state": "failed",
                    "status": "cancelled" if cancelled else "failed",
                    "stage": "cancelled" if cancelled else "failed",
                    "progress": 100,
                    "message": "Job cancelled." if cancelled else msg,
                    "error": None if cancelled else msg,
                    "updated_at": time.time(),
                })
                write_json(mp, meta)
            elif rq_state == "started":
                meta["rq_state"] = "started"
                if meta.get("status") != "cancelled":
                    meta["status"] = "processing"
                if meta.get("status") != "cancelled":
                    if rq_meta.get("stage"):
                        meta["stage"] = rq_meta.get("stage")
                    else:
                        meta["stage"] = meta.get("stage") or "processing"
                    if rq_meta.get("progress") is not None:
                        meta["progress"] = int(rq_meta.get("progress"))
                stage_message = {
                    "uploading": "Uploading...",
                    "queued": "Queued for processing.",
                    "tracking": "Tracking in progress",
                    "clips": "Creating clips",
                    "combined": "Combining video",
                    "done": "Processing complete",
                }
                if meta.get("status") != "cancelled":
                    if rq_meta.get("message"):
                        meta["message"] = str(rq_meta.get("message"))
                    current_message = str(meta.get("message") or "").strip()
                    if not current_message or "queued" in current_message.lower():
                        meta["message"] = stage_message.get(meta.get("stage"), "Processing…")
            elif rq_state == "queued":
                meta["rq_state"] = "queued"
                if meta.get("status") not in {"cancelled", "failed", "done"}:
                    meta["status"] = "queued"
                    meta["stage"] = "queued"
            elif rq_state == "finished":
                meta["rq_state"] = "finished"
                if meta.get("status") not in {"done", "verified", "done_no_clips", "done_no_shifts", "cancelled", "failed"}:
                    meta["status"] = "done"
                    meta["stage"] = "done"
                    meta["progress"] = 100
        except Exception:
            # If Redis/RQ lookup fails, do not break status endpoint.
            pass

    # Convenience flags for the UI
    meta["proxy_ready"] = bool(meta.get("proxy_path") and Path(meta["proxy_path"]).exists())
    if meta.get("proxy_path"):
        meta["proxy_url"] = f"/data/jobs/{job_id}/input_proxy.mp4"
    status_payload = {
        "job_id": job_id,
        "rq_state": rq_state or meta.get("rq_state"),
        "status": meta.get("status"),
        "stage": meta.get("stage"),
        "progress": meta.get("progress", 0),
        "message": meta.get("message"),
        "error": meta.get("error"),
        "updated_at": meta.get("updated_at"),
        "proxy_ready": meta.get("proxy_ready"),
        "proxy_url": meta.get("proxy_url"),
    }
    if meta.get("bytes_received") is not None:
        status_payload["bytes_received"] = meta.get("bytes_received")
    if meta.get("bytes_total") is not None:
        status_payload["bytes_total"] = meta.get("bytes_total")
    return status_payload
@app.get("/data/jobs/{job_id}/{path:path}")
def serve_job_file(job_id: str, path: str):
    full = job_dir(job_id) / path
    if not full.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(full))


@app.post("/jobs/{job_id}/upload")
async def upload_video(job_id: str, request: Request, file: UploadFile = File(...)):
    jd = job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    # Always save to a single predictable filename so the worker/UI agree.
    in_path = str(jd / "in.mp4")
    proxy_path = str(jd / "input_proxy.mp4")

    content_length = request.headers.get("content-length")
    total_bytes = None
    if content_length:
        try:
            total_bytes = max(1, int(content_length))
        except ValueError:
            total_bytes = None

    set_status(
        job_id,
        "uploading",
        stage="uploading",
        progress=1,
        message="Uploading...",
        proxy_ready=False,
        bytes_received=0,
        bytes_total=total_bytes,
    )

    bytes_received = 0
    with open(in_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            bytes_received += len(chunk)
            progress = 5
            if total_bytes:
                progress = min(100, max(1, int((bytes_received / total_bytes) * 100)))
            set_status(
                job_id,
                "uploading",
                stage="uploading",
                progress=progress,
                message="Uploading...",
                bytes_received=bytes_received,
                bytes_total=total_bytes,
            )

    # Persist metadata for the worker (this was missing before)
    meta = read_json(meta_path(job_id), {})
    meta.update({
        "video_path": in_path,
        "orig_filename": file.filename,
        "uploaded_at": time.time(),
    })
    write_json(meta_path(job_id), meta)

    set_status(job_id, "uploaded", extra={
        "video_path": in_path,
        "uploaded_at": time.time(),
        "progress": 10,
        "message": "Upload complete. Building proxy…",
        "stage": "queued",
        "bytes_received": bytes_received,
        "bytes_total": total_bytes,
        "proxy_ready": False,
    })

    ok = make_proxy(in_path, proxy_path, max_h=360, fps=30)
    if not ok:
        set_status(job_id, "error", extra={"progress": 0, "message": "Proxy creation failed (ffmpeg)."})
        raise HTTPException(status_code=500, detail="Proxy creation failed")

    # Store proxy path too (useful for UI)
    meta = read_json(meta_path(job_id), {})
    meta["proxy_path"] = proxy_path
    write_json(meta_path(job_id), meta)

    set_status(job_id, "ready", extra={
        "proxy_path": proxy_path,
        "proxy_ready": True,
        "progress": 18,
        "message": "Proxy ready.",
        "stage": "ready",
    })

    return {"ok": True, "video_path": in_path, "proxy_path": proxy_path}


def _validate_setup_payload(payload: Dict[str, Any]) -> None:
    def _check_01(name: str):
        if name in payload and payload[name] is not None:
            val = float(payload[name])
            if not (0.0 <= val <= 1.0):
                raise HTTPException(status_code=400, detail=f"{name} must be between 0 and 1")

    def _check_non_negative(name: str):
        if name in payload and payload[name] is not None and float(payload[name]) < 0:
            raise HTTPException(status_code=400, detail=f"{name} must be >= 0")

    for nm in [
        "score_lock_threshold", "score_unlock_threshold", "reacquire_score_lock_threshold",
        "seed_iou_min", "seed_dist_max", "ocr_min_conf", "ocr_veto_conf", "swap_guard_bonus",
    ]:
        _check_01(nm)

    for nm in [
        "lost_timeout", "reacquire_window_seconds", "gap_merge_seconds", "lock_seconds_after_confirm",
        "min_track_seconds", "min_clip_seconds", "seed_lock_seconds", "seed_window_s", "ocr_veto_seconds", "swap_guard_seconds",
    ]:
        _check_non_negative(nm)

    if "detect_stride" in payload and int(payload["detect_stride"]) < 1:
        raise HTTPException(status_code=400, detail="detect_stride must be >= 1")
    if "yolo_imgsz" in payload:
        imgsz = int(payload["yolo_imgsz"])
        if imgsz < 256 or imgsz > 1280:
            raise HTTPException(status_code=400, detail="yolo_imgsz must be between 256 and 1280")



@app.put("/jobs/{job_id}/setup")
def setup_job(job_id: str, payload: Dict[str, Any]):
    jd = job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    _validate_setup_payload(payload or {})
    setup = normalize_setup(payload)
    write_json(jd / "setup.json", setup)

    set_status(
        job_id,
        "ready",
        stage="ready",
        progress=25,
        message="Setup saved.",
        setup=setup,
    )
    return {"ok": True, "status": "ready", "job_id": job_id, "setup": setup}


@app.get("/jobs/{job_id}/setup")
def get_setup(job_id: str):
    jd = job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    setup = read_json(jd / "setup.json", {})
    return {"job_id": job_id, "setup": setup}


@app.post("/jobs/{job_id}/run")
def run_job(job_id: str):
    jd = job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    # Make sure we have an input video before we queue work
    meta = read_json(meta_path(job_id), {})
    video_path = meta.get("video_path")
    if not video_path or not Path(video_path).exists():
        # Fallback: accept common filenames if meta.json is missing/old
        for cand in [jd / "in.mp4", jd / "input.mp4", jd / "input.mov", jd / "input.mkv"]:
            if cand.exists():
                video_path = str(cand)
                meta["video_path"] = video_path
                write_json(meta_path(job_id), meta)
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

    setup = read_json(jd / "setup.json", {})
    has_clicks = bool((setup.get("clicks") or []))
    verify_mode = bool(setup.get("verify_mode", False))
    skip_seeding = bool(setup.get("skip_seeding", False))
    if not has_clicks and not verify_mode and not skip_seeding:
        raise HTTPException(status_code=400, detail="At least one player seed click is required unless verify mode or skip seeding is enabled")

    if meta.get("cancel_requested"):
        meta["cancel_requested"] = False
        write_json(meta_path(job_id), meta)

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


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    jd = job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    meta = read_json(meta_path(job_id), {})
    meta["cancel_requested"] = True
    write_json(meta_path(job_id), meta)

    rq_id = meta.get("rq_id")
    if rq_id:
        try:
            rq_job = Job.fetch(rq_id, connection=rconn)
            if rq_job.get_status(refresh=True) == "queued":
                rq_job.cancel()
        except Exception:
            pass

    if rq_id and send_stop_job_command is not None:
        try:
            send_stop_job_command(rconn, rq_id)
        except Exception:
            # Worker may already be idle or not yet started.
            pass

    set_status(
        job_id,
        "cancelled",
        stage="cancelled",
        progress=100,
        message="Job cancelled.",
    )
    return {"ok": True, "job_id": job_id, "status": "cancelled"}


@app.delete("/jobs/{job_id}")
def clear_job(job_id: str):
    jd = job_dir(job_id)
    mp = meta_path(job_id)
    if not jd.exists() and not mp.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    meta = read_json(mp, {})
    rq_id = meta.get("rq_id")

    # Always request cancellation first so running jobs terminate before deletion.
    try:
        cancel_job(job_id)
    except Exception:
        pass

    if rq_id:
        try:
            rq_job = Job.fetch(rq_id, connection=rconn)
            if rq_job.get_status(refresh=True) in {"queued", "started"} and send_stop_job_command is not None:
                send_stop_job_command(rconn, rq_id)
            rq_job.delete()
        except Exception:
            pass

    shutil.rmtree(jd, ignore_errors=True)
    return {"ok": True, "job_id": job_id, "deleted": True}


@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: str):
    jd = job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    meta = read_json(meta_path(job_id), {})
    if meta.get("status") not in {"failed", "cancelled"}:
        raise HTTPException(status_code=400, detail="Only failed/cancelled jobs can be retried")

    from worker.tasks import process_job
    rq_job = q.enqueue(process_job, job_id, job_timeout=3600)
    set_status(
        job_id,
        "queued",
        stage="queued",
        progress=30,
        message="Retried and queued for processing.",
        error=None,
        cancel_requested=False,
        rq_id=rq_job.get_id(),
    )
    return {"ok": True, "job_id": job_id, "rq_id": rq_job.get_id(), "status": "queued"}


@app.post("/jobs/cleanup")
def cleanup_jobs(days: int = 7, max_count: int = 200):
    if max_count < 1:
        raise HTTPException(status_code=400, detail="max_count must be >= 1")
    cutoff = time.time() - (max(0, days) * 86400)
    entries = []
    for p in JOBS_DIR.iterdir():
        if p.is_dir():
            entries.append((p, p.stat().st_mtime))
    entries.sort(key=lambda x: x[1], reverse=True)

    removed = []
    for idx, (path, mtime) in enumerate(entries):
        if idx >= max_count or mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)

    return {"ok": True, "removed": removed, "kept": max(0, len(entries) - len(removed))}


@app.get("/jobs/{job_id}/results")
def results(job_id: str):
    jd = job_dir(job_id)
    results_path = jd / "results.json"
    meta = read_json(meta_path(job_id), {})

    if results_path.exists():
        return JSONResponse(read_json(results_path, {}))

    # Fallback for workers that persist final data in job.json only.
    if meta.get("status") in {"done", "verified", "done_no_clips", "failed"}:
        return JSONResponse(meta)

    raise HTTPException(status_code=404, detail="No results yet")
