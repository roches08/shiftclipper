import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from rq import Queue
from redis import Redis

APP_ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
JOBS_DIR = APP_ROOT / "data" / "jobs"

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
redis_conn = Redis.from_url(REDIS_URL)
q = Queue("jobs", connection=redis_conn)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static web UI
WEB_DIR = APP_ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# serve generated data
DATA_DIR = APP_ROOT / "data"
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "meta.json"


def _set_status(job_id: str, **fields: Any) -> Dict[str, Any]:
    mp = _meta_path(job_id)
    meta = _read_json(mp, {})
    meta.update(fields)
    meta["job_id"] = job_id
    meta["updated_at"] = time.time()
    _write_json(mp, meta)
    return meta


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return "<h1>ShiftClipper</h1><p>Missing web/index.html</p>"
    return index_path.read_text(encoding="utf-8")


@app.post("/jobs")
def create_job():
    job_id = os.urandom(6).hex()
    jd = _job_dir(job_id)
    jd.mkdir(parents=True, exist_ok=True)
    _set_status(job_id, status="created", stage="created", progress=0, message="Job created.")
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    mp = _meta_path(job_id)
    if not mp.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return _read_json(mp, {})


@app.post("/jobs/{job_id}/upload")
async def upload_video(job_id: str, file: UploadFile = File(...)):
    jd = _job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    in_path = jd / "in.mp4"
    orig_filename = file.filename or "upload"
    content = await file.read()
    in_path.write_bytes(content)

    # reset status after upload
    _set_status(
        job_id,
        status="uploaded",
        stage="uploaded",
        progress=5,
        message="Uploaded.",
        orig_filename=orig_filename,
        video_path=str(in_path),
        uploaded_at=time.time(),
    )
    return {"ok": True, "job_id": job_id, "video_path": str(in_path)}


@app.put("/jobs/{job_id}/setup")
async def setup_job(job_id: str, payload: Dict[str, Any]):
    jd = _job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    meta = _set_status(job_id, setup=payload, status="ready", stage="ready", progress=10, message="Setup saved.")
    return {"ok": True, "job_id": job_id, "setup": meta.get("setup", {})}


@app.post("/jobs/{job_id}/run")
def run_job(job_id: str):
    """
    IMPORTANT FIX:
    - Do NOT enqueue again if already queued/processing/done.
    - If previous attempt failed, allow re-run (new rq_id) but only once per click.
    """
    mp = _meta_path(job_id)
    if not mp.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    meta = _read_json(mp, {})
    status = (meta.get("status") or "").lower()
    stage = (meta.get("stage") or "").lower()

    # already running or done -> don't enqueue again
    if status in {"queued", "processing"} or stage in {"queued", "processing"}:
        return {"ok": True, "job_id": job_id, "status": meta.get("status"), "rq_id": meta.get("rq_id")}

    if status == "done":
        return {"ok": True, "job_id": job_id, "status": "done", "rq_id": meta.get("rq_id")}

    # enqueue
    _set_status(job_id, status="queued", stage="queued", progress=12, message="Queued for processing...")
    from worker.tasks import process_job  # local import for worker

    rq_job = q.enqueue(process_job, job_id, job_timeout=60 * 60 * 6)  # up to 6 hours
    _set_status(job_id, rq_id=rq_job.get_id())

    return {"ok": True, "job_id": job_id, "status": "queued", "rq_id": rq_job.get_id()}

