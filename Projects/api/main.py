from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from redis import Redis
from rq import Queue

# Project dirs
BASE_DIR = Path(__file__).resolve().parents[1]  # .../Projects
WEB_DIR = BASE_DIR / "web"
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"

# ✅ Ensure required directories exist BEFORE mounting StaticFiles
WEB_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
queue = Queue("jobs", connection=redis_conn)

app = FastAPI()

# Static mounts
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


class CreateJobReq(BaseModel):
    camera_mode: str = "broadcast"


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _read_status(job_id: str) -> Dict[str, Any]:
    p = _job_dir(job_id) / "status.json"
    if not p.exists():
        return {"job_id": job_id, "status": "new", "progress": 0, "message": "New job."}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"job_id": job_id, "status": "error", "progress": 0, "message": "Corrupt status.json"}


def _write_status(job_id: str, payload: Dict[str, Any]) -> None:
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "status.json"
    payload.setdefault("job_id", job_id)
    payload["updated_at"] = time.time()
    p.write_text(json.dumps(payload, indent=2))


@app.get("/", response_class=HTMLResponse)
def root() -> Any:
    index = WEB_DIR / "index.html"
    if not index.exists():
        return HTMLResponse(
            "<h3>Missing web/index.html</h3>"
            "<p>Expected at /workspace/shiftclipper/Projects/web/index.html</p>",
            status_code=500,
        )
    return FileResponse(index)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "web_dir": str(WEB_DIR),
        "data_dir": str(DATA_DIR),
        "jobs_dir": str(JOBS_DIR),
        "redis": REDIS_URL,
    }


@app.post("/jobs")
def create_job(req: CreateJobReq) -> Dict[str, Any]:
    job_id = os.urandom(6).hex()
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    _write_status(
        job_id,
        {
            "status": "created",
            "progress": 0,
            "message": "Job created.",
            "camera_mode": req.camera_mode,
        },
    )
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/status")
def get_status(job_id: str) -> Dict[str, Any]:
    return _read_status(job_id)


@app.post("/jobs/{job_id}/upload")
async def upload(job_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    d = _job_dir(job_id)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Unknown job_id")

    in_path = d / "in.mp4"
    with in_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    _write_status(
        job_id,
        {
            "status": "uploaded",
            "progress": 5,
            "message": "Uploaded.",
            "orig_filename": file.filename,
            "video_path": str(in_path),
        },
    )
    return {"ok": True}


class SetupReq(BaseModel):
    player_number: str = ""
    jersey_color: str = "#203524"
    opponent_color: str = "#ffffff"
    extend_sec: float = 2.0
    verify_mode: bool = False
    clicks: list[dict] = []


@app.put("/jobs/{job_id}/setup")
def set_setup(job_id: str, req: SetupReq) -> Dict[str, Any]:
    d = _job_dir(job_id)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Unknown job_id")

    setup_path = d / "setup.json"
    setup_path.write_text(req.model_dump_json(indent=2))

    st = _read_status(job_id)
    st["setup"] = req.model_dump()
    st["clicks_count"] = len(req.clicks)
    st["status"] = "ready"
    st["progress"] = max(int(st.get("progress", 0)), 10)
    st["message"] = "Setup saved."
    _write_status(job_id, st)

    return {"ok": True}


@app.post("/jobs/{job_id}/run")
def run_job(job_id: str) -> Dict[str, Any]:
    d = _job_dir(job_id)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Unknown job_id")

    in_path = d / "in.mp4"
    if not in_path.exists() or in_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Upload a video first.")

    setup_path = d / "setup.json"
    if not setup_path.exists():
        raise HTTPException(status_code=400, detail="Save setup first.")

    rq_job = queue.enqueue("worker.tasks.process_job", job_id, job_timeout=3600)
    st = _read_status(job_id)
    st.update(
        {
            "status": "queued",
            "progress": 15,
            "message": "Queued for processing.",
            "rq_id": rq_job.id,
        }
    )
    _write_status(job_id, st)
    return {"ok": True, "rq_id": rq_job.id}

