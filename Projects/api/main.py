cd /workspace/shiftclipper/Projects

cat > api/main.py <<'PY'
import os
import json
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ------------------------
# Paths / constants
# ------------------------
ROOT = Path(__file__).resolve().parents[1]  # Projects/
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"

# Ensure required dirs exist (prevents "Directory does not exist" crash)
DATA_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Helpers
# ------------------------
def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id

def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"

def _load_status(job_id: str) -> Dict[str, Any]:
    p = _status_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(p.read_text())

def _save_status(job_id: str, data: Dict[str, Any]) -> None:
    p = _status_path(job_id)
    p.write_text(json.dumps(data, indent=2))

def _now() -> float:
    return time.time()

def _new_job_id() -> str:
    return os.urandom(6).hex()

def _init_status(job_id: str) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "created",
        "stage": "created",
        "progress": 0,
        "message": "Created.",
        "uploaded_at": None,
        "updated_at": _now(),
        "orig_filename": None,
        "video_path": None,
        "proxy_ready": False,
        "proxy_path": None,
        "proxy_url": None,
        "setup": {
            "camera_mode": "broadcast",
            "player_number": "",
            "jersey_color": "#203524",
            "opponent_color": "#ffffff",
            "extend_sec": 2,
            "verify_mode": False,
            "clicks": [],
            "clicks_count": 0
        },
        "clips": [],
        "combined_path": None,
        "combined_url": None,
        "error": None,
    }

# ------------------------
# API Models
# ------------------------
class SetupPayload(BaseModel):
    camera_mode: str = "broadcast"
    player_number: str = ""
    jersey_color: str = "#203524"
    opponent_color: str = "#ffffff"
    extend_sec: float = 2.0
    verify_mode: bool = False
    clicks: list = []

# ------------------------
# App
# ------------------------
app = FastAPI()

# Static routes
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

@app.get("/", response_class=HTMLResponse)
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "Missing web/index.html. Copy web files into Projects/web/",
            status_code=500
        )
    return HTMLResponse(index_path.read_text())

@app.post("/jobs")
def create_job():
    """
    Create a job. No body required.
    """
    job_id = _new_job_id()
    jd = _job_dir(job_id)
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "clips").mkdir(parents=True, exist_ok=True)

    st = _init_status(job_id)
    _save_status(job_id, st)
    return {"job_id": job_id}

@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    return _load_status(job_id)

@app.post("/jobs/{job_id}/upload")
async def upload_video(job_id: str, file: UploadFile = File(...)):
    jd = _job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    in_path = jd / "in.mp4"
    # Save upload
    with in_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    st = _load_status(job_id)
    st["status"] = "uploaded"
    st["stage"] = "uploaded"
    st["progress"] = 5
    st["message"] = "Uploaded."
    st["uploaded_at"] = _now()
    st["updated_at"] = _now()
    st["orig_filename"] = file.filename
    st["video_path"] = str(in_path)
    st["error"] = None
    _save_status(job_id, st)

    # Worker will generate proxy + clips after /run
    return {"ok": True}

@app.put("/jobs/{job_id}/setup")
def save_setup(job_id: str, payload: SetupPayload):
    st = _load_status(job_id)

    clicks = payload.clicks or []
    st["setup"] = {
        "camera_mode": payload.camera_mode,
        "player_number": payload.player_number,
        "jersey_color": payload.jersey_color,
        "opponent_color": payload.opponent_color,
        "extend_sec": payload.extend_sec,
        "verify_mode": payload.verify_mode,
        "clicks": clicks,
        "clicks_count": len(clicks),
    }
    st["updated_at"] = _now()
    _save_status(job_id, st)
    return {"ok": True}

@app.post("/jobs/{job_id}/run")
def run_job(job_id: str):
    """
    Enqueue the job in Redis/RQ.
    """
    st = _load_status(job_id)
    if not st.get("video_path"):
        raise HTTPException(status_code=400, detail="Missing input video. Upload a video first.")

    # Lazy import so API can boot even if worker deps are still installing
    try:
        import redis
        from rq import Queue
        from worker.tasks import process_job
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Worker/queue not ready: {e}")

    r = redis.Redis.from_url("redis://127.0.0.1:6379")
    q = Queue("jobs", connection=r)

    st["status"] = "queued"
    st["stage"] = "queued"
    st["progress"] = 10
    st["message"] = "Queued for processing…"
    st["updated_at"] = _now()
    _save_status(job_id, st)

    job = q.enqueue(process_job, job_id)
    st = _load_status(job_id)
    st["rq_id"] = job.id
    st["updated_at"] = _now()
    _save_status(job_id, st)

    return {"ok": True, "rq_id": job.id}
PY

