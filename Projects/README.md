# ShiftClipper Tracker

## Tracking + Queue Audit (pre-change)

- **Single source of truth (before changes):** `Projects/data/jobs/<job_id>/job.json` was the practical source of truth for UI state; RQ state was only partially consulted in `/jobs/{job_id}/status`. API wrote status fields in multiple endpoints and worker also wrote them directly. This created drift risk.  
- **Redis config:** `REDIS_URL` env var in API and worker, default `redis://127.0.0.1:6379/0`.  
- **Queue names:** API enqueued to hardcoded `jobs`; worker listened on `RQ_QUEUES` (default `jobs`). Potential mismatch if env changed for worker only.  
- **Artifacts path:** `JOBS_DIR` env var defaulted to `Projects/data/jobs` in both API and worker. Shared only if both containers mount same host volume.  
- **Status fields:** `status`, `stage`, `progress`, `message`, `error`, `updated_at` in `job.json`; worker also updated RQ `job.meta` (`stage`/`progress`) while running. Reads were split across UI polling + `/status` + `/results`.  
- **Mismatches found:**
  - `/upload` overwrote `job.json`, dropping prior fields (including setup/status bits).
  - API queue name was fixed (`jobs`) while worker used `RQ_QUEUES`.
  - `/status` did not return stable contract and only handled failed RQ state explicitly.
  - Cancellation marked `cancelled` immediately even if queued/started race existed.

## Local setup (Docker Compose)

```bash
cd /workspace/shiftclipper
docker compose up --build
```

Services:
- `redis` on `redis://redis:6379/0`
- `api` on `http://127.0.0.1:8000`
- `worker` consuming `RQ_QUEUES` (default `jobs`)

Both API and worker print on startup:
- Redis endpoint (including DB index)
- Queue name(s)
- Job artifacts dir

## Environment variables

- `REDIS_URL` (default `redis://127.0.0.1:6379/0`)
- `RQ_QUEUES` (default `jobs`)
- `JOBS_DIR` (default `Projects/data/jobs` locally, `/app/data/jobs` in containers)
- `WORKER_VERIFY_ONLY` (`1` to run verify-only queue/status wiring mode without cv2-heavy tracking)

## RunPod startup

Use the startup script (idempotent order: Redis -> venv/deps -> API -> worker):

```bash
cd /workspace/shiftclipper/Projects
chmod +x runpod_start.sh
REDIS_URL=redis://localhost:6379/0 JOBS_DIR=/workspace/shiftclipper/Projects/data/jobs bash runpod_start.sh
```

The script will:
- enforce `set -euo pipefail`
- default `REDIS_URL` to `redis://localhost:6379/0`
- start Redis only when needed
- create/use `.venv` and install pinned deps from `requirements.txt`
- start API on `0.0.0.0:8000` and wait for readiness
- start worker with `rq worker -u "$REDIS_URL" jobs`
- print RQ version, Redis ping, queue length, and API/worker PIDs

### One-command verification

Run this single command after startup. It creates a job, uploads a tiny generated video, queues the run, then asserts queue activity + terminal status:

```bash
python -c 'import os,subprocess,tempfile,time,requests,redis; b="http://127.0.0.1:8000"; r=redis.from_url(os.getenv("REDIS_URL","redis://localhost:6379/0")); jid=requests.post(f"{b}/jobs",json={"name":"verify"},timeout=10).json()["job_id"]; tmp=tempfile.NamedTemporaryFile(suffix=".mp4",delete=False).name; subprocess.run(["ffmpeg","-y","-f","lavfi","-i","color=c=black:s=320x240:d=1","-c:v","libx264","-pix_fmt","yuv420p",tmp],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); requests.post(f"{b}/jobs/{jid}/upload",files={"file":("verify.mp4",open(tmp,"rb"),"video/mp4")},timeout=60).raise_for_status(); requests.put(f"{b}/jobs/{jid}/setup",json={"camera_mode":"broadcast"},timeout=10).raise_for_status(); requests.post(f"{b}/jobs/{jid}/run",timeout=10).raise_for_status(); saw_queued=False; saw_started=False; status=None; t0=time.time();
while time.time()-t0<180:
 qlen=r.llen("rq:queue:jobs"); s=requests.get(f"{b}/jobs/{jid}/status",timeout=10).json(); st=(s.get("status") or "").lower(); saw_queued=saw_queued or qlen>0 or st=="queued"; saw_started=saw_started or s.get("rq_state")=="started" or st in {"processing","running"};
 if st in {"done","failed","cancelled","error"}: status=st; break; time.sleep(1)
assert saw_queued, "job never reached queue"; assert saw_started, "job never started"; assert status in {"done","failed","cancelled","error"}, f"non-terminal status: {status}"; assert r.llen("rq:queue:jobs")==0, "jobs queue did not drain"; print({"job_id":jid,"final_status":status})'
```

## API examples

```bash
curl -s -X POST http://127.0.0.1:8000/jobs
curl -s http://127.0.0.1:8000/jobs/<job_id>/status
curl -s http://127.0.0.1:8000/jobs/<job_id>/results
curl -s -X POST http://127.0.0.1:8000/jobs/<job_id>/cancel
curl -s -X POST http://127.0.0.1:8000/jobs/<job_id>/retry
curl -s -X POST "http://127.0.0.1:8000/jobs/cleanup?days=7&max_count=100"
```

## Troubleshooting: “queued forever”

- Confirm worker is running and listening on the same `RQ_QUEUES` queue as API enqueue target.
- Confirm API and worker share the same `JOBS_DIR` mount.
- Use `/jobs/<id>/status` and check `rq_state`; status now derives from RQ (`queued/started/finished/failed`) and merges file status.
- For dependency issues, run worker in verify-only mode (`WORKER_VERIFY_ONLY=1`) to validate queue + status wiring.
- Run smoke test:

```bash
python Projects/scripts/smoke_test.py
```

## Tracker v2 tuning guide

### Camera mode selection
- **Broadcast**: standard TV side angle; balanced OCR + motion tracking.
- **Broadcast (Wide / Youth)**: wider view and smaller jersey numbers; use stronger color persistence and longer lock window.
- **Tactical**: overhead/high-wide where OCR is often weak; rely on motion + color.

### Tracking mode
- **clip**: creates highlight clips from LOCKED windows.
- **shift**: runs OFF_ICE -> ON_ICE -> EXITING -> OFF_ICE state machine and reports shifts + total TOI.

### OCR/persistence tuning
- Lower `ocr_min_conf` to catch angled/small numbers.
- Increase `lock_seconds_after_confirm` for frequent occlusion/turns.
- Increase `lost_timeout` if player disappears behind traffic.
- Increase `gap_merge_seconds` to stitch short dropouts.
- Increase `color_tolerance` in poor lighting; reduce it when drift occurs.

### Debug interpretation
- Enable `debug_overlay` to inspect bbox, OCR conf, color score, and SEARCHING/CONFIRMED/LOCKED state.
- Enable `debug_timeline` to inspect state transitions, OCR hits, merges, shift boundaries, and end reasons.
