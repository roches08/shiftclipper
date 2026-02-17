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

### Broadcast vs Tactical
- **broadcast**: use when player is smaller/farther away and camera moves often. Defaults prioritize recall: `detect_stride=1`, lower `ocr_min_conf`, longer lock window.
- **tactical**: use when player is larger and spacing is stable. Defaults prioritize precision/perf: `detect_stride=3`, higher `ocr_min_conf`, tighter color tolerance.

### Recommended starting defaults
- `detect_stride`: broadcast `1`, tactical `3`
- `ocr_min_conf`: broadcast `0.28`, tactical `0.42`
- `min_track_seconds`: `1.0-1.2`
- `gap_merge_seconds`: `0.8-1.0`
- `lock_seconds_after_confirm`: `1.0-1.5`
- `jersey_color_tolerance`: `70-95`

### Advanced tracker behavior
1. OCR confirms identity when target number appears enough times in the recent window.
2. Confirmed identity remains **locked** for `lock_seconds_after_confirm` even if OCR drops.
3. During lock, tracker uses motion continuity + color gating to avoid drift.
4. Track loss beyond `lost_timeout_seconds` closes the segment.
5. Segments are merged with `gap_merge_seconds`, then filtered by `min_track_seconds`.

### Debug overlay + timeline
- Enable `debug_overlay` to save `debug_overlay.mp4` with:
  - bbox + track id
  - OCR text/confidence
  - jersey color score
  - state (`searching`, `confirmed`, `locked`)
- `debug.json` provides timeline/state transitions, merge events, raw/merged segments.
- If no output clips are created in run mode, the job fails with a clear message pointing to debug artifacts.

### GPU device selection
- Resolver order:
  1. `SHIFTCLIPPER_DEVICE` env override
  2. `cuda:0` if `torch.cuda.is_available()`
  3. fallback `cpu`
- Worker startup logs include chosen device, torch version, CUDA availability, and GPU name.
- `runpod_start.sh` defaults `SHIFTCLIPPER_DEVICE=cuda:0` and installs CUDA torch wheels when running on GPU pods.
