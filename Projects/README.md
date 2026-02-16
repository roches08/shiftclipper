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

Use the existing script:

```bash
cd /workspace/shiftclipper/Projects
chmod +x runpod_start.sh
REDIS_URL=redis://127.0.0.1:6379/0 RQ_QUEUES=jobs JOBS_DIR=/workspace/shiftclipper/Projects/data/jobs bash runpod_start.sh
```

Optional CPU/verify-only mode for reliability:

```bash
export WORKER_VERIFY_ONLY=1
```

Worker self-test command (no GPU needed):

```bash
cd /workspace/shiftclipper/Projects
python -m worker.main --self-test
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

Quick queue verify after posting a job:

```bash
redis-cli llen rq:queue:jobs
```

The length should drop as soon as the worker picks the queued job.

- Confirm worker is running and listening on the same `RQ_QUEUES` queue as API enqueue target.
- Confirm API and worker share the same `JOBS_DIR` mount.
- Use `/jobs/<id>/status` and check `rq_state`; status now derives from RQ (`queued/started/finished/failed`) and merges file status.
- For dependency issues, run worker in verify-only mode (`WORKER_VERIFY_ONLY=1`) to validate queue + status wiring.
- Run smoke test:

```bash
python Projects/scripts/smoke_test.py
```
