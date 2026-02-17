# ShiftClipper Operational Audit (pre-upgrade)

## 1) Job state truth source (RQ vs `status.json` vs meta files)
- There is **no `status.json`** file used in the current code path.
- Primary persisted state is `Projects/data/jobs/<job_id>/job.json` (API helper `meta_path()` and worker writes). 
- Runtime queue state comes from RQ (`Job.fetch(...).get_status()`), and `/jobs/{job_id}/status` merges that with `job.json`.
- This means effective truth is **split**:
  - RQ is source for transient execution state (`queued/started/finished/failed`).
  - `job.json` is source for durable job metadata and user-facing status fields.

## 2) Redis config used by API + worker (effective `REDIS_URL` / DB index)
- API default: `REDIS_URL=redis://127.0.0.1:6379/0` (DB index `0`).
- Worker default: `REDIS_URL=redis://127.0.0.1:6379/0` (DB index `0`).
- Docker Compose sets both services to `redis://redis:6379/0`.

## 3) Queue names used by enqueue vs worker listen
- API enqueue queue: first queue from `RQ_QUEUES` env (default `jobs`).
- Worker listen queues: all queue names in `RQ_QUEUES` (default `jobs`).
- Effective default aligns (`jobs`) if both services share the same env.

## 4) Job artifacts directory + whether shared between services
- API artifacts root: `JOBS_DIR` env or default `Projects/data/jobs`.
- Worker artifacts root: `JOBS_DIR` env or default `Projects/data/jobs`.
- Docker Compose mounts `./Projects/data/jobs:/app/data/jobs` into both API and worker, so artifacts are shared in the container deployment.

## 5) Current status contract fields + where written/read
- Common fields observed in `job.json` and `/status` output:
  - `job_id`, `status`, `stage`, `progress`, `message`, `error`, `updated_at`, `rq_id`
  - optionally `proxy_path`, `proxy_ready`, `proxy_url`, `video_path`, `setup`, clip/result fields.
- Written by:
  - API: `set_status()` and upload/setup/run/cancel/retry flows.
  - Worker: direct writes to `job.json` and `results.json` at completion/failure.
  - RQ metadata (`job.meta`): worker updates transient `stage`/`progress`.
- Read by:
  - API `/jobs/{job_id}/status` and `/jobs/{job_id}/results`.
  - UI polls `/jobs/{job_id}/status` and fetches result artifacts.

## 6) Top 5 risks causing “queued forever” in Docker/RunPod
1. **Queue mismatch**: API enqueues to one queue while worker listens to another (`RQ_QUEUES` divergence).
2. **Redis endpoint mismatch**: API and worker using different `REDIS_URL` host/db.
3. **Worker not healthy/running**: container process exits or dependency crash before consuming jobs.
4. **Shared volume misconfiguration**: API and worker not sharing the same `JOBS_DIR`, causing missing inputs/metadata confusion.
5. **Status drift between RQ and file metadata**: RQ state changes but `job.json` not updated due to crashes/race, making UI appear stuck.
