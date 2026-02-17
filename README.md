# ShiftClipper

Queue-driven video clipping service (FastAPI + RQ worker + Redis) designed for local Docker and RunPod deployments.

## Quick start (local)

```bash
docker compose up --build
```

- API: `http://127.0.0.1:8000`
- Health: `GET /healthz`
- Readiness: `GET /readyz`

Smoke test:

```bash
python Projects/scripts/smoke_test.py
```

## RunPod startup

```bash
cd /workspace/shiftclipper/Projects
chmod +x runpod_start.sh
REDIS_URL=redis://127.0.0.1:6379/0 \
RQ_QUEUES=jobs \
JOBS_DIR=/workspace/shiftclipper/Projects/data/jobs \
bash runpod_start.sh
```

Optional lightweight mode:

```bash
export WORKER_VERIFY_ONLY=1
```

## Environment variables

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | API + Worker | Redis endpoint + DB index |
| `RQ_QUEUES` | `jobs` | API + Worker | Queue names (API enqueues first queue; worker listens all) |
| `JOBS_DIR` | `Projects/data/jobs` | API + Worker | Shared artifacts directory |
| `WORKER_VERIFY_ONLY` | `0` | Worker | Verify queue/status flow without heavy CV stack |
| `REDIS_MAX_RETRIES` | `3` | API + Worker | Redis connect retries |
| `REDIS_RETRY_DELAY_S` | `0.5` | API + Worker | Delay between Redis retries |
| `FFMPEG_TIMEOUT_S` | `1800` | Worker | ffmpeg command timeout |
| `FFMPEG_RETRIES` | `2` | Worker | ffmpeg retry attempts |

## Operational notes

- Job metadata: `Projects/data/jobs/<job_id>/job.json`
- Results: `.../results.json`
- Per-job artifact manifest: `.../manifest.json`
- Cleanup tooling:
  - API endpoint: `POST /jobs/cleanup?days=7&max_count=200`
  - Script: `python Projects/scripts/cleanup_jobs.py --days 7 --max-count 200`

## “Queued forever” troubleshooting

1. Confirm API + worker use the same `REDIS_URL` and `RQ_QUEUES`.
2. Verify worker process is running and logs `worker_ready`.
3. Verify API readiness (`/readyz`) and Redis reachability.
4. Ensure API and worker share the same mounted `JOBS_DIR`.
5. Poll `/jobs/<job_id>/status` and inspect `rq_state` + `manifest_status`.
6. Run `python Projects/scripts/smoke_test.py` to validate end-to-end flow.

## Security baseline

- Dependabot config is included in `.github/dependabot.yml`.
- Security policy is in `SECURITY.md`.
- Enable **Secret scanning** and **Push protection** in GitHub repository settings:
  - `Settings` → `Security & analysis` → toggle both features on.

## Suggested GitHub repo description/topics

Suggested description:

> Queue-based video clipping API and worker stack for sports highlight extraction (FastAPI, Redis, RQ, Docker/RunPod).

Suggested topics:

`fastapi`, `redis`, `rq`, `video-processing`, `ffmpeg`, `docker`, `runpod`, `mlops`, `sports-analytics`
