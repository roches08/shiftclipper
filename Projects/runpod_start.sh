#!/usr/bin/env bash
set -euo pipefail

PROJECTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECTS_DIR"

VENV_DIR="$PROJECTS_DIR/.venv"
API_LOG="/workspace/api.log"
WORKER_LOG="/workspace/worker.log"

export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export JOBS_DIR="${JOBS_DIR:-$PROJECTS_DIR/data/jobs}"
export RQ_QUEUES="${RQ_QUEUES:-jobs}"

mkdir -p "$JOBS_DIR"

echo "==> Installing system dependencies"
apt-get update -y
apt-get install -y ffmpeg redis-server python3-venv python3-pip curl

echo "==> Preparing Python virtualenv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "==> Ensuring Redis is running"
if ! redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
  redis-server --daemonize yes
fi
redis-cli -u "$REDIS_URL" ping >/dev/null

echo "==> Stopping old API/worker processes"
pkill -f "uvicorn api.main:app" || true
pkill -f "rq worker -u" || true

echo "==> Starting API (0.0.0.0:8000)"
nohup "$VENV_DIR/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 --log-level info > "$API_LOG" 2>&1 &
API_PID=$!

echo "==> Waiting for API readiness"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:8000/healthz" >/dev/null 2>&1 \
    || curl -fsS "http://127.0.0.1:8000/jobs" >/dev/null 2>&1 \
    || curl -fsS "http://127.0.0.1:8000/docs" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:8000/docs" >/dev/null 2>&1; then
  echo "ERROR: API failed readiness checks. Last API logs:"
  tail -n 100 "$API_LOG" || true
  exit 1
fi

echo "==> Starting worker for queue: jobs"
nohup "$VENV_DIR/bin/rq" worker -u "$REDIS_URL" jobs > "$WORKER_LOG" 2>&1 &
WORKER_PID=$!

sleep 1

echo
echo "===== ShiftClipper RunPod status ====="
echo "REDIS_URL=$REDIS_URL"
echo "RQ version: $("$VENV_DIR/bin/python" -c 'import rq; print(rq.__version__)')"
echo "Redis ping: $(redis-cli -u "$REDIS_URL" ping)"
echo "jobs queue length: $(redis-cli -u "$REDIS_URL" llen rq:queue:jobs)"
echo "API PID: $API_PID"
echo "Worker PID: $WORKER_PID"
echo "API log: $API_LOG"
echo "Worker log: $WORKER_LOG"
echo "======================================="
