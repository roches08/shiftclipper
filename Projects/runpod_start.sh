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
export APP_ROOT="$PROJECTS_DIR"

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

WORKER_QUEUE="${RQ_QUEUES%%,*}"
echo "==> Starting worker for queue: $WORKER_QUEUE"
nohup "$VENV_DIR/bin/rq" worker -u "$REDIS_URL" "$WORKER_QUEUE" > "$WORKER_LOG" 2>&1 &
WORKER_PID=$!

echo "==> Starting API (0.0.0.0:8000)"
nohup "$VENV_DIR/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 --log-level info > "$API_LOG" 2>&1 &
API_PID=$!

echo "==> Waiting for API readiness"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null 2>&1; then
  echo "ERROR: API failed readiness checks. Last API logs:"
  tail -n 100 "$API_LOG" || true
  exit 1
fi

echo "==> Verifying UI static assets"
if ! curl -f "http://localhost:8000/" >/dev/null; then
  echo "ERROR: UI root check failed: curl -f http://localhost:8000/"
  tail -n 100 "$API_LOG" || true
  exit 1
fi

if ! curl -f "http://localhost:8000/static/app.js" >/dev/null; then
  echo "ERROR: UI static check failed: curl -f http://localhost:8000/static/app.js"
  tail -n 100 "$API_LOG" || true
  exit 1
fi

UI_URL="http://localhost:8000/"
echo "UI URL: $UI_URL"

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
echo "UI URL: $UI_URL"
echo "======================================="
