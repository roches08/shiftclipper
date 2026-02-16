#!/usr/bin/env bash
set -euo pipefail

# One-command RunPod startup for ShiftClipper (PRO tracking)

PROJECTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECTS_DIR"

echo "==> System deps"
apt-get update -y
apt-get install -y ffmpeg redis-server python3-venv python3-pip git

echo "==> Python venv"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "==> Python requirements"
REQ_FILE="requirements.runpod_pro.txt"
if [ ! -f "$REQ_FILE" ]; then
  echo "Missing $REQ_FILE. Put it in Projects/ next to this script."
  exit 1
fi
pip install -r "$REQ_FILE"

echo "==> Redis"
redis-server --daemonize yes
redis-cli ping >/dev/null

echo "==> Stop old"
pkill -f "uvicorn api.main:app" || true
pkill -f "python -m worker.main" || true

echo "==> Start worker"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
export JOBS_DIR="${JOBS_DIR:-$PROJECTS_DIR/data/jobs}"
export RQ_QUEUES="${RQ_QUEUES:-jobs}"
mkdir -p "$JOBS_DIR"
echo "Worker env: REDIS_URL=$REDIS_URL RQ_QUEUES=$RQ_QUEUES JOBS_DIR=$JOBS_DIR"
nohup "$PROJECTS_DIR/.venv/bin/python" -m worker.main > /workspace/worker.log 2>&1 &
sleep 1
if pgrep -f "python -m worker.main" >/dev/null; then
  echo "Worker started and running."
else
  echo "Worker failed to stay running; see /workspace/worker.log"
  tail -n 120 /workspace/worker.log || true
  exit 1
fi

echo "==> Start API"
nohup "$PROJECTS_DIR/.venv/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/api.log 2>&1 &

echo
echo "✅ Started (PRO)."
echo "API log:    tail -n 200 /workspace/api.log"
echo "Worker log: tail -n 200 /workspace/worker.log"
echo "Queue check: redis-cli llen rq:queue:jobs"
