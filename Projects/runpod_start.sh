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
export SHIFTCLIPPER_DEVICE="${SHIFTCLIPPER_DEVICE:-cuda:0}"

mkdir -p "$JOBS_DIR"

apt-get update -y
apt-get install -y ffmpeg redis-server python3-venv python3-pip curl

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
if [[ "$SHIFTCLIPPER_DEVICE" == cuda* ]]; then
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
else
  pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
fi
pip install -r requirements.txt

if ! redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
  redis-server --daemonize yes
fi
redis-cli -u "$REDIS_URL" ping >/dev/null

pkill -f "uvicorn api.main:app" || true
pkill -f "rq worker -u" || true

nohup "$VENV_DIR/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 --log-level info > "$API_LOG" 2>&1 &
API_PID=$!

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null
curl -fsS -o /dev/null -w "%{http_code}" "http://127.0.0.1:8000/" | grep -q "200"
curl -fsS -o /dev/null -w "%{http_code}" "http://127.0.0.1:8000/static/app.js" | grep -q "200"

nohup "$VENV_DIR/bin/python" -m worker.main > "$WORKER_LOG" 2>&1 &
WORKER_PID=$!

sleep 2
if ! kill -0 "$WORKER_PID" >/dev/null 2>&1; then
  echo "ERROR: worker exited"
  tail -n 200 "$WORKER_LOG" || true
  exit 1
fi

echo "API PID: $API_PID"
echo "Worker PID: $WORKER_PID"
echo "SHIFTCLIPPER_DEVICE=$SHIFTCLIPPER_DEVICE"
