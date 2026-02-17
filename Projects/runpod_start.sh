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
export SHIFTCLIPPER_REQS="${SHIFTCLIPPER_REQS:-auto}"

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

REQ_PROFILE=""
case "$SHIFTCLIPPER_REQS" in
  pro)
    REQ_PROFILE="pro"
    ;;
  base)
    REQ_PROFILE="base"
    ;;
  auto)
    if [[ "$SHIFTCLIPPER_DEVICE" == cuda* ]]; then
      REQ_PROFILE="pro"
    else
      REQ_PROFILE="base"
    fi
    ;;
  *)
    echo "ERROR: Invalid SHIFTCLIPPER_REQS value '$SHIFTCLIPPER_REQS'. Expected pro|base|auto."
    exit 1
    ;;
esac

if [[ "$REQ_PROFILE" == "pro" ]]; then
  REQ_FILE="requirements.runpod_pro.txt"
else
  REQ_FILE="requirements.runpod.txt"
fi

echo "Installing RunPod requirements profile '$REQ_PROFILE' from $REQ_FILE"
pip install -r "$REQ_FILE"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cuda_ver', torch.version.cuda)"
if [[ "$SHIFTCLIPPER_DEVICE" == cuda* ]]; then
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: SHIFTCLIPPER_DEVICE=$SHIFTCLIPPER_DEVICE but torch.cuda.is_available() is False."
    exit 1
  }
fi

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

check_api() {
  local url="$1"
  local expect="${2:-200}"
  local code
  code="$(curl -sS -o /dev/null -w "%{http_code}" "$url" || true)"
  if [[ "$code" != "$expect" ]]; then
    echo "ERROR: API check failed for $url (expected $expect, got ${code:-n/a})"
    echo "--- API log tail ---"
    tail -n 200 "$API_LOG" || true
    exit 1
  fi
}

check_api "http://127.0.0.1:8000/api/health"
check_api "http://127.0.0.1:8000/"
check_api "http://127.0.0.1:8000/static/app.js"

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
