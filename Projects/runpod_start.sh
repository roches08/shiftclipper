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
apt-get install -y ffmpeg redis-server python3-venv python3-pip curl git

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
# Non-negotiable: use only venv python/pip for every install/check in this script.
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if [[ "$SHIFTCLIPPER_DEVICE" == cuda* ]]; then
  "$PIP_BIN" install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
else
  "$PIP_BIN" install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
fi

REQUIREMENTS_FILE="requirements.runpod_pro.txt"
echo "Installing ShiftClipper requirements from $REQUIREMENTS_FILE"
"$PIP_BIN" install -r "$REQUIREMENTS_FILE"

"$PYTHON_BIN" -c "import pkg_resources; print('pkg_resources ok')" || "$PIP_BIN" install -U setuptools
"$PYTHON_BIN" -c "import pkg_resources; print('pkg_resources ok')"

if ! "$PYTHON_BIN" -c "import pkg_resources, ultralytics, torch; print('ok', torch.__version__, torch.cuda.is_available())"; then
  echo "ERROR: dependency import check failed."
  tail -n 120 "$API_LOG" || true
  tail -n 120 "$WORKER_LOG" || true
  exit 1
fi

if ! "$PYTHON_BIN" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"; then
  echo "ERROR: Torch import check failed."
  tail -n 120 "$API_LOG" || true
  tail -n 120 "$WORKER_LOG" || true
  exit 1
fi
"$PYTHON_BIN" -c "import pkg_resources; print('pkg_resources', pkg_resources.__name__)"
if [[ "$SHIFTCLIPPER_DEVICE" == cuda* ]]; then
  "$PYTHON_BIN" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
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
