#!/usr/bin/env bash
set -euo pipefail

# ShiftClipper RunPod start script (no guessing)
# - creates venv if missing
# - installs requirements
# - starts Redis + RQ worker + API
# - uses `python -m ...` so PATH issues cannot break (rq "No such file" etc)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_API=/workspace/api.log
LOG_WORKER=/workspace/worker.log
LOG_REDIS=/workspace/redis.log

echo "[runpod_start] cwd: $PWD"

# System deps
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg libgl1 libglib2.0-0

# Virtualenv
if [ ! -d ".venv" ]; then
  echo "[runpod_start] creating venv..."
  python3 -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

REQ_FILE="requirements.runpod.txt"
if [ -f "requirements.runpod_pro.txt" ]; then
  # If you are using the 'pro' tracking variant, keep this file present.
  REQ_FILE="requirements.runpod_pro.txt"
fi

echo "[runpod_start] installing python deps from $REQ_FILE..."
python -m pip install -r "$REQ_FILE"

# Ensure model weights present (optional; Ultralytics can also auto-download)
if [ ! -f "yolov8s.pt" ]; then
  echo "[runpod_start] downloading yolov8s.pt..."
  python - <<'PY'
from ultralytics.utils.downloads import safe_download
safe_download('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt', file='yolov8s.pt')
PY
fi

# Kill anything stale
pkill -f "redis-server.*:6379" || true
pkill -f "python -m rq worker" || true
pkill -f "uvicorn api.main:app" || true

# Start Redis
nohup redis-server --bind 0.0.0.0 --port 6379 > "$LOG_REDIS" 2>&1 &

# Start Worker (NO --serializer flags)
nohup python -m rq worker jobs --url redis://127.0.0.1:6379 > "$LOG_WORKER" 2>&1 &

# Start API
nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info > "$LOG_API" 2>&1 &

echo ""
echo "Started:"
echo "  API:    tail -f $LOG_API"
echo "  Worker: tail -f $LOG_WORKER"
echo "  Redis:  tail -f $LOG_REDIS"
echo ""
echo "Quick health checks:"
echo "  curl -sSf http://127.0.0.1:8000/ | head"

