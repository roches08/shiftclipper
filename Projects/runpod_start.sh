#!/usr/bin/env bash
set -euo pipefail

# ShiftClipper RunPod bootstrap (one command)
# - creates/uses Projects/.venv
# - installs requirements.runpod.txt
# - starts redis + rq worker + uvicorn
# Logs:
#   /workspace/api.log
#   /workspace/worker.log

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../Projects
cd "$ROOT_DIR"

echo "==> Working dir: $ROOT_DIR"

echo "==> Quick GPU sanity"
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "==> OS deps (ffmpeg + redis)"
apt-get update -y
apt-get install -y ffmpeg redis-server python3-venv python3-pip

echo "==> Python venv + deps"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.runpod.txt

echo "==> Redis"
redis-server --daemonize yes
redis-cli ping

echo "==> Stop any old processes (best-effort)"
pkill -f "uvicorn api.main:app" || true
pkill -f "rq worker" || true

echo "==> Start RQ worker"
nohup .venv/bin/rq worker jobs --url redis://127.0.0.1:6379 > /workspace/worker.log 2>&1 &

echo "==> Start API"
nohup .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/api.log 2>&1 &

sleep 1
echo
echo "✅ Started."
echo "API log:    tail -n 200 /workspace/api.log"
echo "Worker log: tail -n 200 /workspace/worker.log"
echo
echo "If the UI says 'Queued for processing' forever, run:"
echo "  tail -n 200 /workspace/worker.log"
