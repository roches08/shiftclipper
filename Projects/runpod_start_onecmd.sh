#!/usr/bin/env bash
set -euo pipefail

# ShiftClipper RunPod bootstrap (GPU-safe, one-command start)
# Usage:
#   bash runpod_start.sh
#
# Logs:
#   /workspace/api.log
#   /workspace/worker.log

PROJECT_DIR="/workspace/shiftclipper/Projects"
REDIS_URL="redis://127.0.0.1:6379"

echo "==> cd ${PROJECT_DIR}"
cd "${PROJECT_DIR}"

echo "==> System deps (ffmpeg, redis, curl, git)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  ffmpeg redis-server curl git ca-certificates \
  python3-venv python3-dev build-essential
rm -rf /var/lib/apt/lists/*

echo "==> Python venv"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

echo "==> Python deps (RunPod)"
# Install your pinned app deps
python -m pip install -r requirements.runpod.txt

echo "==> PyTorch GPU (CUDA 12.1 wheels)"
# L4 pods are typically CUDA 12.x. cu121 wheels are the safest default.
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo "==> Quick GPU sanity"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "==> Redis"
redis-server --daemonize yes
redis-cli ping

echo "==> Stop any old processes (best-effort)"
pkill -f "uvicorn api.main:app" >/dev/null 2>&1 || true
pkill -f "rq worker jobs" >/dev/null 2>&1 || true

echo "==> Start RQ worker"
nohup .venv/bin/rq worker jobs --url "${REDIS_URL}" > /workspace/worker.log 2>&1 &

echo "==> Start API"
nohup .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/api.log 2>&1 &

echo ""
echo "✅ Started."
echo "API log:    tail -n 200 /workspace/api.log"
echo "Worker log: tail -n 200 /workspace/worker.log"
echo ""
echo "If the UI says 'Queued for processing' forever, run:"
echo "  tail -n 200 /workspace/worker.log"
