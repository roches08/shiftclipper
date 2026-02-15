#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Max accuracy defaults (override by exporting before running if you want)
# -----------------------------
export SHIFTCLIPPER_MAX_ACCURACY="${SHIFTCLIPPER_MAX_ACCURACY:-1}"
export YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8x.pt}"
export YOLO_IMGSZ="${YOLO_IMGSZ:-1280}"
export YOLO_CONF="${YOLO_CONF:-0.20}"
export YOLO_IOU="${YOLO_IOU:-0.45}"
export YOLO_HALF="${YOLO_HALF:-0}"
export DETECT_STRIDE="${DETECT_STRIDE:-1}"

# Always run from the Projects folder (where this script lives)
cd "$(dirname "$0")"

echo "==> Install system deps"
apt-get update -y
apt-get install -y ffmpeg redis-server python3 python3-venv python3-pip

echo "==> Quick GPU sanity"
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "==> Create venv (Projects/.venv) if needed"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrade pip + install python deps"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.runpod.txt

echo "==> Start Redis"
redis-server --daemonize yes
redis-cli ping

echo "==> Stop any old processes (best-effort)"
pkill -f "uvicorn api.main:app" || true
pkill -f ".venv/bin/rq worker" || true
pkill -f "rq worker jobs" || true

# IMPORTANT: prevent the 'Invalid attribute name: pickle' crash if anything set these
unset RQ_SERIALIZER || true
unset RQ_DEFAULT_SERIALIZER || true

echo "==> Start RQ worker (venv binary, no serializer flags)"
nohup "$(pwd)/.venv/bin/rq" worker jobs --url redis://127.0.0.1:6379 > /workspace/worker.log 2>&1 &

echo "==> Start API (venv uvicorn)"
nohup "$(pwd)/.venv/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/api.log 2>&1 &

sleep 1

echo
echo "✅ Started."
echo "Open your RunPod proxy for port 8000."
echo "API log:    tail -n 200 /workspace/api.log"
echo "Worker log: tail -n 200 /workspace/worker.log"
echo
echo "If the UI says 'Queued for processing' forever, check:"
echo "  tail -n 200 /workspace/worker.log"

