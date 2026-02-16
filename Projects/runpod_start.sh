#!/usr/bin/env bash
set -euo pipefail

# ShiftClipper RunPod bootstrap (no guessing).
# Run from: /workspace/shiftclipper/Projects

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR"

LOG_API=/workspace/api.log
LOG_WORKER=/workspace/worker.log
LOG_REDIS=/workspace/redis.log

echo "[runpod] working dir: $ROOT_DIR"

# --- system deps (fresh pods need these) ---
echo "[runpod] installing system deps..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  ffmpeg \
  redis-server \
  ca-certificates \
  curl \
  git \
  libgl1 \
  libglib2.0-0

# --- venv ---
if [[ ! -d ".venv" ]]; then
  echo "[runpod] creating venv..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[runpod] upgrading pip..."
python -m pip install --upgrade pip wheel setuptools

REQ_FILE="requirements.runpod.txt"
if [[ -f "requirements.runpod_pro.txt" ]]; then
  # If you've added the pro requirements file, prefer it.
  REQ_FILE="requirements.runpod_pro.txt"
fi

echo "[runpod] installing requirements from $REQ_FILE ..."
python -m pip install -r "$REQ_FILE"

# --- start redis ---
if pgrep -f "redis-server.*:6379" >/dev/null 2>&1; then
  echo "[runpod] redis already running"
else
  echo "[runpod] starting redis..."
  nohup redis-server --bind 0.0.0.0 --port 6379 > "$LOG_REDIS" 2>&1 &
  sleep 0.5
fi

# --- warm up models (pre-download weights so jobs don't look 'stuck') ---
echo "[runpod] warming up models (YOLO + EasyOCR). This can take a few minutes only on first run..."
python - <<'PY'
import os
import torch
from ultralytics import YOLO

# YOLO weights (download/cache)
w = os.environ.get("YOLO_WEIGHTS", "yolov8s.pt")
YOLO(w)
print("[warmup] YOLO ok:", w)

# EasyOCR models (download/cache)
try:
    import easyocr
    use_gpu = bool(torch.cuda.is_available())
    print(f"[warmup] torch.cuda.is_available() = {use_gpu}")
    easyocr.Reader(['en'], gpu=use_gpu)
    print(f"[warmup] EasyOCR ok (gpu={use_gpu})")
except Exception as e:
    # OCR is optional; don't block startup if it fails
    print("[warmup] EasyOCR warmup skipped:", e)
PY

# --- start worker ---
if pgrep -f "rq worker jobs" >/dev/null 2>&1; then
  echo "[runpod] rq worker already running"
else
  echo "[runpod] starting rq worker..."
  # IMPORTANT: do NOT pass --serializer/-S (RQ 2.0 treats 'pickle' as invalid attribute)
  nohup "$ROOT_DIR/.venv/bin/rq" worker jobs --url redis://127.0.0.1:6379 > "$LOG_WORKER" 2>&1 &
  sleep 0.5
fi

# --- start api ---
if pgrep -f "uvicorn api.main:app" >/dev/null 2>&1; then
  echo "[runpod] api already running"
else
  echo "[runpod] starting api..."
  nohup "$ROOT_DIR/.venv/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 --log-level info > "$LOG_API" 2>&1 &
  sleep 0.5
fi

echo ""
echo "Started:"
echo "  API:    tail -f $LOG_API"
echo "  Worker: tail -f $LOG_WORKER"
echo "  Redis:  tail -f $LOG_REDIS"
echo ""
echo "[runpod] Open the web UI via your RunPod exposed port (8000)."


