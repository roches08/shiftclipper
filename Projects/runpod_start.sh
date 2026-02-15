#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export APP_ROOT="$SCRIPT_DIR"
export JOBS_ROOT="${JOBS_ROOT:-$APP_ROOT/data/jobs}"
export PYTHONPATH="$APP_ROOT:${PYTHONPATH:-}"

# MAX accuracy defaults (override any time by setting env vars)
export YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8x.pt}"
export YOLO_IMGSZ="${YOLO_IMGSZ:-960}"
export YOLO_CONF="${YOLO_CONF:-0.25}"
export DETECT_STRIDE="${DETECT_STRIDE:-1}"

echo "[ShiftClipper] Ensuring system deps..."
need_pkgs=()
command -v ffmpeg >/dev/null 2>&1 || need_pkgs+=(ffmpeg)
command -v redis-server >/dev/null 2>&1 || need_pkgs+=(redis-server)

if [ ${#need_pkgs[@]} -gt 0 ]; then
  apt-get update -y
  apt-get install -y "${need_pkgs[@]}"
fi

echo "[ShiftClipper] Installing Python deps..."
python3 -m pip install -r requirements.runpod.txt

echo "[ShiftClipper] Starting Redis..."
redis-server --daemonize yes
sleep 0.5

echo "[ShiftClipper] Warming YOLO weights: ${YOLO_WEIGHTS} (imgsz=${YOLO_IMGSZ}, conf=${YOLO_CONF})"
python3 - << 'PY'
import os
from ultralytics import YOLO
w = os.getenv("YOLO_WEIGHTS", "yolov8x.pt")
YOLO(w)
print("YOLO weights ready:", w)
PY

echo "[ShiftClipper] Starting RQ worker..."
pkill -f "rq worker" >/dev/null 2>&1 || true
nohup rq worker jobs --url redis://127.0.0.1:6379 > /workspace/worker.log 2>&1 &

echo "[ShiftClipper] Starting API..."
pkill -f uvicorn >/dev/null 2>&1 || true
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/api.log 2>&1 &

sleep 1
echo "[ShiftClipper] Ready."
echo "Open: http://<pod-ip>:8000"
echo "Logs: tail -f /workspace/api.log /workspace/worker.log"

