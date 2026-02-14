#!/usr/bin/env bash
set -euo pipefail

# Always run from the folder that contains this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export APP_ROOT="$SCRIPT_DIR"
export JOBS_ROOT="${JOBS_ROOT:-$APP_ROOT/data/jobs}"
export PYTHONPATH="$APP_ROOT:${PYTHONPATH:-}"

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

echo "[ShiftClipper] Warming YOLO weights..."
python3 - << 'PY'
from ultralytics import YOLO
YOLO("yolov8n.pt")
print("YOLO weights ready")
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

