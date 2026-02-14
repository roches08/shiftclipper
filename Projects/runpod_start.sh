#!/usr/bin/env bash

# Always run from the folder that contains this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export APP_ROOT="$SCRIPT_DIR"

set -euo pipefail

APP_ROOT="${APP_ROOT:-/workspace/Projects}"
JOBS_ROOT="${JOBS_ROOT:-$APP_ROOT/data/jobs}"
export APP_ROOT JOBS_ROOT
export PYTHONPATH="$APP_ROOT:${PYTHONPATH:-}"

cd "$APP_ROOT"

echo "[ShiftClipper] Ensuring system deps..."
need_pkgs=()
command -v ffmpeg >/dev/null 2>&1 || need_pkgs+=(ffmpeg)
command -v redis-server >/dev/null 2>&1 || need_pkgs+=(redis-server)
command -v unzip >/dev/null 2>&1 || need_pkgs+=(unzip)

if [ ${#need_pkgs[@]} -gt 0 ]; then
  apt-get update -y
  apt-get install -y "${need_pkgs[@]}"
fi

echo "[ShiftClipper] Installing Python deps..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.runpod.txt

echo "[ShiftClipper] Starting Redis..."
REDIS_BIN="$(command -v redis-server || true)"
if [ -z "$REDIS_BIN" ] && [ -x /usr/bin/redis-server ]; then REDIS_BIN="/usr/bin/redis-server"; fi
if [ -z "$REDIS_BIN" ]; then
  echo "ERROR: redis-server not found even after install."
  exit 1
fi
$REDIS_BIN --daemonize yes
sleep 0.5
echo "[ShiftClipper] Warming YOLO weights (pre-download)..."
python3 - << 'PY'
from ultralytics import YOLO
YOLO("yolov8n.pt")  # downloads once if missing
print("YOLO weights ready")
PY

echo "[ShiftClipper] Starting RQ worker..."
nohup rq worker jobs --url redis://127.0.0.1:6379 --default-job-timeout 3600 > /workspace/worker.log 2>&1 &

echo "[ShiftClipper] Starting API..."
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/api.log 2>&1 &

sleep 1
echo "[ShiftClipper] Ready."
echo "Open: http://<pod-ip>:8000"
echo "Logs: tail -f /workspace/api.log /workspace/worker.log"
