#!/usr/bin/env bash
set -euo pipefail

# Always run from Projects dir
cd "$(dirname "$0")"

echo "[runpod] ensuring python venv..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip

REQ_FILE="${REQ_FILE:-requirements.runpod.txt}"
echo "[runpod] installing requirements from ${REQ_FILE}..."
pip install -r "${REQ_FILE}"

# Ensure dirs exist so FastAPI StaticFiles mount never crashes
mkdir -p data/jobs web

echo "[runpod] starting redis..."
nohup redis-server --bind 0.0.0.0 --port 6379 > /workspace/redis.log 2>&1 &

echo "[runpod] starting rq worker..."
nohup python -m rq worker jobs --url redis://127.0.0.1:6379 > /workspace/worker.log 2>&1 &

echo "[runpod] starting api..."
nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/api.log 2>&1 &

echo ""
echo "Started:"
echo "  API:    tail -f /workspace/api.log"
echo "  Worker: tail -f /workspace/worker.log"
echo "  Redis:  tail -f /workspace/redis.log"
echo ""
echo "[runpod] Open the web UI via your RunPod exposed port (8000)."

