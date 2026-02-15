#!/usr/bin/env bash
set -euo pipefail

cd /workspace/shiftclipper/Projects

# ---- system deps ----
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg git python3-venv

# ---- venv ----
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip

# install python deps
pip install -r requirements.runpod.txt

# ---- start services ----
# kill any old processes
pkill -f "redis-server" || true
pkill -f "uvicorn api.main:app" || true
pkill -f "rq worker jobs" || true

# start redis
nohup redis-server --bind 127.0.0.1 --port 6379 > /workspace/redis.log 2>&1 &

# start rq worker (USE VENV rq)
nohup .venv/bin/rq worker jobs --url redis://127.0.0.1:6379 > /workspace/worker.log 2>&1 &

# start api (USE VENV uvicorn)
nohup .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/api.log 2>&1 &

echo "Started:"
echo "  API:    tail -f /workspace/api.log"
echo "  Worker: tail -f /workspace/worker.log"
echo "  Redis:  tail -f /workspace/redis.log"

