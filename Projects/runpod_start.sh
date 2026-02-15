cd /workspace/shiftclipper/Projects

cp runpod_start.sh runpod_start.sh.bak.$(date +%s)

cat > runpod_start.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg redis-server python3-venv python3-pip git

echo "==> Python venv"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.runpod.txt

echo "==> Quick GPU sanity (if torch installed)"
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch check skipped:", e)
PY

echo "==> Redis"
redis-server --daemonize yes
redis-cli ping

echo "==> Stop any old processes (best-effort)"
pkill -f "uvicorn api.main:app" || true
pkill -f "rq worker" || true

echo "==> Start RQ worker"
nohup "$ROOT/.venv/bin/rq" worker jobs --url redis://127.0.0.1:6379 > /workspace/worker.log 2>&1 &

echo "==> Start API"
nohup "$ROOT/.venv/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/api.log 2>&1 &

echo
echo "✅ Started."
echo "API log:    tail -n 200 /workspace/api.log"
echo "Worker log: tail -n 200 /workspace/worker.log"
SH

chmod +x runpod_start.sh

