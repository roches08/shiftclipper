#!/usr/bin/env bash
set -euo pipefail

API_LOG="/workspace/api.log"
WORKER_LOG="/workspace/worker.log"

export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export RQ_QUEUES="${RQ_QUEUES:-jobs}"
export SHIFTCLIPPER_DEVICE="${SHIFTCLIPPER_DEVICE:-cuda:0}"

PROJECTS_DIR="/workspace/shiftclipper/Projects"
cd "$PROJECTS_DIR"

ensure_ui_static() {
  local static_index="$PROJECTS_DIR/static/index.html"
  local static_app_js="$PROJECTS_DIR/static/app.js"
  local static_presets_js="$PROJECTS_DIR/static/presets.js"

  if [ ! -f "$static_index" ] || [ ! -f "$static_app_js" ] || [ ! -f "$static_presets_js" ]; then
    echo "ERROR: Missing committed UI assets in $PROJECTS_DIR/static"
    echo "Expected files: $static_index, $static_app_js, $static_presets_js"
    exit 1
  fi

  echo "UI static ready: $PROJECTS_DIR/static"
}

file_size() {
  wc -c < "$1" | tr -d ' '
}

REID_WEIGHTS_DIR="$PROJECTS_DIR/models/reid"
REID_WEIGHTS_DEST="$REID_WEIGHTS_DIR/osnet_x0_25_msmt17.pth"
REID_WEIGHTS_TMP="${REID_WEIGHTS_DEST}.tmp"
REID_WEIGHTS_URL="https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
REID_MIN_BYTES=$((5 * 1024 * 1024))

if [[ "$REID_WEIGHTS_URL" == https://huggingface.co/*/resolve/*.pth ]] && [[ "$REID_WEIGHTS_URL" != *\?* ]]; then
  REID_WEIGHTS_URL="${REID_WEIGHTS_URL}?download=true"
fi

export JOBS_DIR="${JOBS_DIR:-$PROJECTS_DIR/data/jobs}"
mkdir -p "$JOBS_DIR"

VENV_DIR="$PROJECTS_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if [[ "$SHIFTCLIPPER_DEVICE" == cuda* ]]; then
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
else
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
fi

REQUIREMENTS_FILE="requirements.runpod_pro.txt"
"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"

"$PYTHON_BIN" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
"$PYTHON_BIN" -c "import ultralytics; print('ultralytics import ok')"
"$PYTHON_BIN" -c "import torch, pkg_resources, ultralytics, cv2, redis, rq, tensorboard, torchreid; print('dependency sanity checks ok')"

ensure_ui_static

mkdir -p "$REID_WEIGHTS_DIR"
if [ -s "$REID_WEIGHTS_DEST" ] && [ "$(file_size "$REID_WEIGHTS_DEST")" -gt "$REID_MIN_BYTES" ]; then
  echo "ReID weights already present at $REID_WEIGHTS_DEST"
else
  echo "Bootstrapping ReID weights to $REID_WEIGHTS_DEST"
  rm -f "$REID_WEIGHTS_TMP"
  if ! curl -L --fail --retry 3 --retry-delay 2 "$REID_WEIGHTS_URL" -o "$REID_WEIGHTS_TMP"; then
    echo "WARNING: ReID weights download failed for $REID_WEIGHTS_URL; continuing without local ReID weights"
    rm -f "$REID_WEIGHTS_TMP"
  elif head -c 200 "$REID_WEIGHTS_TMP" | grep -qi "<html"; then
    header="$(head -c 200 "$REID_WEIGHTS_TMP" | tr '\n' ' ')"
    echo "WARNING: ReID weights download returned HTML for $REID_WEIGHTS_URL; header=$header; continuing without local ReID weights"
    rm -f "$REID_WEIGHTS_TMP"
  elif [ ! -s "$REID_WEIGHTS_TMP" ] || [ "$(file_size "$REID_WEIGHTS_TMP")" -le "$REID_MIN_BYTES" ]; then
    size="$(file_size "$REID_WEIGHTS_TMP" 2>/dev/null || echo 0)"
    header="$(head -c 200 "$REID_WEIGHTS_TMP" 2>/dev/null | tr '\n' ' ')"
    echo "WARNING: ReID weights download failed validation for $REID_WEIGHTS_URL; size=$size header=$header; continuing without local ReID weights"
    rm -f "$REID_WEIGHTS_TMP"
  else
    mv "$REID_WEIGHTS_TMP" "$REID_WEIGHTS_DEST"
  fi
fi

if [ -s "$REID_WEIGHTS_DEST" ] && [ "$(file_size "$REID_WEIGHTS_DEST")" -gt "$REID_MIN_BYTES" ]; then
  if ! "$PYTHON_BIN" -c "from worker.reid import OSNetEmbedder, ReIDConfig; import torch; OSNetEmbedder(ReIDConfig(model_name='osnet_x0_25', device='cuda:0' if torch.cuda.is_available() else 'cpu', batch_size=1, use_fp16=torch.cuda.is_available(), weights_path='$REID_WEIGHTS_DEST')); print('reid ok')"; then
    echo "ERROR: ReID bootstrap sanity check failed. UI/API will continue running."
  fi
else
  echo "WARNING: ReID weights missing or invalid after bootstrap; UI/API will continue running."
fi

if ! redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
  redis-server --daemonize yes
fi
redis-cli -u "$REDIS_URL" ping >/dev/null

pkill -f "uvicorn api.main:app" || true
pkill -f "python -m worker.main" || true
sleep 1

nohup "$VENV_DIR/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 --log-level info > "$API_LOG" 2>&1 &
API_PID=$!

API_UP=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null 2>&1; then
    API_UP=1
    break
  fi
  sleep 1
done

if [ "$API_UP" -ne 1 ]; then
  echo "ERROR: API never became healthy"
  echo "--- API log tail ---"
  tail -n 250 "$API_LOG" || true
  exit 1
fi

check_api() {
  local url="$1"
  local expect="${2:-200}"
  local code
  code="$(curl -sS -o /dev/null -w "%{http_code}" "$url" || true)"
  if [[ "$code" != "$expect" ]]; then
    echo "ERROR: API check failed for $url (expected $expect, got ${code:-n/a})"
    echo "--- curl -I ---"
    curl -sS -I "$url" || true
    echo "--- API log tail ---"
    tail -n 250 "$API_LOG" || true
    exit 1
  fi
}

check_api "http://127.0.0.1:8000/api/health"
check_api "http://127.0.0.1:8000/"
check_api "http://127.0.0.1:8000/static/app.js"

if ! curl -fsS "http://127.0.0.1:8000/" | grep -q "ShiftClipper — Tracker v2"; then
  echo "ERROR: UI root page did not contain expected application markup"
  tail -n 250 "$API_LOG" || true
  exit 1
fi

nohup "$VENV_DIR/bin/python" -m worker.main > "$WORKER_LOG" 2>&1 &
WORKER_PID=$!

sleep 2
if ! kill -0 "$WORKER_PID" >/dev/null 2>&1; then
  echo "ERROR: worker exited"
  tail -n 200 "$WORKER_LOG" || true
  exit 1
fi

echo "API PID: $API_PID"
echo "Worker PID: $WORKER_PID"
echo "SHIFTCLIPPER_DEVICE=$SHIFTCLIPPER_DEVICE"
