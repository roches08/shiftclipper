#!/usr/bin/env bash
set -euo pipefail

API_LOG="/workspace/api.log"
WORKER_LOG="/workspace/worker.log"

export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export RQ_QUEUES="${RQ_QUEUES:-jobs}"
export SHIFTCLIPPER_DEVICE="${SHIFTCLIPPER_DEVICE:-cuda:0}"

apt-get update -y
apt-get install -y ffmpeg redis-server python3-venv python3-pip curl git

REPO_URL="https://github.com/roches08/shiftclipper.git"
REPO_DIR="/workspace/shiftclipper"

if [ ! -d "$REPO_DIR/.git" ]; then
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch --all --prune
  git -C "$REPO_DIR" reset --hard origin/main
fi

PROJECTS_DIR="$REPO_DIR/Projects"
cd "$PROJECTS_DIR"

ensure_ui_static() {
  local static_js="$PROJECTS_DIR/static/app.js"

  if [ -f "$static_js" ]; then
    echo "UI static already present: $static_js"
    return 0
  fi

  echo "UI static missing; building UI..."

  apt-get update -y
  apt-get install -y nodejs npm

  local pkg=""
  for p in "$PROJECTS_DIR/ui/package.json" "$PROJECTS_DIR/web/package.json" "$PROJECTS_DIR/frontend/package.json"; do
    if [ -f "$p" ]; then
      pkg="$p"
      break
    fi
  done

  if [ -z "$pkg" ]; then
    pkg="$(find "$PROJECTS_DIR" -maxdepth 4 -name package.json | head -n 1 || true)"
  fi

  if [ -z "$pkg" ]; then
    echo "ERROR: package.json not found; cannot build UI"
    return 1
  fi

  local ui_dir
  ui_dir="$(dirname "$pkg")"
  echo "Building UI in $ui_dir"

  pushd "$ui_dir" >/dev/null
  if [ -f package-lock.json ]; then
    npm ci
  else
    npm install
  fi
  npm run build
  popd >/dev/null

  mkdir -p "$PROJECTS_DIR/static"
  if [ -f "$ui_dir/dist/app.js" ]; then
    cp -r "$ui_dir/dist/"* "$PROJECTS_DIR/static/"
  elif [ -f "$ui_dir/build/app.js" ]; then
    cp -r "$ui_dir/build/"* "$PROJECTS_DIR/static/"
  else
    local dist_dir
    dist_dir="$(find "$ui_dir" -maxdepth 2 -type d -name dist | head -n 1 || true)"
    if [ -n "$dist_dir" ]; then
      cp -r "$dist_dir/"* "$PROJECTS_DIR/static/"
    fi
  fi

  if [ ! -f "$static_js" ]; then
    echo "ERROR: UI build completed but $static_js still missing."
    echo "Static dir contents:"
    ls -lah "$PROJECTS_DIR/static" || true
    return 1
  fi

  echo "UI static ready: $static_js"
}

ensure_ui_static

REID_WEIGHTS_DIR="$PROJECTS_DIR/models/reid"
REID_WEIGHTS_DEST="$REID_WEIGHTS_DIR/osnet_x0_25_msmt17.pth"
REID_WEIGHTS_TMP="${REID_WEIGHTS_DEST}.tmp"
REID_WEIGHTS_URL="https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
REID_MIN_BYTES=$((5 * 1024 * 1024))

file_size(){ wc -c < "$1" | tr -d ' '; }

export JOBS_DIR="${JOBS_DIR:-$PROJECTS_DIR/data/jobs}"
mkdir -p "$JOBS_DIR"

VENV_DIR="$PROJECTS_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
# Non-negotiable: use only venv python/pip for every install/check in this script.
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if [[ "$SHIFTCLIPPER_DEVICE" == cuda* ]]; then
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
else
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
fi

REQUIREMENTS_FILE="requirements.runpod_pro.txt"
echo "Installing ShiftClipper requirements from $REQUIREMENTS_FILE"
wc -l "$REQUIREMENTS_FILE"
if [ "$(wc -l < "$REQUIREMENTS_FILE")" -lt 5 ]; then
  echo "ERROR: requirements file malformed (no newlines). Fix in repo."
  head -c 500 "$REQUIREMENTS_FILE"
  exit 1
fi
"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"

"$PYTHON_BIN" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
"$PYTHON_BIN" -c "import ultralytics; print('ultralytics import ok')"

if ! "$PYTHON_BIN" -c "import torch, pkg_resources, ultralytics, cv2, redis, rq; print('deps ok')"; then
  echo "ERROR: dependency import check failed."
  tail -n 120 "$API_LOG" || true
  tail -n 120 "$WORKER_LOG" || true
  exit 1
fi

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
    echo "WARNING: ReID weights download returned HTML for $REID_WEIGHTS_URL; continuing without local ReID weights"
    rm -f "$REID_WEIGHTS_TMP"
  elif [ ! -s "$REID_WEIGHTS_TMP" ] || [ "$(file_size "$REID_WEIGHTS_TMP")" -le "$REID_MIN_BYTES" ]; then
    echo "WARNING: ReID weights download failed validation for $REID_WEIGHTS_URL; continuing without local ReID weights"
    rm -f "$REID_WEIGHTS_TMP"
  else
    mv "$REID_WEIGHTS_TMP" "$REID_WEIGHTS_DEST"
  fi
fi

if [ -s "$REID_WEIGHTS_DEST" ] && [ "$(file_size "$REID_WEIGHTS_DEST")" -gt "$REID_MIN_BYTES" ]; then
  "$PYTHON_BIN" -c "from worker.reid import OSNetEmbedder, ReIDConfig; import torch; e=OSNetEmbedder(ReIDConfig(model_name='osnet_x0_25', device='cuda:0' if torch.cuda.is_available() else 'cpu', batch_size=1, use_fp16=torch.cuda.is_available(), weights_path='$REID_WEIGHTS_DEST')); print('reid ok')"
fi

if ! "$PYTHON_BIN" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"; then
  echo "ERROR: Torch import check failed."
  tail -n 120 "$API_LOG" || true
  tail -n 120 "$WORKER_LOG" || true
  exit 1
fi
"$PYTHON_BIN" -c "import pkg_resources; print('pkg_resources', pkg_resources.__name__)"
if [[ "$SHIFTCLIPPER_DEVICE" == cuda* ]]; then
  "$PYTHON_BIN" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: SHIFTCLIPPER_DEVICE=$SHIFTCLIPPER_DEVICE but torch.cuda.is_available() is False."
    exit 1
  }
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
