#!/usr/bin/env bash
set -euo pipefail

API_LOG="/workspace/api.log"
WORKER_LOG="/workspace/worker.log"

export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export RQ_QUEUES="${RQ_QUEUES:-jobs}"
export SHIFTCLIPPER_DEVICE="${SHIFTCLIPPER_DEVICE:-cuda:0}"

REPO_URL="https://github.com/roches08/shiftclipper.git"
REPO_DIR="/workspace/shiftclipper"
PROJECTS_DIR="$REPO_DIR/Projects"

REQUIREMENTS_FILE="requirements.runpod_pro.txt"

REID_WEIGHTS_DIR="$PROJECTS_DIR/models/reid"
REID_WEIGHTS_DEST="$REID_WEIGHTS_DIR/osnet_x0_25_msmt17.pth"
REID_WEIGHTS_TMP="${REID_WEIGHTS_DEST}.tmp"
REID_WEIGHTS_URL="https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"

# If HF "resolve" URL is used, add download=true (helps avoid HTML/redirect edge cases)
if [[ "$REID_WEIGHTS_URL" == https://huggingface.co/*/resolve/* ]] && [[ "$REID_WEIGHTS_URL" != *\?* ]]; then
  REID_WEIGHTS_URL="${REID_WEIGHTS_URL}?download=true"
fi

file_size() { wc -c < "$1" | tr -d ' '; }

echo "=== [1/9] System packages ==="
apt-get update -y
# redis-cli lives in redis-tools on Ubuntu
apt-get install -y ffmpeg redis-server redis-tools python3-venv python3-pip curl git ca-certificates

echo "=== [2/9] Repo sync ==="
if [ ! -d "$REPO_DIR/.git" ]; then
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch --all --prune
  git -C "$REPO_DIR" reset --hard origin/main
fi

if [ ! -d "$PROJECTS_DIR" ]; then
  echo "ERROR: Projects directory missing at $PROJECTS_DIR"
  exit 1
fi
cd "$PROJECTS_DIR"

echo "=== [3/9] Verify committed UI assets ==="
# IMPORTANT: This enforces “UI must exist” and will fail fast if someone removed it from repo.
STATIC_DIR="$PROJECTS_DIR/static"
STATIC_INDEX="$STATIC_DIR/index.html"
STATIC_APP="$STATIC_DIR/app.js"
STATIC_PRESETS="$STATIC_DIR/presets.js"

if [ ! -f "$STATIC_INDEX" ] || [ ! -f "$STATIC_APP" ] || [ ! -f "$STATIC_PRESETS" ]; then
  echo "ERROR: Missing committed UI assets in $STATIC_DIR"
  echo "Expected:"
  echo "  - $STATIC_INDEX"
  echo "  - $STATIC_APP"
  echo "  - $STATIC_PRESETS"
  echo ""
  echo "This means the repo no longer contains the built UI in Projects/static."
  echo "Fix in GitHub (commit the static UI back) — do NOT hack startup to hide it."
  ls -lah "$STATIC_DIR" || true
  exit 1
fi
echo "UI static OK."

echo "=== [4/9] Python venv + deps ==="
VENV_DIR="$PROJECTS_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# Install torch based on device setting
if [[ "$SHIFTCLIPPER_DEVICE" == cuda* ]]; then
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
else
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "ERROR: requirements file missing: $REQUIREMENTS_FILE"
  exit 1
fi

# Guard: requirements file should have real lines (prevents “one-line blob” bug)
if [ "$(wc -l < "$REQUIREMENTS_FILE")" -lt 5 ]; then
  echo "ERROR: requirements file malformed (too few lines): $REQUIREMENTS_FILE"
  head -c 500 "$REQUIREMENTS_FILE" || true
  exit 1
fi

"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"

# Torchreid imports tensorboard in many code paths; ensure it's present even if not pinned
"$PYTHON_BIN" -m pip install -U tensorboard yacs gdown

echo "=== [5/9] Core import sanity checks (fatal) ==="
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$PYTHON_BIN" -c "import ultralytics, cv2, redis, rq; print('core deps ok')"

echo "=== [6/9] ReID bootstrap (NON-FATAL) ==="
mkdir -p "$REID_WEIGHTS_DIR"

# Download weights if missing. Validation is done via torch.load (stronger than byte-count).
if [ ! -s "$REID_WEIGHTS_DEST" ]; then
  echo "ReID weights missing; downloading -> $REID_WEIGHTS_DEST"
  rm -f "$REID_WEIGHTS_TMP"
  if ! curl -L --fail --retry 3 --retry-delay 2 "$REID_WEIGHTS_URL" -o "$REID_WEIGHTS_TMP"; then
    echo "WARNING: ReID weights download failed. ReID will be disabled (UI/API still start)."
    rm -f "$REID_WEIGHTS_TMP" || true
  else
    # quick HTML check
    if head -c 200 "$REID_WEIGHTS_TMP" | grep -qi "<html"; then
      echo "WARNING: ReID download returned HTML. ReID will be disabled (UI/API still start)."
      rm -f "$REID_WEIGHTS_TMP" || true
    else
      mv "$REID_WEIGHTS_TMP" "$REID_WEIGHTS_DEST"
    fi
  fi
fi

# Validate torchreid + weights without killing startup.
REID_OK=0
if [ -s "$REID_WEIGHTS_DEST" ]; then
  if "$PYTHON_BIN" - <<PY
import sys
from pathlib import Path
p = Path("$REID_WEIGHTS_DEST")
if not p.exists() or p.stat().st_size < 1024*1024:
    print("too small/missing")
    sys.exit(2)
with p.open("rb") as f:
    header = f.read(200).lower()
if b"<html" in header:
    print("html header")
    sys.exit(3)

# Validate that torch can actually load it
import torch
ckpt = torch.load(str(p), map_location="cpu")
# ckpt can be dict or state_dict; either is OK — just loading successfully is the key check
print("weights load ok", p.stat().st_size)
sys.exit(0)
PY
  then
    # Now check torchreid import (this has been breaking because of tensorboard)
    if "$PYTHON_BIN" -c "import tensorboard, torchreid; print('torchreid import ok')"; then
      REID_OK=1
    else
      echo "WARNING: torchreid import failed (even after tensorboard install). ReID disabled; UI/API continue."
      REID_OK=0
    fi
  else
    echo "WARNING: ReID weights validation failed. ReID disabled; UI/API continue."
    REID_OK=0
  fi
else
  echo "WARNING: ReID weights not present. ReID disabled; UI/API continue."
  REID_OK=0
fi

# Optional: instantiate your embedder if both torchreid + weights are OK (still non-fatal)
if [ "$REID_OK" -eq 1 ]; then
  if ! "$PYTHON_BIN" - <<PY
from worker.reid import OSNetEmbedder, ReIDConfig
import torch
cfg = ReIDConfig(
    model_name="osnet_x0_25",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    batch_size=1,
    use_fp16=bool(torch.cuda.is_available()),
    weights_path="$REID_WEIGHTS_DEST",
)
OSNetEmbedder(cfg)
print("reid embedder init ok")
PY
  then
    echo "WARNING: ReID embedder init failed. ReID disabled; UI/API continue."
    REID_OK=0
  fi
fi

echo "=== [7/9] Redis ==="
if ! redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
  redis-server --daemonize yes
fi
redis-cli -u "$REDIS_URL" ping >/dev/null

echo "=== [8/9] Start API ==="
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

echo "=== [9/9] Start worker ==="
nohup "$VENV_DIR/bin/python" -m worker.main > "$WORKER_LOG" 2>&1 &
WORKER_PID=$!

sleep 2
if ! kill -0 "$WORKER_PID" >/dev/null 2>&1; then
  echo "ERROR: worker exited"
  tail -n 200 "$WORKER_LOG" || true
  exit 1
fi

echo ""
echo "✅ Startup complete"
echo "API PID: $API_PID"
echo "Worker PID: $WORKER_PID"
echo "SHIFTCLIPPER_DEVICE=$SHIFTCLIPPER_DEVICE"
echo "ReID: $([ "$REID_OK" -eq 1 ] && echo ENABLED || echo DISABLED)"
