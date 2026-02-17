#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import requests


BASE = os.getenv("SHIFTCLIPPER_BASE_URL", "http://127.0.0.1:8000")
SAMPLE = os.getenv("SHIFTCLIPPER_SMOKE_VIDEO")
TIMEOUT_S = int(os.getenv("SHIFTCLIPPER_SMOKE_TIMEOUT", "900"))
TERMINAL = {"done", "failed", "cancelled", "verified", "done_no_clips", "done_no_shifts"}


def _assert_artifacts(res: dict) -> None:
    artifacts = res.get("artifacts") or {}
    clip_paths = [c.get("path") for c in (artifacts.get("clips") or []) if c.get("path")]
    for clip_path in clip_paths:
        if not Path(clip_path).exists():
            raise AssertionError(f"missing clip artifact: {clip_path}")

    combined_path = artifacts.get("combined_path") or res.get("combined_path")
    if combined_path and not Path(combined_path).exists():
        raise AssertionError(f"missing combined artifact: {combined_path}")


def main() -> int:
    if not SAMPLE or not Path(SAMPLE).exists():
        print("SKIP: set SHIFTCLIPPER_SMOKE_VIDEO to a small sample file")
        return 0

    job_id = requests.post(f"{BASE}/jobs", json={"name": "runpod-smoke"}, timeout=20).json()["job_id"]
    print("job_id", job_id)

    with open(SAMPLE, "rb") as f:
        requests.post(
            f"{BASE}/jobs/{job_id}/upload",
            files={"file": (Path(SAMPLE).name, f, "video/mp4")},
            timeout=180,
        ).raise_for_status()

    setup = {
        "camera_mode": "broadcast",
        "tracking_mode": "clip",
        "player_number": "5",
        "jersey_color": "#1d3936",
        "detect_stride": 2,
        "ocr_every_n_frames": 10,
        "ocr_max_crops_per_frame": 1,
        "yolo_imgsz": 512,
        "yolo_batch": 4,
        "debug_timeline": True,
    }
    requests.put(f"{BASE}/jobs/{job_id}/setup", json=setup, timeout=30).raise_for_status()
    requests.post(f"{BASE}/jobs/{job_id}/run", timeout=20).raise_for_status()

    started = time.time()
    status = {}
    while True:
        status = requests.get(f"{BASE}/jobs/{job_id}/status", timeout=20).json()
        if status.get("status") in TERMINAL:
            break
        if time.time() - started > TIMEOUT_S:
            raise TimeoutError(f"smoke test timeout after {TIMEOUT_S}s, last status={status}")
        time.sleep(2)

    res = requests.get(f"{BASE}/jobs/{job_id}/results", timeout=20).json()
    perf = res.get("perf") or {}

    timeline = []
    debug_timeline_path = ((res.get("artifacts") or {}).get("debug_timeline_path"))
    if debug_timeline_path and Path(debug_timeline_path).exists():
        timeline = json.loads(Path(debug_timeline_path).read_text(encoding="utf-8"))
    first_lock = next((ev.get("t") for ev in timeline if ev.get("event") in {"present_on", "seed_lock"}), None)

    _assert_artifacts(res)

    print("status", res.get("status"))
    print("perf_summary", json.dumps(perf, indent=2))
    print("first_locked_timestamp", first_lock)
    print("artifacts_ok", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
