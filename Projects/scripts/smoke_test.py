#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path


BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SAMPLE = os.getenv("SHIFTCLIPPER_SMOKE_VIDEO")
MIN_TOTAL_DURATION = float(os.getenv("SHIFTCLIPPER_SMOKE_MIN_DURATION", "1.0"))


def main() -> int:
    if not SAMPLE or not Path(SAMPLE).exists():
        print("SKIP: set SHIFTCLIPPER_SMOKE_VIDEO to a small labeled sample video")
        return 0

    cmd = [
        "python3",
        "Projects/scripts/eval_tracker.py",
        SAMPLE,
        "--base",
        BASE,
        "--target-number",
        "5",
        "--jersey-color",
        "#1d3936",
        "--camera-mode",
        "broadcast",
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)

    if data.get("status") != "done":
        raise RuntimeError(f"eval failed: {data}")

    clips = data.get("timestamps") or []
    total_duration = sum(max(0.0, float(c.get("end") or 0) - float(c.get("start") or 0)) for c in clips)
    if not (len(clips) >= 2 or (len(clips) == 1 and total_duration >= MIN_TOTAL_DURATION)):
        raise AssertionError(f"expected at least 2 clips or 1 merged clip >= {MIN_TOTAL_DURATION}s, got {clips}")

    for p in data.get("artifact_paths") or []:
        if not Path(p).exists():
            raise AssertionError(f"missing artifact listed in results.json: {p}")
    if data.get("combined_path") and not Path(data["combined_path"]).exists():
        raise AssertionError(f"missing combined artifact: {data['combined_path']}")

    print("smoke ok", json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
