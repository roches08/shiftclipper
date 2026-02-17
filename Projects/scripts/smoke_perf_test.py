#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tracker smoke perf test and validate perf + early lock signals")
    parser.add_argument("video", nargs="?", help="Path to sample video")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--python", default="/workspace/shiftclipper/Projects/.venv/bin/python")
    args = parser.parse_args()

    if not args.video:
        print("SKIP: pass a sample video path")
        return 0
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    cmd = [
        args.python,
        "Projects/scripts/eval_tracker.py",
        str(video_path),
        "--base",
        args.base,
        "--camera-mode",
        "broadcast",
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)

    if data.get("status") not in {"done", "done_no_shifts"}:
        raise RuntimeError(f"tracking failed: {data.get('status')}")

    job_id = data.get("job_id")
    if not job_id:
        raise AssertionError("missing job id")

    job_dir = Path("Projects/data/jobs") / job_id
    results_file = job_dir / "results.json"
    if not results_file.exists():
        raise AssertionError(f"missing results file: {results_file}")

    results = json.loads(results_file.read_text())
    print("perf_summary", json.dumps(results.get("perf") or {}, indent=2))

    timeline_file = job_dir / "debug_timeline.json"
    if not timeline_file.exists():
        raise AssertionError(f"missing timeline file: {timeline_file}")
    timeline = json.loads(timeline_file.read_text())
    first_lock = next((ev.get("t") for ev in timeline if ev.get("event") == "lock"), None)
    print("first_lock_timestamp", first_lock)

    perf = data.get("perf") or {}
    print("perf_summary", json.dumps(perf, indent=2))
    clips = data.get("timestamps") or []
    print("clips_count", len(clips))
    print("clips_ranges", clips)

    if not perf:
        raise AssertionError("missing perf block in results")

    debug_overlay_path = data.get("debug_overlay_path")
    if debug_overlay_path and not Path(debug_overlay_path).exists():
        raise AssertionError("debug overlay path listed but missing")

    timeline_paths = [p for p in data.get("artifact_paths") or [] if p]
    for p in timeline_paths:
        if not Path(p).exists():
            raise AssertionError(f"artifact missing: {p}")

    if len(clips) == 0:
        raise AssertionError("expected at least one clip in smoke test")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
