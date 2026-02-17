#!/usr/bin/env python3
import argparse
import json
import time

import requests


TERMINAL = {"done", "failed", "cancelled", "verified", "done_no_clips"}


def main():
    ap = argparse.ArgumentParser(description="Run tracker evaluation against a single video/job")
    ap.add_argument("video")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--target-number", default="5")
    ap.add_argument("--jersey-color", default="#1d3936")
    ap.add_argument("--camera-mode", default="broadcast")
    ap.add_argument("--tracking-mode", default="clip")
    ap.add_argument("--detect-stride", type=int)
    ap.add_argument("--ocr-min-conf", type=float)
    ap.add_argument("--min-track-seconds", type=float)
    ap.add_argument("--gap-merge-seconds", type=float)
    ap.add_argument("--lock-seconds-after-confirm", type=float)
    ap.add_argument("--color-tolerance", type=int)
    ap.add_argument("--verify-mode", action="store_true")
    args = ap.parse_args()

    b = args.base
    jid = requests.post(f"{b}/jobs", json={"name": "eval"}, timeout=20).json()["job_id"]
    with open(args.video, "rb") as f:
        requests.post(f"{b}/jobs/{jid}/upload", files={"file": ("input.mp4", f, "video/mp4")}, timeout=120).raise_for_status()

    setup = {
        "camera_mode": args.camera_mode,
        "tracking_mode": args.tracking_mode,
        "player_number": args.target_number,
        "jersey_color": args.jersey_color,
        "jersey_color_hex": args.jersey_color,
        "verify_mode": args.verify_mode,
        "debug_overlay": True,
        "debug_timeline": True,
    }
    for k in ["detect_stride", "ocr_min_conf", "min_track_seconds", "gap_merge_seconds", "lock_seconds_after_confirm", "color_tolerance"]:
        v = getattr(args, k)
        if v is not None:
            setup[k] = v

    requests.put(f"{b}/jobs/{jid}/setup", json=setup, timeout=30).raise_for_status()
    requests.post(f"{b}/jobs/{jid}/run", timeout=20).raise_for_status()

    while True:
        s = requests.get(f"{b}/jobs/{jid}/status", timeout=20).json()
        if s.get("status") in TERMINAL:
            break
        time.sleep(1)

    res = requests.get(f"{b}/jobs/{jid}/results", timeout=20).json()
    artifacts = res.get("artifacts") or {}
    clips = artifacts.get("clips") or []
    print(json.dumps({
        "job_id": jid,
        "status": res.get("status"),
        "clips_count": len(clips),
        "timestamps": [{"start": c.get("start"), "end": c.get("end")} for c in clips],
        "artifact_paths": [c.get("path") for c in clips],
        "combined_path": artifacts.get("combined_path"),
        "debug_overlay_path": artifacts.get("debug_overlay_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
