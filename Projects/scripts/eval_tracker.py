#!/usr/bin/env python3
import argparse
import json
import time
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--target-number", default="5")
    ap.add_argument("--jersey-color", default="#1d3936")
    ap.add_argument("--camera-mode", default="broadcast_wide")
    ap.add_argument("--tracking-mode", default="clip")
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
        "verify_mode": False,
        "debug_overlay": True,
        "debug_timeline": True,
    }
    requests.put(f"{b}/jobs/{jid}/setup", json=setup, timeout=30).raise_for_status()
    requests.post(f"{b}/jobs/{jid}/run", timeout=20).raise_for_status()

    while True:
        s = requests.get(f"{b}/jobs/{jid}/status", timeout=20).json()
        if s.get("status") in {"done", "failed", "cancelled", "verified", "done_no_clips"}:
            break
        time.sleep(1)

    res = requests.get(f"{b}/jobs/{jid}/results", timeout=20).json()
    print(json.dumps({
        "job_id": jid,
        "status": res.get("status"),
        "clips_count": len((res.get("artifacts") or {}).get("clips", [])),
        "shift_count": res.get("shift_count", 0),
        "shifts": res.get("shifts", []),
        "artifact_paths": [c.get("path") for c in (res.get("artifacts") or {}).get("clips", [])],
        "debug_overlay_path": (res.get("artifacts") or {}).get("debug_overlay_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
