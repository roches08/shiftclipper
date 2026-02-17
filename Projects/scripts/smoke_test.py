#!/usr/bin/env python3
import json
import subprocess
import sys
import time
from pathlib import Path
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def req(method, path, data=None):
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(BASE + path, method=method, data=body, headers=headers)
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_terminal(job_id):
    for _ in range(120):
        s = req("GET", f"/jobs/{job_id}/status")
        if s.get("status") in {"done", "failed", "cancelled", "verified", "done_no_clips"}:
            return s
        time.sleep(1)
    raise RuntimeError("timed out")


def run_mode(tracking_mode: str) -> None:
    job = req("POST", "/jobs", {"name": f"smoke-{tracking_mode}"})
    job_id = job["job_id"]
    job_dir = Path(f"Projects/data/jobs/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    in_path = job_dir / "in.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=15", "-t", "3", str(in_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    req("PUT", f"/jobs/{job_id}/setup", {
        "verify_mode": False,
        "tracking_mode": tracking_mode,
        "camera_mode": "broadcast",
        "player_number": "5",
        "jersey_color": "#1d3936",
        "debug_timeline": True,
    })
    req("POST", f"/jobs/{job_id}/run")
    st = wait_terminal(job_id)
    if st.get("status") == "failed":
        raise RuntimeError(f"job failed: {st}")

    res = req("GET", f"/jobs/{job_id}/results")
    artifacts = res.get("artifacts") or {}
    for c in artifacts.get("clips", []):
        assert Path(c["path"]).exists(), f"missing clip file: {c['path']}"
    if artifacts.get("combined_path"):
        assert Path(artifacts["combined_path"]).exists(), "combined listed but file missing"
    assert "clips" in artifacts, "artifacts.clips missing"
    if tracking_mode == "shift":
        assert res.get("shift_count", 0) > 0, "shift_count must be > 0"
        assert res.get("total_toi") is not None, "total_toi missing"


def main() -> int:
    run_mode("clip")
    run_mode("shift")
    print("smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
