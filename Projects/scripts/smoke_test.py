#!/usr/bin/env python3
import json
import subprocess
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def req(method, path, data=None):
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(BASE + path, method=method, data=body, headers=headers)
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    print("healthz:", req("GET", "/healthz"))
    print("readyz:", req("GET", "/readyz"))
    job = req("POST", "/jobs", {"name": "smoke"})
    job_id = job["job_id"]
    job_dir = f"Projects/data/jobs/{job_id}"
    subprocess.run(["mkdir", "-p", job_dir], check=True)
    in_path = f"{job_dir}/in.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15", "-t", "2", in_path
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    req("PUT", f"/jobs/{job_id}/setup", {"verify_mode": True, "clicks": []})
    status = req("POST", f"/jobs/{job_id}/run")
    print("run:", status)

    final = None
    for _ in range(30):
      time.sleep(1)
      s = req("GET", f"/jobs/{job_id}/status")
      print("status:", s)
      if s.get("status") in {"done", "failed", "cancelled"}:
        final = s
        break

    if final is None:
      print("Timed out waiting for terminal status")
      return 1

    if final.get("status") != "done":
      print("Job failed:", final)
      return 1

    results = req("GET", f"/jobs/{job_id}/results")
    print("results:", results)
    manifest = results.get("manifest") or {}
    if manifest.get("status") != "done":
      print("Manifest not finalized:", manifest)
      return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
