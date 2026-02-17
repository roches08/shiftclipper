#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


def main() -> int:
    out = Path("/tmp/shiftclipper_smoke")
    out.mkdir(parents=True, exist_ok=True)
    video = out / "sample.mp4"

    # Simple synthetic clip with 3 presence windows.
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=20:d=8",
        "-vf",
        "drawbox=enable='between(t,0.5,1.8)':x=220:y=80:w=60:h=140:color=green@0.95:t=fill,"
        "drawbox=enable='between(t,3.0,4.0)':x=260:y=90:w=60:h=140:color=green@0.95:t=fill,"
        "drawbox=enable='between(t,5.2,6.5)':x=300:y=95:w=60:h=140:color=green@0.95:t=fill",
        str(video),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    eval_dir = out / "eval"
    subprocess.run([
        "python", "Projects/scripts/eval_tracker.py",
        "--video", str(video),
        "--target-number", "5",
        "--jersey-color", "#00aa00",
        "--mode", "broadcast",
        "--out-dir", str(eval_dir),
    ], check=True)

    results = json.loads((eval_dir / "results.json").read_text())
    clips = results.get("timestamps", [])
    ok = len(clips) >= 2
    if not ok and len(clips) == 1:
        ok = (clips[0][1] - clips[0][0]) >= 2.0
    assert ok, f"unexpected clips output: {clips}"
    artifacts = results.get("artifacts", [])
    assert artifacts, "results.json must include artifacts"
    for item in artifacts:
        assert Path(item["path"]).exists(), f"missing artifact path {item['path']}"
    print(json.dumps({"clips": clips, "artifacts": len(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
