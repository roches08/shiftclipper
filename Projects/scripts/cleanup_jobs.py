#!/usr/bin/env python3
"""Cleanup job artifacts by age and max count."""

import argparse
import shutil
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup ShiftClipper job artifacts")
    parser.add_argument("--jobs-dir", default="Projects/data/jobs")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max-count", type=int, default=200)
    args = parser.parse_args()

    jobs_dir = Path(args.jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    cutoff = time.time() - (max(0, args.days) * 86400)
    entries = [(p, p.stat().st_mtime) for p in jobs_dir.iterdir() if p.is_dir()]
    entries.sort(key=lambda x: x[1], reverse=True)

    removed = []
    for idx, (path, mtime) in enumerate(entries):
        if idx >= args.max_count or mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)

    print({"ok": True, "removed": removed, "kept": max(0, len(entries) - len(removed))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
