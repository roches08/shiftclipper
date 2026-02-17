#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cv2

from worker.tasks import TrackingParams, cut_clip, concat_clips, track_presence_spans_pro, resolve_device


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--target-number", default="5")
    ap.add_argument("--jersey-color", default="#1d3936")
    ap.add_argument("--opponent-color", default="#ffffff")
    ap.add_argument("--mode", choices=["broadcast", "tactical"], default="broadcast")
    ap.add_argument("--out-dir", default="/tmp/shiftclipper_eval")
    ap.add_argument("--detect-stride", type=int, default=None)
    ap.add_argument("--verify-mode", action="store_true")
    args = ap.parse_args()

    mode_defaults = {
        "broadcast": dict(detect_stride=1, ocr_min_conf=0.28, min_track_seconds=1.2, gap_merge_seconds=1.0, lock_seconds_after_confirm=1.5),
        "tactical": dict(detect_stride=3, ocr_min_conf=0.42, min_track_seconds=1.0, gap_merge_seconds=0.8, lock_seconds_after_confirm=1.0),
    }
    md = mode_defaults[args.mode]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / "debug_overlay.mp4"

    params = TrackingParams(
        detect_stride=args.detect_stride or md["detect_stride"],
        ocr_min_conf=md["ocr_min_conf"],
        min_track_seconds=md["min_track_seconds"],
        gap_merge_seconds=md["gap_merge_seconds"],
        lock_seconds_after_confirm=md["lock_seconds_after_confirm"],
        debug_overlay=True,
        verify_mode=args.verify_mode,
        device=resolve_device(),
    )

    spans, debug = track_presence_spans_pro(
        video_path=args.video,
        clicks=[],
        player_number=args.target_number,
        jersey_color_hex=args.jersey_color,
        opponent_color_hex=args.opponent_color,
        params=params,
        debug_overlay_path=str(overlay_path),
    )

    clip_paths = []
    for i, (a, b) in enumerate(spans, start=1):
        p = clips_dir / f"clip_{i:03d}.mp4"
        cut_clip(args.video, a, b, str(p))
        clip_paths.append(str(p))

    combined_path = out_dir / "combined.mp4"
    concat_clips(clip_paths, str(combined_path))

    results = {
        "clips_count": len(spans),
        "timestamps": spans,
        "debug_overlay_path": str(overlay_path),
        "clips": clip_paths,
        "combined": str(combined_path) if combined_path.exists() else None,
        "artifacts": [{"type": "clip", "path": p} for p in clip_paths],
    }
    if combined_path.exists():
        results["artifacts"].append({"type": "combined", "path": str(combined_path)})
    if overlay_path.exists():
        results["artifacts"].append({"type": "debug_overlay", "path": str(overlay_path)})
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
