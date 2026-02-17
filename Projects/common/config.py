import os
from typing import Any, Dict

CAMERA_PRESETS = {
    "broadcast": {
        "detect_stride": 1,
        "ocr_min_conf": 0.22,
        "lock_seconds_after_confirm": 4.0,
        "gap_merge_seconds": 2.5,
        "lost_timeout": 1.5,
        "min_track_seconds": 0.75,
        "color_weight": 0.35,
    },
    "broadcast_wide": {
        "detect_stride": 1,
        "ocr_min_conf": 0.20,
        "lock_seconds_after_confirm": 6.0,
        "gap_merge_seconds": 3.0,
        "lost_timeout": 1.9,
        "min_track_seconds": 0.75,
        "color_weight": 0.5,
    },
    "tactical": {
        "detect_stride": 2,
        "ocr_min_conf": 0.25,
        "lock_seconds_after_confirm": 5.0,
        "gap_merge_seconds": 2.0,
        "lost_timeout": 1.8,
        "min_track_seconds": 0.75,
        "color_weight": 0.6,
    },
}


def resolve_device() -> str:
    override = os.getenv("SHIFTCLIPPER_DEVICE")
    if override:
        return override
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def normalize_setup(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    src = payload or {}
    camera_mode = str(src.get("camera_mode") or "broadcast").lower()
    if camera_mode not in CAMERA_PRESETS:
        camera_mode = "broadcast"

    tracking_mode = str(src.get("tracking_mode") or "clip").lower()
    if tracking_mode not in {"clip", "shift"}:
        tracking_mode = "clip"

    preset = CAMERA_PRESETS[camera_mode]
    verify_mode = bool(src.get("verify_mode", False))
    extend_sec = max(0.0, min(60.0, float(src.get("extend_sec") or 20.0)))

    clicks = []
    for raw in (src.get("clicks") or []):
        try:
            clicks.append({
                "t": max(0.0, float(raw.get("t", 0.0))),
                "x": max(0.0, min(1.0, float(raw.get("x", 0.0)))),
                "y": max(0.0, min(1.0, float(raw.get("y", 0.0)))),
            })
        except Exception:
            continue

    setup = {
        "camera_mode": camera_mode,
        "tracking_mode": tracking_mode,
        "player_number": str(src.get("player_number") or "").strip(),
        "jersey_color": str(src.get("jersey_color") or "#203524"),
        "opponent_color": str(src.get("opponent_color") or "#ffffff"),
        "extend_sec": extend_sec,
        "post_roll": float(src.get("post_roll") or extend_sec),
        "verify_mode": verify_mode,
        "clicks": clicks,
        "clicks_count": len(clicks),
        "detect_stride": int(src.get("detect_stride") or preset["detect_stride"]),
        "ocr_min_conf": float(src.get("ocr_min_conf") or preset["ocr_min_conf"]),
        "lock_seconds_after_confirm": float(src.get("lock_seconds_after_confirm") or preset["lock_seconds_after_confirm"]),
        "gap_merge_seconds": float(src.get("gap_merge_seconds") or preset["gap_merge_seconds"]),
        "lost_timeout": float(src.get("lost_timeout") or preset["lost_timeout"]),
        "min_track_seconds": float(src.get("min_track_seconds") or preset["min_track_seconds"]),
        "color_weight": float(src.get("color_weight") or preset["color_weight"]),
        "motion_weight": float(src.get("motion_weight") or 0.3),
        "ocr_weight": float(src.get("ocr_weight") or 0.35),
        "identity_weight": float(src.get("identity_weight") or 0.5),
        "color_tolerance": int(src.get("color_tolerance") or 26),
        "bench_zone_ratio": float(src.get("bench_zone_ratio") or 0.8),
        "generate_combined": bool(src.get("generate_combined", True)),
        "debug_overlay": bool(src.get("debug_overlay", False)),
        "debug_timeline": bool(src.get("debug_timeline", True)),
        "ocr_confirm_m": int(src.get("ocr_confirm_m") or 2),
        "ocr_confirm_k": int(src.get("ocr_confirm_k") or 5),
    }
    return setup
