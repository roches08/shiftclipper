import os
from typing import Any, Dict, Tuple

CAMERA_PRESETS = {
    "broadcast": {
        "detect_stride": 1,
        "ocr_min_conf": 0.20,
        "lock_seconds_after_confirm": 4.0,
        "gap_merge_seconds": 2.5,
        "lost_timeout": 1.5,
        "min_track_seconds": 0.75,
        "color_weight": 0.35,
    },
    "broadcast_wide": {
        "detect_stride": 1,
        "ocr_min_conf": 0.18,
        "lock_seconds_after_confirm": 6.0,
        "gap_merge_seconds": 3.0,
        "lost_timeout": 1.9,
        "min_track_seconds": 0.75,
        "color_weight": 0.5,
    },
    "tactical": {
        "detect_stride": 3,
        "ocr_min_conf": 0.30,
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


def _as_float(src: Dict[str, Any], key: str, default: float) -> float:
    val = src.get(key, default)
    try:
        return float(default if val is None else val)
    except Exception:
        return default


def _as_int(src: Dict[str, Any], key: str, default: int) -> int:
    val = src.get(key, default)
    try:
        return int(default if val is None else val)
    except Exception:
        return default


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    c = (hex_color or "").strip().lstrip("#")
    if len(c) != 6:
        return (32, 53, 36)
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


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
    extend_sec = max(0.0, min(60.0, _as_float(src, "extend_sec", 20.0)))

    jersey_color_hex = str(src.get("jersey_color_hex") or src.get("jersey_color") or "#203524")
    jr, jg, jb = _hex_to_rgb(jersey_color_hex)
    jersey_color_rgb = src.get("jersey_color_rgb") or {"r": jr, "g": jg, "b": jb}

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
        "jersey_color": jersey_color_hex,
        "jersey_color_hex": jersey_color_hex,
        "jersey_color_rgb": jersey_color_rgb,
        "opponent_color": str(src.get("opponent_color") or "#ffffff"),
        "extend_sec": extend_sec,
        "post_roll": _as_float(src, "post_roll", extend_sec),
        "verify_mode": verify_mode,
        "skip_seeding": bool(src.get("skip_seeding", False)),
        "clicks": clicks,
        "clicks_count": len(clicks),
        "detect_stride": _as_int(src, "detect_stride", preset["detect_stride"]),
        "ocr_min_conf": _as_float(src, "ocr_min_conf", preset["ocr_min_conf"]),
        "lock_seconds_after_confirm": _as_float(src, "lock_seconds_after_confirm", preset["lock_seconds_after_confirm"]),
        "gap_merge_seconds": _as_float(src, "gap_merge_seconds", preset["gap_merge_seconds"]),
        "lost_timeout": _as_float(src, "lost_timeout", preset["lost_timeout"]),
        "min_track_seconds": _as_float(src, "min_track_seconds", preset["min_track_seconds"]),
        "color_weight": _as_float(src, "color_weight", preset["color_weight"]),
        "motion_weight": _as_float(src, "motion_weight", 0.3),
        "ocr_weight": _as_float(src, "ocr_weight", 0.35),
        "identity_weight": _as_float(src, "identity_weight", 0.5),
        "color_tolerance": _as_int(src, "color_tolerance", 26),
        "bench_zone_ratio": _as_float(src, "bench_zone_ratio", 0.8),
        "generate_combined": bool(src.get("generate_combined", True)),
        "debug_overlay": bool(src.get("debug_overlay", False)),
        "debug_timeline": bool(src.get("debug_timeline", True)),
        "ocr_confirm_m": _as_int(src, "ocr_confirm_m", 2),
        "ocr_confirm_k": _as_int(src, "ocr_confirm_k", 5),
    }
    return setup
