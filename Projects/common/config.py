import os
from typing import Any, Dict, Tuple

CAMERA_PRESETS = {
    "broadcast": {
        "detect_stride": 2,
        "yolo_imgsz": 512,
        "ocr_every_n": 12,
        "ocr_max_crops_per_frame": 1,
        "ocr_min_conf": 0.20,
        "lock_seconds_after_confirm": 4.0,
        "gap_merge_seconds": 2.5,
        "lost_timeout": 1.5,
        "min_track_seconds": 0.75,
        "color_weight": 0.35,
    },
    "broadcast_wide": {
        "detect_stride": 3,
        "yolo_imgsz": 512,
        "ocr_every_n": 12,
        "ocr_max_crops_per_frame": 1,
        "ocr_min_conf": 0.18,
        "lock_seconds_after_confirm": 6.0,
        "gap_merge_seconds": 3.0,
        "lost_timeout": 1.9,
        "min_track_seconds": 0.75,
        "color_weight": 0.5,
    },
    "tactical": {
        "detect_stride": 4,
        "yolo_imgsz": 416,
        "ocr_every_n": 999999,
        "ocr_max_crops_per_frame": 1,
        "ocr_min_conf": 0.30,
        "lock_seconds_after_confirm": 5.0,
        "gap_merge_seconds": 2.0,
        "lost_timeout": 1.8,
        "min_track_seconds": 0.75,
        "color_weight": 0.6,
    },
}


def resolve_device() -> Tuple[str, int | str]:
    override = os.getenv("SHIFTCLIPPER_DEVICE")
    if override:
        if override.startswith("cuda"):
            return override, 0
        return override, "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0", 0
        return "cpu", "cpu"
    except Exception:
        return "cpu", "cpu"


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


def _clamp_float(src: Dict[str, Any], key: str, default: float, min_v: float | None = None, max_v: float | None = None) -> float:
    v = _as_float(src, key, default)
    if min_v is not None:
        v = max(min_v, v)
    if max_v is not None:
        v = min(max_v, v)
    return v


def _clamp_int(src: Dict[str, Any], key: str, default: int, min_v: int | None = None, max_v: int | None = None) -> int:
    v = _as_int(src, key, default)
    if min_v is not None:
        v = max(min_v, v)
    if max_v is not None:
        v = min(max_v, v)
    return v


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

    mode_defaults = {
        "clip": {
            "score_lock_threshold": 0.55,
            "score_unlock_threshold": 0.35,
            "lost_timeout": 1.5,
            "reacquire_window_seconds": 4.0,
            "reacquire_score_lock_threshold": 0.40,
        },
        "shift": {
            "score_lock_threshold": 0.55,
            "score_unlock_threshold": 0.35,
            "lost_timeout": 1.5,
            "reacquire_window_seconds": 4.0,
            "reacquire_score_lock_threshold": 0.40,
        },
    }[tracking_mode]

    tracker_type = str(src.get("tracker_type") or "bytetrack").lower()
    if tracker_type not in {"bytetrack", "iou", "deepsort"}:
        tracker_type = "bytetrack"

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
        "detect_stride": _clamp_int(src, "detect_stride", preset["detect_stride"], min_v=1),
        "yolo_imgsz": _clamp_int(src, "yolo_imgsz", preset["yolo_imgsz"], min_v=256, max_v=1280),
        "ocr_every_n": _clamp_int(src, "ocr_every_n", preset["ocr_every_n"], min_v=1),
        "ocr_every_n_frames": _clamp_int(src, "ocr_every_n_frames", _as_int(src, "ocr_every_n", preset["ocr_every_n"]), min_v=1),
        "ocr_max_crops_per_frame": _as_int(src, "ocr_max_crops_per_frame", preset.get("ocr_max_crops_per_frame", 1)),
        "ocr_disable": bool(src.get("ocr_disable", False)),
        "yolo_batch": _clamp_int(src, "yolo_batch", 4, min_v=1),
        "tracker_type": tracker_type,
        "ocr_min_conf": _clamp_float(src, "ocr_min_conf", preset["ocr_min_conf"], min_v=0.0, max_v=1.0),
        "lock_seconds_after_confirm": _clamp_float(src, "lock_seconds_after_confirm", preset["lock_seconds_after_confirm"], min_v=0.0),
        "gap_merge_seconds": _clamp_float(src, "gap_merge_seconds", 2.5, min_v=0.0),
        "lost_timeout": _clamp_float(src, "lost_timeout", mode_defaults["lost_timeout"], min_v=0.0),
        "reacquire_window_seconds": _clamp_float(src, "reacquire_window_seconds", mode_defaults["reacquire_window_seconds"], min_v=0.0),
        "reacquire_score_lock_threshold": _clamp_float(src, "reacquire_score_lock_threshold", mode_defaults["reacquire_score_lock_threshold"], min_v=0.0, max_v=1.0),
        "min_track_seconds": _clamp_float(src, "min_track_seconds", preset["min_track_seconds"], min_v=0.0),
        "min_clip_seconds": _clamp_float(src, "min_clip_seconds", 1.0, min_v=0.0),
        "score_lock_threshold": _clamp_float(src, "score_lock_threshold", mode_defaults["score_lock_threshold"], min_v=0.0, max_v=1.0),
        "score_unlock_threshold": _clamp_float(src, "score_unlock_threshold", mode_defaults["score_unlock_threshold"], min_v=0.0, max_v=1.0),
        "seed_lock_seconds": _clamp_float(src, "seed_lock_seconds", 8.0, min_v=0.0),
        "seed_iou_min": _clamp_float(src, "seed_iou_min", _as_float(src, "seed_min_iou", 0.12), min_v=0.0, max_v=1.0),
        "seed_dist_max": _clamp_float(src, "seed_dist_max", 0.16, min_v=0.0, max_v=1.0),
        "seed_bonus": _clamp_float(src, "seed_bonus", 0.80, min_v=0.0),
        "seed_window_s": _clamp_float(src, "seed_window_s", _as_float(src, "seed_search_window_seconds", 3.0), min_v=0.0),
        "color_weight": _as_float(src, "color_weight", preset["color_weight"]),
        "motion_weight": _as_float(src, "motion_weight", 0.3),
        "ocr_weight": _as_float(src, "ocr_weight", 0.35),
        "identity_weight": _as_float(src, "identity_weight", 0.5),
        "color_tolerance": _clamp_int(src, "color_tolerance", 26, min_v=0),
        "bench_zone_ratio": _clamp_float(src, "bench_zone_ratio", 0.8, min_v=0.0, max_v=1.0),
        "generate_combined": bool(src.get("generate_combined", True)),
        "debug_overlay": bool(src.get("debug_overlay", False)),
        "debug_timeline": bool(src.get("debug_timeline", True)),
        "ocr_confirm_m": _as_int(src, "ocr_confirm_m", 2),
        "ocr_confirm_k": _as_int(src, "ocr_confirm_k", 5),
        "allow_unconfirmed_clips": bool(src.get("allow_unconfirmed_clips", False)),
        "allow_seed_clips": bool(src.get("allow_seed_clips", True)),
        "ocr_veto_conf": _clamp_float(src, "ocr_veto_conf", 0.85, min_v=0.0, max_v=1.0),
        "ocr_veto_seconds": _clamp_float(src, "ocr_veto_seconds", 2.0, min_v=0.0),
        "closeup_bbox_area_ratio": _as_float(src, "closeup_bbox_area_ratio", 0.18),
    }
    return setup
