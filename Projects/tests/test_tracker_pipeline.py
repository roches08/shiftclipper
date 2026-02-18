import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from worker.tasks import _clamp_segment_window, _compute_clip_end_for_loss, _compute_seed_clip_window, _point_in_polygon


def test_point_in_polygon_basic_square():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert _point_in_polygon((5.0, 5.0), square)
    assert not _point_in_polygon((15.0, 5.0), square)


def test_lost_timeout_clip_end_guard_uses_last_seen_not_video_end():
    end = _compute_clip_end_for_loss(last_seen_time=12.0, loss_timeout=1.5, video_end_time=100.0)
    assert end == 13.5


def test_compute_seed_clip_window_does_not_push_seed_after_click():
    win = _compute_seed_clip_window(
        click_t=6.16429,
        seed_lock_seconds=8.0,
        min_clip_seconds=1.0,
        video_duration=417.63,
        last_clip_end=417.788,
    )
    assert win["final_start"] == 0.0
    assert win["final_end"] == 6.16429


def test_clamp_segment_window_caps_to_video_duration():
    clamped = _clamp_segment_window(417.788, 418.788, 417.63)
    assert clamped is None
