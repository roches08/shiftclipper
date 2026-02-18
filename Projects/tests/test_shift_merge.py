import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from worker.tasks import _merge_segments


def test_shift_merge_requires_small_gap_and_reacquire_window():
    segments = [
        (0.0, 5.0, "unlock"),
        (6.0, 12.0, "unlock"),
        (20.6, 30.0, "unlock"),
    ]

    merged = _merge_segments(
        segments,
        min_clip_seconds=1.0,
        gap_merge_seconds=1.5,
        tracking_mode="shift",
        reacquire_window_seconds=4.0,
    )

    assert merged == [(0.0, 12.0, "merged"), (20.6, 30.0, "unlock")]


def test_shift_merge_does_not_chain_entire_video():
    segments = [
        (0.0, 5.0, "unlock"),
        (6.0, 10.0, "unlock"),
        (11.0, 15.0, "unlock"),
    ]

    merged = _merge_segments(
        segments,
        min_clip_seconds=1.0,
        gap_merge_seconds=1.5,
        tracking_mode="shift",
        reacquire_window_seconds=4.0,
    )

    assert merged == [(0.0, 10.0, "merged"), (11.0, 15.0, "unlock")]


def test_clip_mode_merge_respects_lock_loss_boundary():
    segments = [
        (0.0, 5.0, "lock_lost"),
        (5.4, 9.0, "lost_timeout"),
    ]

    merged = _merge_segments(
        segments,
        min_clip_seconds=1.0,
        gap_merge_seconds=1.5,
        tracking_mode="clip",
        reacquire_window_seconds=4.0,
    )

    assert merged == segments


def test_shift_mode_merge_respects_lock_loss_boundary():
    segments = [
        (0.0, 5.0, "unlock"),
        (5.5, 8.0, "lock_lost"),
        (8.6, 12.0, "unlock"),
    ]

    merged = _merge_segments(
        segments,
        min_clip_seconds=1.0,
        gap_merge_seconds=1.5,
        tracking_mode="shift",
        reacquire_window_seconds=4.0,
    )

    assert merged == segments
