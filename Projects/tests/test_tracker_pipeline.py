import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from worker.tasks import _compute_clip_end_for_loss, _point_in_polygon


def test_point_in_polygon_basic_square():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert _point_in_polygon((5.0, 5.0), square)
    assert not _point_in_polygon((15.0, 5.0), square)


def test_lost_timeout_clip_end_guard_uses_last_seen_not_video_end():
    end = _compute_clip_end_for_loss(last_seen_time=12.0, loss_timeout=1.5, video_end_time=100.0)
    assert end == 13.5
