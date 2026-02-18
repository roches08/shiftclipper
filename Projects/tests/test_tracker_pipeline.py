import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from worker.tasks import (
    _clamp_segment_window,
    _compute_clip_end_for_loss,
    _compute_seed_clip_window,
    _point_in_polygon,
    _should_reject_for_reid,
    _swap_guard_allows_transition,
)


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


def test_reacquire_rejects_low_reid_even_with_mid_identity_score():
    locked_emb = np.array([1.0, 0.0], dtype=np.float32)
    assert _should_reject_for_reid(
        locked_emb=locked_emb,
        reid_sim=0.25,
        reid_min_sim=0.45,
        identity_score=0.7,
        proximity_score=0.7,
    )


def test_swap_guard_blocks_transition_when_reid_low():
    locked_emb = np.array([1.0, 0.0], dtype=np.float32)
    assert not _swap_guard_allows_transition(
        identity_score=0.8,
        threshold=0.7,
        locked_emb=locked_emb,
        reid_sim=0.2,
        reid_min_sim=0.45,
    )


def test_reid_disabled_does_not_initialize_embedder(tmp_path, monkeypatch):
    from worker import tasks as worker_tasks

    calls = {"count": 0}

    class _FakeCapture:
        def __init__(self, _):
            self.read_count = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == worker_tasks.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_COUNT:
                return 0
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_WIDTH:
                return 1920
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_HEIGHT:
                return 1080
            return 0

        def read(self):
            return False, None

        def release(self):
            return None

    class _FakeYOLO:
        def __init__(self, *_args, **_kwargs):
            pass

    def _fake_embedder(*_args, **_kwargs):
        calls["count"] += 1
        raise AssertionError("embedder should not be initialized when reid is disabled")

    if worker_tasks.cv2 is None:
        class _FakeCv2:
            CAP_PROP_FPS = 5
            CAP_PROP_FRAME_COUNT = 7
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            VideoCapture = _FakeCapture

        monkeypatch.setattr(worker_tasks, "cv2", _FakeCv2)
    else:
        monkeypatch.setattr(worker_tasks.cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(worker_tasks, "YOLO", _FakeYOLO)
    monkeypatch.setattr(worker_tasks, "resolve_device", lambda: ("cuda:0", 0))
    monkeypatch.setattr(worker_tasks, "warm_easyocr_models", lambda *_: None)
    monkeypatch.setattr(worker_tasks, "_build_ocr_reader", lambda *_: (None, False))
    monkeypatch.setattr(worker_tasks, "_hex_to_hsv", lambda *_: (60, 80, 80))
    monkeypatch.setattr(worker_tasks, "OSNetEmbedder", _fake_embedder)

    out = worker_tasks.track_presence(str(tmp_path / "dummy.mp4"), {"reid_enable": False, "use_reid": False, "debug_timeline": False})

    assert calls["count"] == 0
    assert isinstance(out, dict)


def test_reid_init_failure_disables_and_continues_when_policy_disable(tmp_path, monkeypatch):
    from worker import tasks as worker_tasks

    class _FakeCapture:
        def __init__(self, _):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == worker_tasks.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_COUNT:
                return 0
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_WIDTH:
                return 1920
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_HEIGHT:
                return 1080
            return 0

        def read(self):
            return False, None

        def release(self):
            return None

    class _FakeYOLO:
        def __init__(self, *_args, **_kwargs):
            pass

    def _broken_embedder(*_args, **_kwargs):
        raise RuntimeError("embedder init failed")

    if worker_tasks.cv2 is None:
        class _FakeCv2:
            CAP_PROP_FPS = 5
            CAP_PROP_FRAME_COUNT = 7
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            VideoCapture = _FakeCapture

        monkeypatch.setattr(worker_tasks, "cv2", _FakeCv2)
    else:
        monkeypatch.setattr(worker_tasks.cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(worker_tasks, "YOLO", _FakeYOLO)
    monkeypatch.setattr(worker_tasks, "resolve_device", lambda: ("cuda:0", 0))
    monkeypatch.setattr(worker_tasks, "warm_easyocr_models", lambda *_: None)
    monkeypatch.setattr(worker_tasks, "_build_ocr_reader", lambda *_: (None, False))
    monkeypatch.setattr(worker_tasks, "_hex_to_hsv", lambda *_: (60, 80, 80))
    monkeypatch.setattr(worker_tasks, "OSNetEmbedder", _broken_embedder)
    monkeypatch.setattr(worker_tasks, "ensure_reid_weights", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("download failed")))

    setup = {
        "reid_enable": True,
        "reid_fail_policy": "disable",
        "reid_auto_download": True,
        "reid_weights_path": str(tmp_path / "missing.pth"),
        "reid_weights_url": "https://example.com/osnet.pth",
        "debug_timeline": True,
    }
    out = worker_tasks.track_presence(str(tmp_path / "dummy.mp4"), setup)

    assert out["reid_disabled_due_to_error"] == "download failed"
    assert any(ev.get("event") == "reid_disabled_due_to_error" for ev in out["timeline"])
    assert setup.get("_runtime_reid_disabled") is True


def test_reid_init_failure_raises_when_policy_fail(tmp_path, monkeypatch):
    from worker import tasks as worker_tasks

    class _FakeCapture:
        def __init__(self, _):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == worker_tasks.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_COUNT:
                return 0
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_WIDTH:
                return 1920
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_HEIGHT:
                return 1080
            return 0

        def read(self):
            return False, None

        def release(self):
            return None

    class _FakeYOLO:
        def __init__(self, *_args, **_kwargs):
            pass

    def _broken_embedder(*_args, **_kwargs):
        raise RuntimeError("embedder init failed")

    if worker_tasks.cv2 is None:
        class _FakeCv2:
            CAP_PROP_FPS = 5
            CAP_PROP_FRAME_COUNT = 7
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            VideoCapture = _FakeCapture

        monkeypatch.setattr(worker_tasks, "cv2", _FakeCv2)
    else:
        monkeypatch.setattr(worker_tasks.cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(worker_tasks, "YOLO", _FakeYOLO)
    monkeypatch.setattr(worker_tasks, "resolve_device", lambda: ("cuda:0", 0))
    monkeypatch.setattr(worker_tasks, "warm_easyocr_models", lambda *_: None)
    monkeypatch.setattr(worker_tasks, "_build_ocr_reader", lambda *_: (None, False))
    monkeypatch.setattr(worker_tasks, "_hex_to_hsv", lambda *_: (60, 80, 80))
    monkeypatch.setattr(worker_tasks, "OSNetEmbedder", _broken_embedder)
    monkeypatch.setattr(worker_tasks, "ensure_reid_weights", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("download failed")))

    try:
        worker_tasks.track_presence(
            str(tmp_path / "dummy.mp4"),
            {
                "reid_enable": True,
                "reid_fail_policy": "fail",
                "reid_auto_download": True,
                "reid_weights_path": str(tmp_path / "missing.pth"),
                "reid_weights_url": "https://example.com/osnet.pth",
                "debug_timeline": False,
            },
        )
        assert False, "Expected ReID init failure to raise when reid_fail_policy=fail"
    except RuntimeError as exc:
        assert "download failed" in str(exc)


def test_reid_init_with_existing_weights_skips_download_and_initializes(tmp_path, monkeypatch):
    from worker import tasks as worker_tasks

    class _FakeCapture:
        def __init__(self, _):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == worker_tasks.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_COUNT:
                return 0
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_WIDTH:
                return 1920
            if prop == worker_tasks.cv2.CAP_PROP_FRAME_HEIGHT:
                return 1080
            return 0

        def read(self):
            return False, None

        def release(self):
            return None

    class _FakeYOLO:
        def __init__(self, *_args, **_kwargs):
            pass

    calls = {"embedder": 0, "download": 0}

    class _Embedder:
        def __init__(self, _cfg):
            calls["embedder"] += 1

    weights_path = tmp_path / "existing.pth"
    weights_path.write_bytes(b"x" * (2 * 1024 * 1024))

    if worker_tasks.cv2 is None:
        class _FakeCv2:
            CAP_PROP_FPS = 5
            CAP_PROP_FRAME_COUNT = 7
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            VideoCapture = _FakeCapture

        monkeypatch.setattr(worker_tasks, "cv2", _FakeCv2)
    else:
        monkeypatch.setattr(worker_tasks.cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(worker_tasks, "YOLO", _FakeYOLO)
    monkeypatch.setattr(worker_tasks, "resolve_device", lambda: ("cuda:0", 0))
    monkeypatch.setattr(worker_tasks, "warm_easyocr_models", lambda *_: None)
    monkeypatch.setattr(worker_tasks, "_build_ocr_reader", lambda *_: (None, False))
    monkeypatch.setattr(worker_tasks, "_hex_to_hsv", lambda *_: (60, 80, 80))
    monkeypatch.setattr(worker_tasks, "OSNetEmbedder", _Embedder)
    monkeypatch.setattr(worker_tasks, "ensure_reid_weights", lambda *_args, **_kwargs: calls.__setitem__("download", calls["download"] + 1))

    worker_tasks.track_presence(
        str(tmp_path / "dummy.mp4"),
        {
            "reid_enable": True,
            "reid_fail_policy": "disable",
            "reid_auto_download": True,
            "reid_weights_path": str(weights_path),
            "reid_weights_url": "https://example.com/osnet.pth",
            "debug_timeline": False,
        },
    )

    assert calls["embedder"] == 1
    assert calls["download"] == 0
