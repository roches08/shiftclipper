import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from worker import tasks as worker_tasks


class _FakeProc:
    def __init__(self, polls_before_done=3):
        self.polls_before_done = polls_before_done
        self.terminated = False
        self.killed = False
        self.wait_called = 0

    def poll(self):
        if self.polls_before_done > 0:
            self.polls_before_done -= 1
            return None
        return 0

    def wait(self, timeout=None):
        self.wait_called += 1
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_cut_clip_with_heartbeat_touches_heartbeat_during_long_ffmpeg(monkeypatch):
    fake_proc = _FakeProc(polls_before_done=4)
    monkeypatch.setattr(worker_tasks.subprocess, "Popen", lambda *args, **kwargs: fake_proc)
    monkeypatch.setattr(worker_tasks.time, "sleep", lambda *_args, **_kwargs: None)

    calls = {"heartbeat": 0}

    worker_tasks.cut_clip_with_heartbeat(
        "in.mp4",
        0.0,
        5.0,
        "out.mp4",
        heartbeat=lambda: calls.__setitem__("heartbeat", calls["heartbeat"] + 1),
        cancel_check=lambda: False,
    )

    assert calls["heartbeat"] >= 4
    assert fake_proc.wait_called >= 1


def test_cut_clip_with_heartbeat_cancels_cleanly(monkeypatch):
    fake_proc = _FakeProc(polls_before_done=10)
    monkeypatch.setattr(worker_tasks.subprocess, "Popen", lambda *args, **kwargs: fake_proc)
    monkeypatch.setattr(worker_tasks.time, "sleep", lambda *_args, **_kwargs: None)

    try:
        worker_tasks.cut_clip_with_heartbeat(
            "in.mp4",
            0.0,
            5.0,
            "out.mp4",
            heartbeat=lambda: None,
            cancel_check=lambda: True,
        )
        assert False, "expected cancellation"
    except RuntimeError as exc:
        assert "cancelled" in str(exc).lower()

    assert fake_proc.terminated is True
