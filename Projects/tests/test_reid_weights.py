import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from worker import reid_weights


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(reid_weights.time, "sleep", lambda *_args, **_kwargs: None)


def test_ensure_reid_weights_creates_parent_and_renames_tmp(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "models" / "reid" / "weights.pth"
    called = {}

    def fake_download(url, tmp_path_arg, timeout):
        called["tmp"] = Path(tmp_path_arg)
        Path(tmp_path_arg).write_bytes(b"x" * (11 * 1024 * 1024))

    monkeypatch.setattr(reid_weights, "_download_with_urllib", fake_download)
    monkeypatch.setattr(reid_weights, "_download_with_curl", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("curl should not be used")))

    out = reid_weights.ensure_reid_weights(str(target), "https://example.com/osnet.pth")

    assert out == target
    assert target.exists()
    assert called["tmp"] == Path(f"{target}.tmp")
    assert not Path(f"{target}.tmp").exists()


def test_ensure_reid_weights_rejects_small_file(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "weights.pth"
    monkeypatch.setattr(reid_weights, "_MAX_RETRIES", 1)

    def fake_download(url, tmp_path_arg, timeout):
        Path(tmp_path_arg).write_bytes(b"small")

    monkeypatch.setattr(reid_weights, "_download_with_urllib", fake_download)
    monkeypatch.setattr(reid_weights, "_download_with_curl", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError) as exc:
        reid_weights.ensure_reid_weights(str(target), "https://example.com/osnet.pth")

    assert "too small" in str(exc.value)
    assert not target.exists()


def test_ensure_reid_weights_rejects_html_payload(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "weights.pth"
    monkeypatch.setattr(reid_weights, "_MAX_RETRIES", 1)

    def fake_download(url, tmp_path_arg, timeout):
        payload = b"<html><body>redirect</body></html>" + (b"x" * (11 * 1024 * 1024))
        Path(tmp_path_arg).write_bytes(payload)

    monkeypatch.setattr(reid_weights, "_download_with_urllib", fake_download)
    monkeypatch.setattr(reid_weights, "_download_with_curl", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError) as exc:
        reid_weights.ensure_reid_weights(str(target), "https://example.com/osnet.pth")

    assert "appear to be HTML" in str(exc.value)


def test_ensure_reid_weights_retries_and_uses_curl_fallback(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "weights.pth"
    calls = {"urllib": 0, "curl": 0}

    def flaky_urllib(url, tmp_path_arg, timeout):
        calls["urllib"] += 1
        raise RuntimeError("urllib failed")

    def good_curl(url, tmp_path_arg, timeout):
        calls["curl"] += 1
        Path(tmp_path_arg).write_bytes(b"x" * (11 * 1024 * 1024))

    monkeypatch.setattr(reid_weights, "_download_with_urllib", flaky_urllib)
    monkeypatch.setattr(reid_weights, "_download_with_curl", good_curl)

    out = reid_weights.ensure_reid_weights(str(target), "https://example.com/osnet.pth")

    assert out == target
    assert calls["urllib"] == 1
    assert calls["curl"] == 1


def test_ensure_reid_weights_uses_existing_valid_file(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "weights.pth"
    target.write_bytes(b"x" * (11 * 1024 * 1024))

    monkeypatch.setattr(reid_weights, "_download_with_urllib", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("urllib should not be used")))
    monkeypatch.setattr(reid_weights, "_download_with_curl", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("curl should not be used")))

    out = reid_weights.ensure_reid_weights(str(target), "https://example.com/osnet.pth")

    assert out == target

