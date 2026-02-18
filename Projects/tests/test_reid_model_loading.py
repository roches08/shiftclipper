import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from worker import reid


class _FakeModel:
    def __init__(self, strict_missing=None, strict_unexpected=None, strict_error=None):
        self.strict_missing = strict_missing or []
        self.strict_unexpected = strict_unexpected or []
        self.strict_error = strict_error
        self.calls = []

    def load_state_dict(self, state_dict, strict):
        self.calls.append((strict, state_dict))
        if strict and self.strict_error:
            raise RuntimeError(self.strict_error)
        return self.strict_missing, self.strict_unexpected


class _FakeTorch:
    def __init__(self, checkpoint):
        self._checkpoint = checkpoint

    def load(self, _path, map_location="cpu"):
        assert map_location == "cpu"
        return self._checkpoint


def _build_embedder_with_mocks(monkeypatch, checkpoint, model):
    build_calls = {}

    def fake_build_model(name, num_classes, pretrained, use_gpu):
        build_calls["name"] = name
        build_calls["num_classes"] = num_classes
        build_calls["pretrained"] = pretrained
        build_calls["use_gpu"] = use_gpu
        return model

    monkeypatch.setattr(reid, "torch", _FakeTorch(checkpoint))
    monkeypatch.setattr(reid, "torchreid_models", SimpleNamespace(build_model=fake_build_model))
    return reid.OSNetEmbedder.__new__(reid.OSNetEmbedder), build_calls


def test_build_model_requires_existing_weights_path(monkeypatch, tmp_path):
    embedder, _ = _build_embedder_with_mocks(monkeypatch, checkpoint={}, model=_FakeModel())
    cfg = reid.ReIDConfig(model_name="osnet_x0_25", device="cpu", weights_path=str(tmp_path / "missing.pth"))

    with pytest.raises(RuntimeError) as exc:
        embedder._build_model(cfg)

    assert "Missing ReID weights" in str(exc.value)


def test_build_model_uses_pretrained_false_and_emits_reid_ready(monkeypatch, tmp_path, caplog):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"ok")
    checkpoint = {"state_dict": {"module.layer": 1, "head": 2}}
    model = _FakeModel()
    embedder, build_calls = _build_embedder_with_mocks(monkeypatch, checkpoint=checkpoint, model=model)
    cfg = reid.ReIDConfig(model_name="osnet_x0_25", device="cpu", weights_path=str(weights))

    with caplog.at_level(logging.INFO, logger="worker"):
        embedder._build_model(cfg)

    assert build_calls["pretrained"] is False
    assert model.calls[0][0] is True
    assert "layer" in model.calls[0][1]
    assert "module.layer" not in model.calls[0][1]
    assert any("timeline event=reid_ready" in rec.message for rec in caplog.records)


def test_build_model_raises_if_too_many_missing_keys_after_strict_fallback(monkeypatch, tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"ok")
    checkpoint = {"model": {f"module.k{i}": i for i in range(5)}}
    model = _FakeModel(strict_missing=[f"m{i}" for i in range(51)], strict_unexpected=["u0"], strict_error="boom")
    embedder, _ = _build_embedder_with_mocks(monkeypatch, checkpoint=checkpoint, model=model)
    cfg = reid.ReIDConfig(model_name="osnet_x0_25", device="cpu", weights_path=str(weights))

    with pytest.raises(RuntimeError) as exc:
        embedder._build_model(cfg)

    assert "too many missing keys" in str(exc.value)
    assert [strict for strict, _ in model.calls] == [True, False]
