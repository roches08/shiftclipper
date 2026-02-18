import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from common import config as cfg


class _CudaOn:
    @staticmethod
    def is_available():
        return True


class _CudaOff:
    @staticmethod
    def is_available():
        return False


class _TorchOn:
    cuda = _CudaOn()


class _TorchOff:
    cuda = _CudaOff()


def test_resolve_device_returns_cuda_when_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _TorchOn())
    device, yolo_device = cfg.resolve_device()
    assert device == "cuda:0"
    assert yolo_device == 0


def test_resolve_device_raises_when_cuda_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _TorchOff())
    with pytest.raises(RuntimeError, match="CUDA is required for ShiftClipper jobs"):
        cfg.resolve_device()
