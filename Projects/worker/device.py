import os
from typing import Any, Dict

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def resolve_device() -> str:
    env_device = (os.getenv("SHIFTCLIPPER_DEVICE") or "").strip()
    if env_device:
        return env_device

    if torch is None:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        return "cpu"
    return "cpu"


def get_runtime_device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "selected_device": resolve_device(),
        "torch_version": getattr(torch, "__version__", "not-installed") if torch is not None else "not-installed",
        "cuda_available": False,
        "gpu_name": None,
    }

    if torch is None:
        return info

    try:
        cuda_available = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda_available
        if cuda_available:
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                info["gpu_name"] = "unknown"
    except Exception:
        info["cuda_available"] = False

    return info
