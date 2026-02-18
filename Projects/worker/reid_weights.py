from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger("worker")

_MIN_WEIGHTS_BYTES = 1 * 1024 * 1024


def ensure_reid_weights(weights_path: str, weights_url: str, timeout_sec: float = 120.0) -> Path:
    target = Path(str(weights_path or "").strip())
    if not str(target):
        raise RuntimeError("reid_weights_path is required")
    url = str(weights_url or "").strip()
    if not url:
        raise RuntimeError("reid_weights_url is required")

    if target.exists() and target.stat().st_size > _MIN_WEIGHTS_BYTES:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()

    log.info("Downloading ReID weights from %s to %s", url, target)
    request = urllib.request.Request(url, headers={"User-Agent": "shiftclipper/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as src, tmp_path.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)

    size = tmp_path.stat().st_size if tmp_path.exists() else 0
    if size <= _MIN_WEIGHTS_BYTES:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Downloaded ReID weights too small ({size} bytes): {url}")

    os.replace(tmp_path, target)
    log.info("ReID weights ready at %s (%d bytes)", target, target.stat().st_size)
    return target
