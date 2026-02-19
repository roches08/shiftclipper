from __future__ import annotations

import logging
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("worker")

_MIN_WEIGHTS_BYTES = 5 * 1024 * 1024
_MAX_RETRIES = 3


def _normalize_weights_url(url: str) -> str:
    parsed = urlparse(url)
    is_huggingface_resolve = parsed.netloc.endswith("huggingface.co") and "/resolve/" in parsed.path and parsed.path.endswith(".pth")
    if is_huggingface_resolve and not parsed.query:
        return f"{url}?download=true"
    return url


def _validate_weights_file(path: Path, min_bytes: int = _MIN_WEIGHTS_BYTES) -> int:
    if not path.exists():
        raise RuntimeError(f"Downloaded ReID weights missing at {path}")
    size = path.stat().st_size
    with path.open("rb") as f:
        header = f.read(200)
    if size <= min_bytes:
        raise RuntimeError(
            f"Downloaded ReID weights too small ({size} bytes). Header preview: {header!r}"
        )
    if b"<html" in header.lower():
        raise RuntimeError(f"Downloaded ReID weights appear to be HTML, likely a failed redirect/auth response. Header preview: {header!r}")
    return size


def _download_with_urllib(url: str, tmp_path: Path, timeout_sec: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "shiftclipper/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as src, tmp_path.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def _download_with_curl(url: str, tmp_path: Path, timeout_sec: float) -> None:
    timeout = max(1, int(timeout_sec))
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--connect-timeout",
        str(min(timeout, 30)),
        "--max-time",
        str(timeout),
        "-o",
        str(tmp_path),
        url,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def ensure_reid_weights(weights_path: str, weights_url: str, timeout_sec: float = 120.0) -> Path:
    target = Path(str(weights_path or "").strip())
    if not str(target):
        raise RuntimeError("reid_weights_path is required")
    url = str(weights_url or "").strip()
    if not url:
        raise RuntimeError("reid_weights_url is required")
    url = _normalize_weights_url(url)

    if target.exists():
        _validate_weights_file(target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{target}.tmp")

    errors = []
    for attempt in range(1, _MAX_RETRIES + 1):
        if tmp_path.exists():
            tmp_path.unlink()
        log.info("Downloading ReID weights (attempt %d/%d) from %s", attempt, _MAX_RETRIES, url)
        try:
            _download_with_urllib(url, tmp_path, timeout_sec)
        except Exception as urllib_exc:
            log.warning("urllib ReID weights download failed (attempt %d/%d): %s", attempt, _MAX_RETRIES, urllib_exc)
            try:
                _download_with_curl(url, tmp_path, timeout_sec)
            except Exception as curl_exc:
                errors.append(f"attempt {attempt}: urllib={urllib_exc}; curl={curl_exc}")
                if attempt < _MAX_RETRIES:
                    time.sleep(float(attempt))
                continue

        try:
            size = _validate_weights_file(tmp_path)
            os.replace(tmp_path, target)
            log.info("ReID weights ready at %s (%d bytes)", target, size)
            return target
        except Exception as validate_exc:
            errors.append(f"attempt {attempt}: validation={validate_exc}")
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < _MAX_RETRIES:
                time.sleep(float(attempt))

    if tmp_path.exists():
        tmp_path.unlink()
    raise RuntimeError(f"Failed to download valid ReID weights from {url} after {_MAX_RETRIES} attempts: {' | '.join(errors)}")
