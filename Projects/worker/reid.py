from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("worker")

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from torchreid import models as torchreid_models
except Exception as e:  # pragma: no cover
    torchreid_models = None
    log.exception("torchreid import failed (ReID will be unavailable): %s", e)


_MAX_MISSING_KEYS = 50


def normalize_vector(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vec is None:
        return None
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return (vec / norm).astype(np.float32)


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-6 or bn < 1e-6:
        return -1.0
    return float(np.dot(a, b) / (an * bn))


def expand_and_crop(frame: np.ndarray, box: Tuple[int, int, int, int], expand: float, min_side_px: int) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if min(bw, bh) < min_side_px:
        return None
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    scale = 1.0 + max(0.0, float(expand))
    nw = bw * scale
    nh = bh * scale
    ex1 = max(0, min(w - 1, int(round(cx - nw / 2.0))))
    ex2 = max(ex1 + 1, min(w, int(round(cx + nw / 2.0))))
    ey1 = max(0, min(h - 1, int(round(cy - nh / 2.0))))
    ey2 = max(ey1 + 1, min(h, int(round(cy + nh / 2.0))))
    crop = frame[ey1:ey2, ex1:ex2]
    if crop.size == 0 or min(crop.shape[:2]) < min_side_px:
        return None
    return crop


@dataclass
class ReIDConfig:
    model_name: str = "osnet_x0_25"
    device: str = "cuda:0"
    batch_size: int = 16
    use_fp16: bool = True
    weights_path: str = ""


def _extract_state_dict(checkpoint: object) -> object:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], (dict,)):
                return checkpoint[key]
    return checkpoint


def _strip_prefixes(state_dict: object) -> object:
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict:
        return state_dict
    out = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith("module."):
            out[key[len("module."):]] = value
        else:
            out[key] = value
    return out


def _drop_classifier(state_dict: dict) -> tuple[dict, bool]:
    """Drop classifier weights if present (we only need embeddings)."""
    if not isinstance(state_dict, dict):
        return state_dict, False
    removed = False
    for k in ["classifier.weight", "classifier.bias"]:
        if k in state_dict:
            state_dict.pop(k, None)
            removed = True
    return state_dict, removed


class OSNetEmbedder:
    def __init__(self, cfg: ReIDConfig):
        if torch is None:
            raise RuntimeError("torch is required for ReID")
        if cv2 is None:
            raise RuntimeError("opencv is required for ReID")
        self.cfg = cfg
        if cfg.model_name not in {"osnet_x0_25", "osnet_x1_0"}:
            raise RuntimeError(f"Unsupported reid_model={cfg.model_name}")
        if torchreid_models is None:
            raise RuntimeError("torchreid is required for OSNet ReID model loading")
        self.model = self._build_model(cfg)
        # Remove classifier head so model outputs embeddings only
        if hasattr(self.model, "classifier"):
            self.model.classifier = torch.nn.Identity()
        self.device = torch.device(cfg.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.use_fp16 = bool(cfg.use_fp16 and self.device.type == "cuda")
        self.input_hw = (256, 128)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _build_model(self, cfg: ReIDConfig):
        use_gpu = str(cfg.device).startswith("cuda")
        weights_path = str(cfg.weights_path or "").strip()
        try:
            if not weights_path:
                raise RuntimeError("Missing ReID weights; set reid_weights_path")
            local_path = Path(weights_path)
            if not local_path.exists():
                raise RuntimeError(f"Missing ReID weights at {local_path}")
            checkpoint = torch.load(str(local_path), map_location="cpu")
            state_dict = _strip_prefixes(_extract_state_dict(checkpoint))

            # We only need backbone embeddings for tracking.
            # Many checkpoints (e.g. MSMT17 combineall) have classifier shapes that don't match.
            state_dict, dropped = _drop_classifier(state_dict)

            model = torchreid_models.build_model(
                name=cfg.model_name,
                num_classes=1000,  # arbitrary since classifier head is unused
                pretrained=False,
                use_gpu=use_gpu,
            )

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            missing_keys = list(missing_keys) if missing_keys is not None else []
            unexpected_keys = list(unexpected_keys) if unexpected_keys is not None else []

            log.info(
                "ReID OSNet weights loaded (strict=False) model=%s dropped_classifier=%s missing=%d unexpected=%d",
                cfg.model_name,
                dropped,
                len(missing_keys),
                len(unexpected_keys),
            )

            if len(missing_keys) > _MAX_MISSING_KEYS:
                raise RuntimeError(
                    f"too many missing keys; wrong checkpoint/model mismatch (missing={len(missing_keys)} threshold={_MAX_MISSING_KEYS})"
                )
            log.info("timeline event=reid_ready model=%s weights_path=%s", cfg.model_name, str(local_path))
            return model
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load OSNet model {cfg.model_name}: {exc}") from exc

    def _preprocess(self, crop: np.ndarray):
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_hw[1], self.input_hw[0]), interpolation=cv2.INTER_LINEAR)
        ten = torch.from_numpy(img)
        ten = ten.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ten = ten.to(self.device)
        ten = (ten - self.mean) / self.std
        return ten

    def embed(self, crops: Sequence[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        if not crops:
            return []
        outputs: List[Optional[np.ndarray]] = [None] * len(crops)
        valid = [(i, c) for i, c in enumerate(crops) if c is not None and c.size > 0]
        if not valid:
            return outputs
        with torch.inference_mode():
            for start in range(0, len(valid), max(1, int(self.cfg.batch_size))):
                chunk = valid[start:start + max(1, int(self.cfg.batch_size))]
                batch = torch.cat([self._preprocess(c) for _, c in chunk], dim=0)
                batch = batch.to(self.device)
                if self.use_fp16 and self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        feats = self.model(batch)
                else:
                    feats = self.model(batch)
                feats = feats.float()
                feats = torch.nn.functional.normalize(feats, dim=1)
                arr = feats.detach().cpu().numpy().astype(np.float32)
                for (idx, _), emb in zip(chunk, arr):
                    outputs[idx] = emb
        return outputs
