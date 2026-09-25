"""
PE-Core text tower only.

The retrieval server never embeds images at query time -- the frame embeddings were
produced offline (Colab) with the SAME checkpoint -- so the ~1.9B-parameter vision
tower is dropped right after loading. That leaves the ~0.5B text tower (~1 GB in
fp16) on the GPU next to the frame matrix and the translator.

Loading goes through CPU first: putting the whole fp32 model on an 8 GB card needs
~9.7 GB transiently and segfaults (see embedding/pecore_encoder.py).
"""

from __future__ import annotations

import gc
import sys
import threading
import time
from typing import List, Optional, Sequence

import numpy as np
import torch
from loguru import logger

# Keep the vendored TULIP open_clip fork from shadowing the real package (same trick as
# the other encoders in this folder).
_tulip_paths = [p for p in sys.path if "tulip" in p and "open_clip" in p]
for _p in _tulip_paths:
    sys.path.remove(_p)
import open_clip  # noqa: E402
sys.path.extend(_tulip_paths)

from config import settings  # noqa: E402


class PECoreText:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None, fp16: bool = True):
        self.model_name = model_name or settings.PECORE_MODEL_NAME
        self.device = device or (f"cuda:{settings.GPU_DEVICE_ID}" if settings.GPU_ENABLED and torch.cuda.is_available() else "cpu")
        self.half = bool(fp16 and self.device.startswith("cuda"))
        self._lock = threading.Lock()
        t0 = time.perf_counter()
        logger.info(f"Loading PE-Core text tower ({self.model_name}) ...")
        model, _, _ = open_clip.create_model_and_transforms(self.model_name, device="cpu")
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        model.visual = torch.nn.Identity()          # free the vision tower before it ever touches the GPU
        gc.collect()
        if self.half:
            model = model.half()
        self.model = model.to(self.device).eval()
        self.dim = int(self.model.text_projection.shape[-1]) if hasattr(self.model, "text_projection") and self.model.text_projection is not None \
            else int(settings.EMBEDDING_DIM_PECORE)
        logger.info(f"PE-Core text tower ready on {self.device} in {time.perf_counter() - t0:.1f}s (dim={self.dim}, fp16={self.half})")

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """[n, D] L2-normalised text embeddings, on the model device (fp16 on GPU)."""
        texts = [t if t.strip() else " " for t in texts]
        with self._lock:
            tokens = self.tokenizer(list(texts)).to(self.device)
            f = self.model.encode_text(tokens)
            return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def encode_np(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts).float().cpu().numpy()

    def warmup(self) -> None:
        for _ in range(2):
            self.encode(["a person walking on the street", "warm up"])
