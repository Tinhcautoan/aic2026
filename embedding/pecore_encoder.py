"""
PE-Core encoder (Meta's Perception Encoder) -- replaces the CLIP/TULIP
branch (the `clip_embedding` field / `clip_encoder` slot in
HybridEmbeddingEngine).

Loads `hf-hub:timm/PE-Core-bigG-14-448` through the exact same open_clip
hf-hub call already validated by `export_embeddings_from_keyframes.py`,
which produced the 1280-dim vectors in `embeddings_pecore.pkl`:

    open_clip.create_model_and_transforms('hf-hub:timm/PE-Core-bigG-14-448',
                                           device=device)

Native output is 1280-dim -- no projection/truncation, matching the
already-exported frame embeddings.

CAUTION -- text side is unverified
-----------------------------------
The export script only ran `encode_image` (image-only extraction job).
`encode_text` below follows the same open_clip tokenizer/encode_text
pattern used by clip_encoder.py and siglip2_encoder.py, but has not been
smoke-tested against this checkpoint. Sanity-check before relying on it:
cosine(image, correct caption) should sit clearly above cosine(image,
random caption).
"""

import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image
from loguru import logger

# Same isolation as siglip2_encoder.py: keep the vendored TULIP open_clip
# fork from shadowing the real package on sys.path.
_tulip_paths = [p for p in sys.path if "tulip" in p and "open_clip" in p]
for _p in _tulip_paths:
    sys.path.remove(_p)
import open_clip  # noqa: E402
sys.path.extend(_tulip_paths)

from config import settings  # noqa: E402


class PECoreEncoder:
    """PE-Core (Meta) image/text encoder. Fixed native output, no projection."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Initializing PE-Core encoder on {self.device}")

        # Load on CPU first, THEN half()+move to GPU. bigG is ~2.4B params --
        # loading straight to CUDA in fp32 (create_model_and_transforms'
        # device= arg) needs ~9.7GB transiently before .half() can shrink it,
        # which segfaults on an 8GB card. CPU-load -> half() -> .to(device)
        # peaks at ~4.9GB instead.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.PECORE_MODEL_NAME,
            device="cpu",
        )
        if settings.USE_FP16 and "cuda" in self.device:
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(settings.PECORE_MODEL_NAME)

        self.embedding_dim = int(settings.EMBEDDING_DIM_PECORE)
        self.logit_scale, self.logit_bias = self._read_logit_params()

        logger.info(
            f"PE-Core ready: {settings.PECORE_MODEL_NAME} dim={self.embedding_dim} "
            f"logit_scale={self.logit_scale:.2f} logit_bias={self.logit_bias:.3f}"
        )

    # ------------------------------------------------------------------ #

    @property
    def output_dim(self) -> int:
        """Dimensionality actually written to the vector store."""
        return self.embedding_dim

    def _read_logit_params(self):
        from retrieval.scoring import get_logit_params
        return get_logit_params(self.model)

    def _normalize(self, f: torch.Tensor) -> torch.Tensor:
        return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # ------------------------------------------------------------------ #

    def encode_image(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            t = self.preprocess(image).unsqueeze(0).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device:
                t = t.half()
            f = self._normalize(self.model.encode_image(t))
        return f.float().cpu().numpy().astype(np.float32).flatten()

    def encode_images_batch(self, images: list) -> np.ndarray:
        with torch.no_grad():
            t = torch.stack([self.preprocess(i) for i in images]).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device:
                t = t.half()
            f = self._normalize(self.model.encode_image(t))
        return f.float().cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            f = self._normalize(self.model.encode_text(tokens))
        return f.float().cpu().numpy().astype(np.float32).flatten()

    def encode_texts_batch(self, texts: list) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            f = self._normalize(self.model.encode_text(tokens))
        return f.float().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #

    def get_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """Raw cosine. Kept for compatibility -- prefer `get_logit`."""
        return float(np.dot(image_features, text_features))

    def get_logit(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """Cosine mapped into the model's trained logit space."""
        return float(np.dot(image_features, text_features)) * self.logit_scale + self.logit_bias
