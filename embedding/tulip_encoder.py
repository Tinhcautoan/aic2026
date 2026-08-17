"""
TULIP Encoder (UC Berkeley) — Token-length Upgraded CLIP.
Imports directly from the tulip/open_clip fork source files.
Uses the original open_clip_torch package for SigLIP2 (they coexist via filesystem import).
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from loguru import logger

from config import settings

# Point to TULIP's open_clip fork source (before original open_clip_torch in sys.path)
_tulip_dir = os.path.join(os.path.dirname(__file__), '..', 'tulip', 'open_clip', 'src')
if os.path.isdir(_tulip_dir):
    if _tulip_dir not in sys.path:
        sys.path.insert(0, _tulip_dir)

class TULIPEncoder:
    """TULIP-based encoder using UC Berkeley open_clip fork. 1024-dim embeddings."""

    def __init__(self, device: str = None):
        self.device = device or (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )

        logger.info(f"Initializing TULIP encoder on {self.device}")

        import open_clip

        self.model, _, self.preprocess = open_clip.factory.create_model_and_transforms(
            model_name=settings.TULIP_MODEL_NAME,
            pretrained=settings.TULIP_PRETRAINED,
            device=self.device,
        )

        if settings.USE_FP16 and "cuda" in self.device:
            self.model = self.model.half()

        self.model.eval()
        self._tokenizer = open_clip.tokenizer.SimpleTokenizer()
        self.embedding_dim = settings.EMBEDDING_DIM_CLIP

        logger.info(
            f"TULIP encoder loaded: {settings.TULIP_MODEL_NAME} "
            f"(dim={self.embedding_dim}, fp16={settings.USE_FP16})"
        )

    def _tokenize(self, texts):
        """Tokenize using TULIP's SimpleTokenizer (handles padding/truncation)"""
        if isinstance(texts, str):
            texts = [texts]
        return self._tokenizer(texts)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            t = self.preprocess(image).unsqueeze(0).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device: t = t.half()
            f = self.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32).flatten()

    def encode_images_batch(self, images: list) -> np.ndarray:
        with torch.no_grad():
            t = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device: t = t.half()
            f = self.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self._tokenize(text).to(self.device)
            f = self.model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32).flatten()

    def encode_texts_batch(self, texts: list) -> np.ndarray:
        with torch.no_grad():
            tokens = self._tokenize(texts).to(self.device)
            f = self.model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)

    def get_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
