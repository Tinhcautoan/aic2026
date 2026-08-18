"""
TULIP Encoder (UC Berkeley) — Token-length Upgraded CLIP.
Loads the TULIP fork's open_clip in isolation during __init__ so it never
pollutes sys.modules['open_clip'] used by siglip2_encoder.py.
"""

import sys
import os
import types
import torch
import numpy as np
from PIL import Image
from loguru import logger

from config import settings


def _load_tulip_isolated():
    """
    Load TULIP's open_clip fork as an isolated 'tulip_open_clip' module.
    Uses importlib to avoid polluting sys.modules['open_clip'].
    """
    if 'tulip_open_clip' in sys.modules:
        return sys.modules['tulip_open_clip']

    tulip_src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'tulip', 'open_clip', 'src')
    )
    if not os.path.isdir(tulip_src):
        raise RuntimeError(f"TULIP open_clip fork not found at {tulip_src}")

    # Save current sys.path and open_clip state
    saved_path = list(sys.path)
    saved_oc = sys.modules.get('open_clip')
    saved_oc_subs = {k: v for k, v in sys.modules.items()
                     if k == 'open_clip' or k.startswith('open_clip.')}

    # Remove standard open_clip temporarily
    for key in list(saved_oc_subs):
        sys.modules.pop(key)

    # Add TULIP src to path and import
    if tulip_src not in sys.path:
        sys.path.insert(0, tulip_src)
    try:
        import open_clip as tulip_oc
    finally:
        sys.path[:] = saved_path

    # Collect ALL TULIP modules that were registered
    tulip_all = {}
    for key in list(sys.modules):
        if key == 'open_clip' or key.startswith('open_clip.'):
            tulip_all[key] = sys.modules.pop(key)

    # Restore standard open_clip
    if saved_oc is not None:
        sys.modules['open_clip'] = saved_oc
    for k, v in saved_oc_subs.items():
        if k != 'open_clip':
            sys.modules[k] = v

    # Create isolated tulip_open_clip package
    pkg = types.ModuleType('tulip_open_clip')
    pkg.__path__ = [os.path.join(tulip_src, 'open_clip')]
    pkg.__package__ = 'tulip_open_clip'
    sys.modules['tulip_open_clip'] = pkg

    # Install all TULIP submodules under tulip_open_clip.*
    for key, mod in tulip_all.items():
        new_key = key.replace('open_clip', 'tulip_open_clip', 1)
        mod.__name__ = new_key
        sys.modules[new_key] = mod
        # Set as attribute on parent
        parts = new_key.split('.')
        if len(parts) == 2:
            setattr(pkg, parts[1], mod)

    # Set top-level attrs from the TULIP open_clip module
    for attr in ['create_model', 'create_model_and_transforms',
                 'get_tokenizer', 'list_pretrained',
                 'factory', 'model', 'tokenizer']:
        if hasattr(tulip_oc, attr):
            setattr(pkg, attr, getattr(tulip_oc, attr))

    logger.info("Loaded TULIP open_clip fork as isolated 'tulip_open_clip'")
    return pkg


# Lazy-loaded — only initialized when TULIPEncoder.__init__ is called
tulip_open_clip = None


def _get_tulip():
    global tulip_open_clip
    if tulip_open_clip is None:
        tulip_open_clip = _load_tulip_isolated()
    return tulip_open_clip


class TULIPEncoder:
    """TULIP-based encoder using UC Berkeley open_clip fork. 1024-dim embeddings."""

    def __init__(self, device: str = None):
        self.device = device or (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )

        logger.info(f"Initializing TULIP encoder on {self.device}")

        # Load TULIP fork in isolation (only when first needed)
        tulip = _get_tulip()

        self.model, _, self.preprocess = tulip.factory.create_model_and_transforms(
            model_name=settings.TULIP_MODEL_NAME,
            pretrained=settings.TULIP_PRETRAINED,
            device=self.device,
        )

        if settings.USE_FP16 and "cuda" in self.device:
            self.model = self.model.half()

        self.model.eval()
        self._tokenizer = tulip.tokenizer.SimpleTokenizer()
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
