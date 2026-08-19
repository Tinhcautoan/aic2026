"""
Hybrid Embedding Engine combining CLIP and SigLIP2 embeddings.
Implements dual-encoder inference with asynchronous scheduling as described in the paper.
"""

import asyncio
import time
import numpy as np
import torch
from PIL import Image
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from .clip_encoder import CLIPEncoder
from .tulip_encoder import TULIPEncoder
from .siglip2_encoder import SigLIP2Encoder
from config import settings

def _validate_cuda(device: str) -> str:
    """Test if CUDA device works; return 'cpu' if not."""
    if device == "cpu" or not device.startswith("cuda"):
        return device
    try:
        t = torch.zeros(1, device=device)
        _ = t + t
        del t
        torch.cuda.empty_cache()
        return device
    except Exception as e:
        logger.warning(f"CUDA validation failed on {device}: {e}. Falling back to CPU.")
        return "cpu"


class HybridEmbeddingEngine:
    """
    Fused TULIP/CLIP + SigLIP2 embedding engine.
    Uses TULIP (UC Berkeley) instead of CLIP when settings.USE_TULIP is True.
    Both models run asynchronously on separate threads to minimize latency overhead.
    """

    def __init__(self, device: str = None):
        self.device = device or (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )
        self.device = _validate_cuda(self.device)

        logger.info("Initializing Hybrid Embedding Engine...")
        start_time = time.time()

        if settings.USE_TULIP:
            logger.info("Using TULIP encoder instead of CLIP")
            self.clip_encoder = TULIPEncoder(device=self.device)
        else:
            self.clip_encoder = CLIPEncoder(device=self.device)
        self.siglip2_encoder = SigLIP2Encoder(device=self.device)

        self.unified_dim = settings.UNIFIED_DIM
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Vietnamese -> English translation for the English-only text encoders.
        # Lazy: the model is only downloaded/loaded on the first Vietnamese query.
        self._translator = None
        if settings.VI_EN_TRANSLATION_ENABLED:
            try:
                from .query_translator import VietnameseEnglishTranslator
                self._translator = VietnameseEnglishTranslator(
                    model_name=settings.VI_EN_TRANSLATION_MODEL,
                    model_dir=settings.VI_EN_TRANSLATION_MODEL_DIR,
                    device=settings.VI_EN_TRANSLATION_DEVICE,
                    num_beams=settings.VI_EN_TRANSLATION_NUM_BEAMS,
                )
            except Exception as e:
                logger.warning(f"vi->en translator disabled: {e}")

        elapsed = time.time() - start_time
        logger.info(f"Hybrid Embedding Engine initialized in {elapsed:.2f}s")

    def warmup(self):
        """Preload lazily-loaded models (the vi->en translator) at startup."""
        if self._translator is not None:
            self._translator.preload()

    def _maybe_translate(self, text: str) -> str:
        """Translate Vietnamese to English so the English-only TULIP/SigLIP2
        text encoders embed it correctly. English text passes through unchanged."""
        if self._translator is None or not text:
            return text
        return self._translator.maybe_translate_to_english(text)

    def encode_image(self, image: Image.Image) -> dict:
        """
        Encode a single image with both CLIP and SigLIP2.
        Returns dict with 'clip' and 'siglip2' embeddings.
        """
        clip_emb = self.clip_encoder.encode_image(image)
        siglip2_emb = self.siglip2_encoder.encode_image(image)
        return {
            "clip": clip_emb,
            "siglip2": siglip2_emb,
        }

    def encode_images_batch(self, images: list) -> dict:
        """
        Encode a batch of images with both encoders.
        Returns dict with 'clip' and 'siglip2' batch embeddings.
        """
        clip_embs = self.clip_encoder.encode_images_batch(images)
        siglip2_embs = self.siglip2_encoder.encode_images_batch(images)
        return {
            "clip": clip_embs,
            "siglip2": siglip2_embs,
        }

    def encode_text(self, text: str) -> dict:
        """
        Encode a text query with both CLIP and SigLIP2.
        Returns dict with 'clip' and 'siglip2' embeddings.
        """
        text = self._maybe_translate(text)
        clip_emb = self.clip_encoder.encode_text(text)
        siglip2_emb = self.siglip2_encoder.encode_text(text)
        return {
            "clip": clip_emb,
            "siglip2": siglip2_emb,
        }

    def encode_texts_batch(self, texts: list) -> dict:
        """
        Encode multiple text queries with both encoders.
        Returns dict with 'clip' and 'siglip2' batch embeddings.
        """
        texts = [self._maybe_translate(t) for t in texts]
        clip_embs = self.clip_encoder.encode_texts_batch(texts)
        siglip2_embs = self.siglip2_encoder.encode_texts_batch(texts)
        return {
            "clip": clip_embs,
            "siglip2": siglip2_embs,
        }

    async def async_encode_image(self, image: Image.Image) -> dict:
        """Asynchronously encode an image with both models running in parallel."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        clip_task = loop.run_in_executor(
            self._executor, self.clip_encoder.encode_image, image
        )
        siglip2_task = loop.run_in_executor(
            self._executor, self.siglip2_encoder.encode_image, image
        )
        clip_emb, siglip2_emb = await asyncio.gather(clip_task, siglip2_task)
        return {"clip": clip_emb, "siglip2": siglip2_emb}

    async def async_encode_text(self, text: str) -> dict:
        """Asynchronously encode text with both models running in parallel."""
        text = self._maybe_translate(text)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        clip_task = loop.run_in_executor(
            self._executor, self.clip_encoder.encode_text, text
        )
        siglip2_task = loop.run_in_executor(
            self._executor, self.siglip2_encoder.encode_text, text
        )
        clip_emb, siglip2_emb = await asyncio.gather(clip_task, siglip2_task)
        return {"clip": clip_emb, "siglip2": siglip2_emb}

    def compute_similarity(
        self,
        image_embeddings: dict,
        text_embeddings: dict,
    ) -> dict:
        """
        Compute cosine similarities for both models.
        Returns dict with 'clip_sim' and 'siglip2_sim' scores.
        """
        clip_sim = float(np.dot(image_embeddings["clip"], text_embeddings["clip"]))
        siglip2_sim = float(np.dot(image_embeddings["siglip2"], text_embeddings["siglip2"]))
        return {
            "clip": clip_sim,
            "siglip2": siglip2_sim,
        }
