"""
Vietnamese -> English query translation.

TULIP (CLIP ViT-L-14) and SigLIP2 were trained on English text, so a
Vietnamese query is tokenized into subword fragments that do not map to the
training distribution and therefore produce poor text embeddings.  This module
detects Vietnamese text and translates it to English *before* the text reaches
the dense text encoders, while leaving English queries untouched.

The BM25 / cross-encoder stages already handle Vietnamese natively (underthesea
word segmentation + multilingual BGE reranker), and the calling code keeps
passing them the ORIGINAL query string — only the dense embeddings are
translated.
"""

from __future__ import annotations

import re
import threading
from typing import Optional

from loguru import logger

# Vietnamese-only letters (including tone marks).  English never contains these,
# so this check is safe and is the primary detection signal.
_VIET_CHAR_RE = re.compile(
    r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
)

# High-frequency Vietnamese function words, used only as a fallback when the
# text has NO diacritics (telex-style input).  Requiring >= 2 hits avoids
# false positives on ambiguous single tokens like "la" / "co" / "tim".
_VIET_WORDS = {
    "và", "là", "của", "có", "những", "người", "tìm", "đang", "với", "một",
    "trên", "dưới", "trong", "ngoài", "các", "giữa", "không", "được",
    "va", "la", "cua", "co", "nhung", "nguoi", "tim", "dang", "voi", "mot",
    "tren", "duoi", "trong", "ngoai", "cac", "giua", "khong", "duoc",
}


def is_vietnamese(text: str) -> bool:
    """Heuristic Vietnamese detection (no model needed)."""
    if not text:
        return False
    if _VIET_CHAR_RE.search(text):
        return True
    words = set(re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE))
    return len(words & _VIET_WORDS) >= 2


class VietnameseEnglishTranslator:
    """Lazy-loaded MarianMT vi->en translator."""

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-vi-en",
        device: Optional[str] = "cpu",
        max_length: int = 128,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if self._failed:
                return False
            try:
                logger.info(f"Loading vi->en translation model {self.model_name} ...")
                from transformers import MarianMTModel, MarianTokenizer

                self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
                self._model = MarianMTModel.from_pretrained(self.model_name)
                if self.device and str(self.device).startswith("cuda"):
                    self._model = self._model.to(self.device)
                self._model.eval()
                logger.info("vi->en translation model loaded")
                return True
            except Exception as e:
                self._failed = True
                logger.warning(
                    f"vi->en translation model failed to load ({e}); "
                    f"falling back to original text"
                )
                return False

    def translate(self, text: str) -> str:
        if not self._load():
            return text
        try:
            import torch

            with self._lock:
                with torch.no_grad():
                    encoded = self._tokenizer(
                        [text], return_tensors="pt", truncation=True,
                        max_length=self.max_length,
                    )
                    if self.device and str(self.device).startswith("cuda"):
                        encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    generated = self._model.generate(
                        **encoded, max_new_tokens=self.max_length,
                    )
                out = self._tokenizer.batch_decode(
                    generated, skip_special_tokens=True,
                )[0].strip()
            return out or text
        except Exception as e:
            logger.warning(f"vi->en translation failed ({e}); using original text")
            return text

    def maybe_translate_to_english(self, text: str) -> str:
        return self.translate(text) if is_vietnamese(text) else text
