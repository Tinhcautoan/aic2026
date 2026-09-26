"""
Fast Vietnamese -> English for query text.

PE-Core is English-only, and the translator sits on the critical path of every
Vietnamese query, so three things are done to keep it short:

  * runs on the GPU in fp16 (the base class supports it; here it is the default);
  * translates per SENTENCE with a cache: the contest reveals a description piece by
    piece, so when a sentence is appended only the new one is translated;
  * translates all missing sentences of a request (all TRAKE events, all new sentences)
    in ONE batched generate() call.

English input is passed through untouched, so a operator can type English to bypass it.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

from embedding.query_translator import VietnameseEnglishTranslator, is_vietnamese

_SENT_SPLIT = re.compile(r"(?<=[.!?;…])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]


class FastTranslator(VietnameseEnglishTranslator):
    def translate_many(self, texts: Sequence[str]) -> List[str]:
        """Translate each text (Vietnamese parts only), batching every uncached sentence."""
        per_text: List[List[str]] = [split_sentences(t) or [t] for t in texts]
        todo: List[str] = []
        seen = set()
        for sents in per_text:
            for s in sents:
                if is_vietnamese(s) and s not in self._cache and s not in seen:
                    seen.add(s)
                    todo.append(s)
        if todo and self._load():
            self._translate_batch(todo)
        out: List[str] = []
        for t, sents in zip(texts, per_text):
            parts = [self._cache.get(s, s) if is_vietnamese(s) else s for s in sents]
            out.append(" ".join(p for p in parts if p).strip() or t)
        return out

    def translate_text(self, text: str) -> str:
        return self.translate_many([text])[0]

    def _translate_batch(self, sentences: List[str]) -> None:
        import torch
        # longest-first keeps padding small; chunk so one huge request cannot blow VRAM
        order = sorted(range(len(sentences)), key=lambda i: -len(sentences[i]))
        CH = 24
        try:
            for c in range(0, len(order), CH):
                idxs = order[c:c + CH]
                batch = [sentences[i] for i in idxs]
                with self._lock, torch.inference_mode():
                    enc = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                          max_length=self.max_length)
                    if self.device and str(self.device).startswith("cuda"):
                        enc = {k: v.to(self.device) for k, v in enc.items()}
                    gen = self._model.generate(**enc, **self._gen_kwargs())
                    dec = self._tokenizer.batch_decode(gen, skip_special_tokens=True)
                for src, en in zip(batch, dec):
                    if len(self._cache) >= self._cache_max:
                        self._cache.clear()
                    self._cache[src] = en.strip() or src
        except Exception as e:                       # keep serving with the original text
            logger.warning(f"batched translation failed ({e}); using original text")
            for s in sentences:
                self._cache.setdefault(s, s)


def make_translator(device: str, num_beams: int, max_tokens: int, model_name: str, model_dir: str = "") -> FastTranslator:
    return FastTranslator(model_name=model_name, model_dir=model_dir, device=device,
                          max_length=max_tokens, num_beams=num_beams)
