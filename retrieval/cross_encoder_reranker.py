"""
Stage 4: cross-encoder re-ranking.

WHAT CHANGED
------------
Behaviour is unchanged; instrumentation was added. This stage scores
(query, OCR+ASR+object text) pairs, so it silently does nothing whenever a
candidate carries no text -- which is most purely visual frames. There was
no way to tell how often that happened. `stats_summary()` now reports it,
and `text_hit_rate` is the number to watch: if it is low, this stage is
costing GPU time and contributing nothing, and the honest move is to turn
it off or add a visual re-ranker instead.

Note on what this stage is NOT: bge-reranker scores text against text. It
is not a vision-language re-ranker and cannot judge whether an image
matches a description. For frames whose only text is a video-level title
merged in at ingestion, it can actively push the correct frame down.
"""

from typing import List, Optional

import numpy as np
from loguru import logger

from config import settings
from .bm25_search import build_frame_document_text


class CrossEncoderReranker:
    """HuggingFace sequence-classification cross-encoder (BGE-Reranker)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name or settings.CROSS_ENCODER_MODEL_NAME
        self.model_dir = settings.CROSS_ENCODER_MODEL_DIR
        self.device = device or (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )
        self.batch_size = batch_size
        self.max_length = max_length

        self.stats = {
            "calls": 0,
            "skipped_no_text": 0,
            "candidates_scored": 0,
            "candidates_textless": 0,
        }

        if self.device.startswith("cuda"):
            try:
                t = torch.zeros(1, device=self.device)
                _ = t + t
                del t
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"CUDA validation failed for cross-encoder: {e}. Falling back to CPU.")
                self.device = "cpu"

        logger.info(f"Loading Cross-Encoder '{self.model_name}' on {self.device}...")
        model_path = self.model_dir or self.model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        if settings.USE_FP16 and "cuda" in self.device:
            try:
                self.model = self.model.half()
            except Exception as e:
                logger.warning(f"Could not cast cross-encoder to fp16: {e}")
        self.model.eval()
        logger.info("Cross-Encoder loaded")

    # ------------------------------------------------------------------ #

    def stats_summary(self) -> dict:
        """Runtime counters plus the derived hit rate.

        `text_hit_rate` is the fraction of calls where at least one candidate
        had text to score. Below ~0.3 this stage is close to dead code.
        """
        s = dict(self.stats)
        calls = max(s["calls"], 1)
        s["text_hit_rate"] = 1.0 - (s["skipped_no_text"] / calls)
        total = s["candidates_scored"] + s["candidates_textless"]
        s["candidate_text_rate"] = s["candidates_scored"] / max(total, 1)
        return s

    def reset_stats(self) -> None:
        for k in self.stats:
            self.stats[k] = 0

    # ------------------------------------------------------------------ #

    def _predict(self, pairs: List[tuple]) -> np.ndarray:
        import torch

        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                features = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**features).logits
                scores.extend(logits[:, 0].cpu().numpy().astype(np.float32).tolist())
        return np.asarray(scores, dtype=np.float32)

    def _candidate_text(self, candidate: dict) -> str:
        meta = candidate.get("metadata", {}) or {}
        return build_frame_document_text(meta) or ""

    def has_any_text(self, candidates: List[dict]) -> bool:
        """Whether running this stage can add anything at all.

        Counts the call so `text_hit_rate` reflects real pipeline usage.
        """
        self.stats["calls"] += 1
        hit = any(self._candidate_text(c) for c in candidates)
        if not hit:
            self.stats["skipped_no_text"] += 1
        return hit

    def rerank(self, query: str, candidates: List[dict], top_k: Optional[int] = None) -> List[dict]:
        if not candidates:
            return []
        top_k = top_k or len(candidates)

        texted, textless = [], []
        for c in candidates:
            (texted if self._candidate_text(c) else textless).append(c)

        self.stats["candidates_scored"] += len(texted)
        self.stats["candidates_textless"] += len(textless)

        if texted:
            pairs = [(query, self._candidate_text(c)) for c in texted]
            raw_scores = np.asarray(self._predict(pairs), dtype=np.float32)
            probs = 1.0 / (1.0 + np.exp(-raw_scores))
            for c, raw, p in zip(texted, raw_scores, probs):
                c["cross_encoder_score"] = float(raw)
                c["cross_encoder_prob"] = float(p)
            texted.sort(key=lambda c: c["cross_encoder_score"], reverse=True)

            # Textless candidates keep their pre-rerank relative order but sit
            # below every scored one. Their scores stay on the same scale so
            # downstream banding does not see a discontinuity.
            min_texted = min(c["cross_encoder_score"] for c in texted)
            textless.sort(key=lambda c: c.get("rrf_fusion_score", c.get("final_score", 0.0)), reverse=True)
            span = max(1e-6, abs(min_texted))
            for i, c in enumerate(textless):
                c["cross_encoder_score"] = min_texted - span * (0.01 * (i + 1))
                c["cross_encoder_prob"] = 0.0
                c["cross_encoder_skipped"] = True
        else:
            for c in textless:
                c["cross_encoder_score"] = c.get("rrf_fusion_score", c.get("final_score", 0.0))
                c["cross_encoder_prob"] = 0.0
                c["cross_encoder_skipped"] = True

        return (texted + textless)[:top_k]

    def score_pairs(self, pairs: List[tuple]) -> np.ndarray:
        return np.asarray(self._predict(pairs), dtype=np.float32)
