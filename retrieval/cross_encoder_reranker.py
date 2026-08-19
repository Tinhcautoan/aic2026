"""
Cross-Encoder Re-ranking Module — Stage 4 of the Textual KIS funnel.

    Dense Vector Search -> BM25 Metadata Search -> Hybrid Fusion (RRF)
    -> [Cross-Encoder Re-ranker] -> Frame Neighbor Expansion

Why this matters for the scoring formula:
    Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5
The bi-encoder stage (TULIP/SigLIP2 dense search) is good at recall — getting
the right frame somewhere in the top 100 — but weak at precision at rank 1,
because it scores query and image independently and never lets them attend
to each other. A cross-encoder jointly encodes (query, candidate) and is
much better at pushing the single best match to position 1, which is worth
R@1 = 1.0 -> contributes 0.2 to the final score outright.

We use a TEXT cross-encoder (BGE-Reranker, multilingual — handles Vietnamese
queries out of the box) rather than a vision-language cross-encoder, because:
  1. It's far cheaper to run on Kaggle T4 (no image tensors through the model).
  2. It only makes sense on the ~50 candidates surviving RRF fusion, not the
     full corpus — cost is bounded regardless of dataset size.
  3. It needs each candidate frame's text signal (OCR/ASR/object labels) —
     the same `build_frame_document_text()` used to build the BM25 index —
     as its "document" side of the (query, document) pair.

If your team also wants true vision-language re-ranking (scoring the actual
image against the query, not just its extracted text), that's a separate
component — ask and I'll add a BLIP-ITM / SigLIP-based visual re-ranker as
a 5th, optional stage.
"""

from typing import List, Dict, Optional
import numpy as np
from loguru import logger

from config import settings
from .bm25_search import build_frame_document_text


class CrossEncoderReranker:
    """
    Wraps a HuggingFace sequence-classification cross-encoder (BGE-Reranker).

    Default model: BAAI/bge-reranker-v2-m3
      - Multilingual (100+ languages incl. Vietnamese) — matches your
        Vietnamese-language query set.
      - ~568M params, fits comfortably in fp16 on a T4 (16GB) alongside
        TULIP + SigLIP2 if you unload the ones not currently in use, or
        run re-ranking as a separate stage after dense search completes.
      - max_length=512 tokens is enough for OCR+ASR+object-label text.

    Lighter alternative for tighter VRAM budgets: BAAI/bge-reranker-base
    (~278M) or cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (~120M, much
    faster, some quality loss).
    """

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

        # Validate CUDA works on this device
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
        # Load the model directly via transformers (NOT sentence-transformers).
        # sentence-transformers' CrossEncoder can fail with "Could not import
        # module 'AutoProcessor'" on hosts that ship an older sentence-transformers
        # (e.g. some vast.ai images). Loading the plain XLMRobertaForSequenceClassification
        # ourselves and scoring logits directly gives the identical result without
        # the version-sensitive dependency.
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

    def _predict(self, pairs: List[tuple]) -> np.ndarray:
        """Score (query, document) pairs -> raw logits (1-D float32 array).

        bge-reranker-v2-m3 is a single-label sequence classifier, so we return
        ``logits[:, 0]``. Callers apply sigmoid to get probabilities.
        """
        import torch

        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                features = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**features).logits  # (batch, num_labels=1)
                scores.extend(logits[:, 0].cpu().numpy().astype(np.float32).tolist())
        return np.asarray(scores, dtype=np.float32)

    def _candidate_text(self, candidate: dict) -> str:
        """Text side of the (query, candidate) pair. Reuses the same builder
        as BM25 so both stages 'see' the frame the same way."""
        meta = candidate.get("metadata", {}) or {}
        text = build_frame_document_text(meta)
        if not text:
            # No OCR/ASR/object signal at all (rare but happens on plain
            # visual scenes) — the cross-encoder can't add anything here,
            # so we fall back to letting the dense/fusion score carry it.
            return ""
        return text

    def has_any_text(self, candidates: List[dict]) -> bool:
        """
        True if at least one candidate has extractable OCR/ASR/object text.
        Call this BEFORE rerank() to decide whether running the cross-encoder
        is worthwhile at all. If your pipeline never populates per-frame
        metadata (OCR/ASR/objects), this will always be False and the
        cross-encoder stage should be skipped entirely — running it anyway
        wastes GPU time and produces a meaningless constant fallback score
        for every candidate.
        """
        return any(self._candidate_text(c) for c in candidates)

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """
        Scores every candidate against the query and re-sorts.
        Candidates with empty text keep their pre-rerank fusion rank
        (pushed to the bottom of the reranked slice, not discarded) so
        pure-visual matches aren't unfairly punished for lacking OCR/ASR.
        """
        if not candidates:
            return []
        top_k = top_k or len(candidates)

        texted, textless = [], []
        for c in candidates:
            if self._candidate_text(c):
                texted.append(c)
            else:
                textless.append(c)

        if texted:
            pairs = [(query, self._candidate_text(c)) for c in texted]
            raw_scores = self._predict(pairs)
            raw_scores = np.asarray(raw_scores, dtype=np.float32)
            probs = 1.0 / (1.0 + np.exp(-raw_scores))  # sigmoid, bge-reranker outputs logits

            for c, raw, p in zip(texted, raw_scores, probs):
                c["cross_encoder_score"] = float(raw)
                c["cross_encoder_prob"] = float(p)

            texted.sort(key=lambda c: c["cross_encoder_score"], reverse=True)

        # Textless candidates: keep their relative order from before rerank
        # by reusing their existing fusion score (rrf_fusion_score / final_score),
        # instead of inventing an arbitrary constant. A constant fallback here
        # is dangerous downstream: Stage 5 frame expansion multiplies scores
        # by a <1 decay factor, and multiplying a made-up NEGATIVE constant
        # by a decay <1 makes it LESS negative (= better), which would rank
        # synthetic expanded neighbors above real results. Reusing the actual
        # fusion score keeps everything on a consistent, meaningful scale.
        if texted:
            # Push genuinely-scored (texted) results above ALL textless ones,
            # while preserving textless candidates' relative fusion order.
            min_texted_score = min(c["cross_encoder_score"] for c in texted)
            textless.sort(key=lambda c: c.get("rrf_fusion_score", c.get("final_score", 0.0)), reverse=True)
            span = max(1e-6, abs(min_texted_score))
            for i, c in enumerate(textless):
                # Monotonically decreasing, always below min_texted_score,
                # but still ordered relative to each other by fusion score.
                c["cross_encoder_score"] = min_texted_score - span * (0.01 * (i + 1))
                c["cross_encoder_prob"] = 0.0
                c["cross_encoder_skipped"] = True
        else:
            # No candidate in this batch has any OCR/ASR/object text at all.
            # Don't fabricate a score — carry the fusion score through
            # unchanged so downstream ranking stays meaningful. (Prefer
            # calling has_any_text() before rerank() to skip this stage
            # entirely and save the GPU pass, but this keeps behavior safe
            # even if that check is skipped.)
            for c in textless:
                c["cross_encoder_score"] = c.get("rrf_fusion_score", c.get("final_score", 0.0))
                c["cross_encoder_prob"] = 0.0
                c["cross_encoder_skipped"] = True

        merged = texted + textless
        return merged[:top_k]

    def score_pairs(self, pairs: List[tuple]) -> np.ndarray:
        """Low-level batch scoring for (query, text) tuples, if you need
        raw scores outside the candidate-dict rerank() flow."""
        raw = self._predict(pairs)
        return np.asarray(raw, dtype=np.float32)
