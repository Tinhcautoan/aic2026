"""
Stage 1 fusion: combine the two dense encoders into one ranking.

CHANGES FROM THE PREVIOUS VERSION
---------------------------------
1. CombMNZ used z-score normalisation. After z-scoring, the mean is 0, so
   roughly half the pool is negative on both encoders, which made
   `positive_count` zero and collapsed CombMNZ to exactly 0.0 for all of
   them. Those mediocre candidates then sorted ABOVE any candidate with a
   negative sum, and a large block of exact ties formed at 0.0. CombMNZ
   assumes non-negative normalised inputs; it now uses min-max.

2. The multiplier is floored at 1.0 so a candidate can never be multiplied
   into oblivion. Since `hybrid_search` returns the same candidate set for
   both encoders, the count is always 2 and CombMNZ correctly degenerates
   to CombSUM.

3. Emits `dense_logit`: the two cosines mapped into the models' own logit
   space and blended. This is the signal the confidence layer reads, and it
   is why confidence no longer depends on whether the cross-encoder happened
   to run.

The public API (`HybridReranker.rerank`, `compute_final_scores`) is
unchanged so `hybrid_fusion.py` and `evaluation/metrics.py` keep working.
"""

from typing import List, Optional

import numpy as np
from loguru import logger

from config import settings
from .scoring import minmax, zscore, calibrated_similarity


class HybridReranker:
    """Fuses CLIP/TULIP and SigLIP2 rankings.

    Final(d) = alpha * RRF(d) + (1 - alpha) * CombMNZ(d), both in [0, 1].
    """

    def __init__(
        self,
        rrf_k: Optional[int] = None,
        fusion_alpha: Optional[float] = None,
        clip_logit_scale: Optional[float] = None,
        siglip2_logit_scale: Optional[float] = None,
        siglip2_logit_bias: Optional[float] = None,
    ):
        self.rrf_k = rrf_k if rrf_k is not None else settings.RRF_K
        self.fusion_alpha = fusion_alpha if fusion_alpha is not None else settings.FUSION_ALPHA

        # Model-native temperatures. Defaults match CLIP's trained value and
        # SigLIP's published bias; override from the loaded encoders at
        # startup via `bind_encoders()` for exact numbers.
        self.clip_logit_scale = clip_logit_scale if clip_logit_scale is not None else 100.0
        self.siglip2_logit_scale = siglip2_logit_scale if siglip2_logit_scale is not None else 100.0
        self.siglip2_logit_bias = siglip2_logit_bias if siglip2_logit_bias is not None else 0.0

        self.clip_weight = float(getattr(settings, "DENSE_LOGIT_CLIP_WEIGHT", 0.5))

        logger.info(
            f"HybridReranker: k={self.rrf_k}, alpha={self.fusion_alpha}, "
            f"clip_scale={self.clip_logit_scale:.1f}, "
            f"siglip2_scale={self.siglip2_logit_scale:.1f}/bias={self.siglip2_logit_bias:.2f}"
        )

    def bind_encoders(self, clip_encoder=None, siglip2_encoder=None) -> None:
        """Read the real logit_scale / logit_bias off the loaded models.

        Call once at startup. Without this the reranker uses CLIP's default
        temperature of 100 for both, which is close enough for ordering but
        not for cross-encoder-free confidence.
        """
        from .scoring import get_logit_params

        if clip_encoder is not None and getattr(clip_encoder, "model", None) is not None:
            self.clip_logit_scale, _ = get_logit_params(clip_encoder.model)
        if siglip2_encoder is not None and getattr(siglip2_encoder, "model", None) is not None:
            self.siglip2_logit_scale, self.siglip2_logit_bias = get_logit_params(
                siglip2_encoder.model
            )
        logger.info(
            f"Reranker bound to encoders: clip_scale={self.clip_logit_scale:.2f}, "
            f"siglip2_scale={self.siglip2_logit_scale:.2f}, bias={self.siglip2_logit_bias:.3f}"
        )

    # ------------------------------------------------------------------ #

    def _reciprocal_rank_fusion(self, clip_ranks: np.ndarray, siglip2_ranks: np.ndarray) -> np.ndarray:
        return 1.0 / (self.rrf_k + clip_ranks + 1) + 1.0 / (self.rrf_k + siglip2_ranks + 1)

    @staticmethod
    def _z_score_normalize(scores: np.ndarray) -> np.ndarray:
        """Kept for backwards compatibility; no longer used by CombMNZ."""
        return zscore(scores)

    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        return minmax(scores)

    def _comb_mnz(self, clip_scores: np.ndarray, siglip2_scores: np.ndarray) -> np.ndarray:
        """CombMNZ over min-max normalised scores.

        Multiplier is the number of branches that produced a positive
        normalised score, floored at 1.0. Both branches always return the
        same candidate set here, so in practice this is CombSUM -- which is
        the correct behaviour, not a degradation.
        """
        clip_norm = minmax(clip_scores)
        siglip2_norm = minmax(siglip2_scores)
        score_sum = clip_norm + siglip2_norm
        count = (clip_norm > 0).astype(np.float64) + (siglip2_norm > 0).astype(np.float64)
        return score_sum * np.maximum(count, 1.0)

    def dense_logits(self, clip_scores: np.ndarray, siglip2_scores: np.ndarray) -> np.ndarray:
        """Blend the two encoders in their own logit space.

        Raw cosine is not comparable between a softmax-trained CLIP and a
        sigmoid-trained SigLIP, and neither is on a scale a human can read.
        Applying each model's `logit_scale` (and SigLIP's `logit_bias`) puts
        them on the scale the models were actually trained to produce.
        """
        clip_logit = calibrated_similarity(clip_scores, self.clip_logit_scale, 0.0)
        siglip2_logit = calibrated_similarity(
            siglip2_scores, self.siglip2_logit_scale, self.siglip2_logit_bias
        )
        w = self.clip_weight
        return w * clip_logit + (1.0 - w) * siglip2_logit

    def compute_final_scores(
        self,
        clip_scores: np.ndarray,
        siglip2_scores: np.ndarray,
        clip_ranks: Optional[np.ndarray] = None,
        siglip2_ranks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fused score in [0, 1]. Higher is better."""
        clip_scores = np.asarray(clip_scores, dtype=np.float64)
        siglip2_scores = np.asarray(siglip2_scores, dtype=np.float64)

        if clip_ranks is None:
            clip_ranks = np.argsort(np.argsort(-clip_scores))
        if siglip2_ranks is None:
            siglip2_ranks = np.argsort(np.argsort(-siglip2_scores))

        rrf = minmax(self._reciprocal_rank_fusion(clip_ranks, siglip2_ranks))
        comb = minmax(self._comb_mnz(clip_scores, siglip2_scores))

        final = self.fusion_alpha * rrf + (1.0 - self.fusion_alpha) * comb
        return np.clip(final, 0.0, 1.0)

    def rerank(self, candidates: List[dict], top_k: Optional[int] = None) -> List[dict]:
        """Attach `final_score` (ordering, [0,1]) and `dense_logit`
        (calibrated evidence, used by the confidence layer) then sort."""
        if not candidates:
            return []

        top_k = top_k or settings.TOP_K_DEFAULT

        clip_scores = np.array([c.get("clip_score", 0.0) for c in candidates], dtype=np.float64)
        siglip2_scores = np.array([c.get("siglip2_score", 0.0) for c in candidates], dtype=np.float64)

        final_scores = self.compute_final_scores(clip_scores, siglip2_scores)
        logits = self.dense_logits(clip_scores, siglip2_scores)

        for i, c in enumerate(candidates):
            c["final_score"] = float(final_scores[i])
            c["dense_logit"] = float(logits[i])

        candidates.sort(key=lambda c: c["final_score"], reverse=True)
        return candidates[:top_k]

    def batch_rerank(self, batch_candidates: List[List[dict]], top_k: Optional[int] = None):
        return [self.rerank(c, top_k) for c in batch_candidates]
