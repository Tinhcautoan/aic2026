"""
Multimodal Reranking Module implementing Reciprocal Rank Fusion (RRF) 
and CombMNZ (Combine Maximize Non-Zero) as described in the paper.

Pipeline:
1. Multifaceted Scoring: RRF + CombMNZ
2. Weighted Fusion: Final = α * RRF + (1-α) * CombMNZ
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from config import settings

@dataclass
class Candidate:
    """A retrieval candidate with scores from both models."""
    doc_id: str
    clip_score: float
    siglip2_score: float
    clip_rank: int = -1
    siglip2_rank: int = -1
    metadata: dict = None

class HybridReranker:
    """
    Hybrid reranker that fuses CLIP and SigLIP2 results using
    RRF (Reciprocal Rank Fusion) and CombMNZ (Combined Multiple Normalized Scores).
    
    Final score: Final(d) = α * RRF(d) + (1-α) * CombMNZ(d)
    """

    def __init__(
        self,
        rrf_k: int = None,
        fusion_alpha: float = None,
    ):
        self.rrf_k = rrf_k if rrf_k is not None else settings.RRF_K
        self.fusion_alpha = fusion_alpha if fusion_alpha is not None else settings.FUSION_ALPHA

        logger.info(
            f"HybridReranker initialized: k={self.rrf_k}, α={self.fusion_alpha}"
        )

    def _reciprocal_rank_fusion(
        self,
        clip_ranks: np.ndarray,
        siglip2_ranks: np.ndarray,
    ) -> np.ndarray:
        """
        Compute RRF scores.
        RRF(d) = Σ_m 1/(k + rank_m(d))
        
        Args:
            clip_ranks: 0-indexed ranks from CLIP
            siglip2_ranks: 0-indexed ranks from SigLIP2
        
        Returns:
            RRF scores array
        """
        clip_term = 1.0 / (self.rrf_k + clip_ranks + 1)  # +1 to make 1-indexed
        siglip2_term = 1.0 / (self.rrf_k + siglip2_ranks + 1)
        return clip_term + siglip2_term

    def _z_score_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Z-score normalization: ŝ_m(d) = (S_m(d) - μ_m) / σ_m
        Handles edge case where σ = 0.
        """
        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-8:
            return np.zeros_like(scores) if mean == 0 else scores - mean
        return (scores - mean) / std

    def _comb_mnz(
        self,
        clip_scores: np.ndarray,
        siglip2_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Compute CombMNZ scores.
        
        1. Z-score normalize each model's scores
        2. CombMNZ(d) = (Σ_m ŝ_m(d)) * count(ŝ_m(d) > 0)
        
        Args:
            clip_scores: cosine similarity scores from CLIP
            siglip2_scores: cosine similarity scores from SigLIP2
        
        Returns:
            CombMNZ scores array
        """
        # Z-score normalization
        clip_norm = self._z_score_normalize(clip_scores)
        siglip2_norm = self._z_score_normalize(siglip2_scores)

        # Sum of normalized scores
        score_sum = clip_norm + siglip2_norm

        # Count how many models have positive normalized scores (> 0)
        positive_count = (clip_norm > 0).astype(np.float32) + (siglip2_norm > 0).astype(np.float32)

        return score_sum * positive_count

    def compute_final_scores(
        self,
        clip_scores: np.ndarray,
        siglip2_scores: np.ndarray,
        clip_ranks: Optional[np.ndarray] = None,
        siglip2_ranks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the final fused scores.
        
        Final(d) = α * RRF(d) + (1-α) * CombMNZ(d)
        
        Args:
            clip_scores: cosine similarity scores from CLIP
            siglip2_scores: cosine similarity scores from SigLIP2
            clip_ranks: ranks from CLIP (0-indexed). If None, derived from scores.
            siglip2_ranks: ranks from SigLIP2 (0-indexed). If None, derived from scores.
        
        Returns:
            Final fused scores array (higher is better)
        """
        n = len(clip_scores)

        # Derive ranks from scores if not provided
        if clip_ranks is None:
            # Higher score = lower rank (rank 0 = best)
            clip_ranks = np.argsort(np.argsort(-clip_scores))
        if siglip2_ranks is None:
            siglip2_ranks = np.argsort(np.argsort(-siglip2_scores))

        # Compute RRF scores
        rrf_scores = self._reciprocal_rank_fusion(clip_ranks, siglip2_ranks)

        # Compute CombMNZ scores
        comb_mnz_scores = self._comb_mnz(clip_scores, siglip2_scores)

        # Normalize RRF scores to [0, 1] range for consistent fusion
        rrf_max = np.max(rrf_scores)
        if rrf_max > 0:
            rrf_scores = rrf_scores / rrf_max

        # Normalize CombMNZ scores to [0, 1] range
        comb_max = np.max(np.abs(comb_mnz_scores))
        if comb_max > 0:
            comb_mnz_scores = comb_mnz_scores / comb_max

        # Weighted fusion
        final_scores = (
            self.fusion_alpha * rrf_scores
            + (1.0 - self.fusion_alpha) * comb_mnz_scores
        )

        return final_scores

    def rerank(
        self,
        candidates: List[dict],
        top_k: int = None,
    ) -> List[dict]:
        """
        Rerank a list of candidates using the two-step fusion process.
        
        Each candidate dict should have:
            - 'id': unique identifier
            - 'clip_score': cosine similarity from CLIP
            - 'siglip2_score': cosine similarity from SigLIP2
            - optional metadata
        
        Args:
            candidates: list of candidate dicts with CLIP and SigLIP2 scores
            top_k: number of top results to return (default from config)
        
        Returns:
            Reranked list of candidates with added 'final_score' field
        """
        if not candidates:
            return []

        top_k = top_k or settings.TOP_K_DEFAULT
        n = len(candidates)

        # Extract scores
        clip_scores = np.array([c["clip_score"] for c in candidates], dtype=np.float32)
        siglip2_scores = np.array([c["siglip2_score"] for c in candidates], dtype=np.float32)

        # Compute final fused scores
        final_scores = self.compute_final_scores(clip_scores, siglip2_scores)

        # Attach final scores and sort
        for i, candidate in enumerate(candidates):
            candidate["final_score"] = float(final_scores[i])

        # Sort by final_score descending
        candidates.sort(key=lambda c: c["final_score"], reverse=True)

        return candidates[:top_k]

    def batch_rerank(
        self,
        batch_candidates: List[List[dict]],
        top_k: int = None,
    ) -> List[List[dict]]:
        """
        Rerank multiple query result batches.
        
        Args:
            batch_candidates: list of candidate lists for multiple queries
            top_k: number of top results per query
        
        Returns:
            List of reranked candidate lists
        """
        return [self.rerank(candidates, top_k) for candidates in batch_candidates]
