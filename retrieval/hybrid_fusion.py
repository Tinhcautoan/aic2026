"""
Full Textual KIS retrieval funnel.

    Stage 1  Dense vector search (CLIP/TULIP + SigLIP2) -> dense fusion
    Stage 2  BM25 metadata search
    Stage 3  RRF fusion of dense vs sparse
    Stage 4  Cross-encoder re-rank
    Stage 5  Frame neighbour expansion
    Stage 6  Deduplication
    Stage 7  Banding, confidence, final ordering        <-- new

THE SCORING BUG THIS FILE USED TO HAVE
--------------------------------------
`final_rerank_score` was written by four different branches on four
incompatible scales:

    cross-encoder ran      c["final_rerank_score"] = cross_encoder_prob      (0, 1)
    cross-encoder skipped  c["final_rerank_score"] = rrf_fusion_score        <= 0.0328
    below fusion_top_k     c["final_rerank_score"] = rrf_fusion_score - 1e6  ~ -1e6
    no BM25 hits at all    rrf_fusion_score = final_score                    [0, 1]

Raw RRF caps at 2/(k+1) = 2/61 = 0.0328 with k=60. The cross-encoder only
runs when a candidate carries OCR/ASR/object text, which most purely visual
frames do not. So on the common path the number the UI displayed as
"confidence" could not exceed 0.033, and candidates past the rerank cutoff
displayed as roughly minus one million. That is the entire cause of
"confidence is always extremely low". Nothing was wrong with the models.

Stage 7 now separates the two concepts:

    final_score   ordering only, in [0, 1], banded per tier so a
                  cross-encoded hit always outranks a merely-fused one
    confidence    calibrated, in [0, 1], derived from the candidate's
                  separation from the rest of this query's pool, using the
                  dense logits every candidate has -- so it no longer
                  depends on whether the cross-encoder happened to fire

`final_rerank_score` is kept as an alias of `final_score` so existing
callers (dedup, the API, the frontend) keep working.
"""

from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from config import settings
from . import scoring
from .bm25_search import BM25Index
from .cross_encoder_reranker import CrossEncoderReranker
from .reranker import HybridReranker
from .result_dedup import deduplicate_results
from .scoring import TIER_CROSS_ENCODED, TIER_EXPANDED, TIER_FUSED


def _frame_key(c: dict) -> str:
    return f"{c['video_id']}_{c['frame_idx']}"


def _decay_score(score: float, decay: float) -> float:
    """Worsen a score multiplicatively.

    Scores are non-negative after Stage 7 banding, so this is a plain
    multiply. The old sign-dependent branch existed because scores could be
    negative (z-scored fusion, raw cross-encoder logits); that can no longer
    happen, but the guard is cheap and keeps the function safe if a caller
    hands it a raw logit.
    """
    return score * decay if score >= 0 else score / decay


def _merge_expansion_preserving_rank(
    base_results: List[dict],
    expanded_neighbors: List[dict],
) -> List[dict]:
    """Insert each synthetic neighbour directly after the hit it came from,
    without letting it jump above a higher-scoring real result."""
    neighbor_by_source: Dict[str, List[dict]] = {}
    real_keys = {_frame_key(r) for r in base_results}

    for n in expanded_neighbors:
        if not n.get("is_expanded_neighbor"):
            continue
        if _frame_key(n) in real_keys:
            continue
        if n.get("expanded_from_frame") is None:
            continue
        src_key = f"{n['video_id']}_{n['expanded_from_frame']}"
        neighbor_by_source.setdefault(src_key, []).append(n)

    for neighbors in neighbor_by_source.values():
        neighbors.sort(
            key=lambda c: c.get("final_rerank_score", c.get("rrf_fusion_score", 0.0)),
            reverse=True,
        )

    merged: List[dict] = []
    inserted_keys: set = set()
    for r in base_results:
        merged.append(r)
        inserted_keys.add(_frame_key(r))
        for n in neighbor_by_source.get(f"{r['video_id']}_{r['frame_idx']}", []):
            nkey = _frame_key(n)
            if nkey in inserted_keys:
                continue
            merged.append(n)
            inserted_keys.add(nkey)
    return merged


class HybridFusionPipeline:
    def __init__(
        self,
        vector_store,
        bm25_index: Optional[BM25Index] = None,
        dense_reranker: Optional[HybridReranker] = None,
        cross_encoder: Optional[CrossEncoderReranker] = None,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.dense_reranker = dense_reranker or HybridReranker()
        self.cross_encoder = cross_encoder
        self.last_audit: Optional[scoring.ScoreAudit] = None

    # ------------------------------------------------------------------ #

    def _rrf_merge_dense_sparse(
        self,
        dense_candidates: List[dict],
        sparse_candidates: List[dict],
        k: Optional[int] = None,
    ) -> List[dict]:
        """RRF across two retrievers whose scores are not comparable.

        RRF is the right tool here precisely because dense cosine and BM25
        term weights live on unrelated scales. It is the wrong tool inside
        Stage 1, where both branches are cosines -- that fusion is handled by
        HybridReranker instead.

        The raw RRF value is tiny by construction (<= 2/(k+1)). It is stored
        as `rrf_fusion_score` for diagnostics and used only for ordering;
        Stage 7 is what turns ordering into a displayable number.
        """
        k = settings.RRF_K if k is None else k

        dense_sorted = sorted(dense_candidates, key=lambda c: c.get("final_score", 0.0), reverse=True)
        sparse_sorted = sorted(sparse_candidates, key=lambda c: c.get("bm25_score", 0.0), reverse=True)

        dense_rank = {_frame_key(c): i for i, c in enumerate(dense_sorted)}
        sparse_rank = {_frame_key(c): i for i, c in enumerate(sparse_sorted)}

        merged: Dict[str, dict] = {}
        for c in dense_sorted:
            merged[_frame_key(c)] = dict(c)
        for c in sparse_sorted:
            key = _frame_key(c)
            if key not in merged:
                merged[key] = dict(c)
            else:
                merged[key]["bm25_score"] = c.get("bm25_score", 0.0)
                if not merged[key].get("metadata"):
                    merged[key]["metadata"] = c.get("metadata")

        fallback_rank = len(dense_sorted) + len(sparse_sorted) + 1000
        for key in set(dense_rank) | set(sparse_rank):
            dr = dense_rank.get(key, fallback_rank)
            sr = sparse_rank.get(key, fallback_rank)
            merged[key]["rrf_fusion_score"] = 1.0 / (k + dr + 1) + 1.0 / (k + sr + 1)

        results = list(merged.values())
        results.sort(key=lambda c: c["rrf_fusion_score"], reverse=True)
        return results

    # ------------------------------------------------------------------ #

    def _expand_neighbors(
        self,
        results: List[dict],
        windows: Optional[List[int]] = None,
        decay: float = 0.85,
        *,
        return_neighbors_only: bool = False,
    ) -> List[dict]:
        windows = windows or settings.FRAME_EXPANSION_WINDOWS
        expanded: Dict[str, dict] = {} if return_neighbors_only else {_frame_key(r): r for r in results}
        neighbors_only: List[dict] = []

        for r in results:
            base = r.get("_raw_score", r.get("final_rerank_score", r.get("rrf_fusion_score", 0.0)))
            for w in windows:
                for sign in (-1, 1):
                    neighbor_idx = r["frame_idx"] + sign * w
                    if neighbor_idx < 0:
                        continue
                    key = f"{r['video_id']}_{neighbor_idx}"
                    if key in expanded:
                        continue
                    neighbor = {
                        **r,
                        "frame_idx": neighbor_idx,
                        "_raw_score": _decay_score(float(base), decay),
                        "tier": TIER_EXPANDED,
                        "is_expanded_neighbor": True,
                        "expanded_from_frame": r["frame_idx"],
                    }
                    expanded[key] = neighbor
                    neighbors_only.append(neighbor)

        if return_neighbors_only:
            return neighbors_only
        out = list(expanded.values())
        out.sort(key=lambda c: c.get("_raw_score", 0.0), reverse=True)
        return out

    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        clip_query_emb: np.ndarray,
        siglip2_query_emb: np.ndarray,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        fusion_top_k: Optional[int] = None,
        cross_encoder_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        expr: Optional[str] = None,
        use_bm25: bool = True,
        use_cross_encoder: bool = True,
        expand_neighbors: bool = True,
        dedup: Optional[bool] = None,
    ) -> List[dict]:
        """Run the funnel.

        Every returned candidate carries:
            clip_score, siglip2_score        raw cosines (Stage 1)
            dense_logit                      calibrated, model-native (Stage 1)
            final_score (dense fusion)       [0, 1]
            bm25_score                       if the sparse branch matched
            rrf_fusion_score                 diagnostics (Stage 3)
            cross_encoder_score / _prob      if Stage 4 ran
            tier                             0 cross-encoded, 1 fused, 2 synthetic
            final_score / final_rerank_score ordering key, [0, 1]
            confidence                       calibrated, [0, 1]
            confidence_basis                 which signal confidence came from
        """
        if dedup is None:
            dedup = settings.RESULT_DEDUP_ENABLED
        dense_top_k = dense_top_k or settings.DENSE_TOP_K
        sparse_top_k = sparse_top_k or settings.SPARSE_TOP_K
        fusion_top_k = fusion_top_k or settings.FUSION_TOP_K
        cross_encoder_top_k = cross_encoder_top_k or settings.CROSS_ENCODER_TOP_K
        final_top_k = final_top_k or settings.FINAL_TOP_K

        # ---- Stage 1: dense retrieval + dense fusion ----
        dense_candidates = self.vector_store.hybrid_search(
            clip_query=clip_query_emb, siglip2_query=siglip2_query_emb,
            top_k=dense_top_k, expr=expr,
        )
        dense_candidates = self.dense_reranker.rerank(dense_candidates, top_k=dense_top_k)

        # ---- Stage 2: sparse BM25 ----
        sparse_candidates: List[dict] = []
        if use_bm25 and self.bm25_index is not None and self.bm25_index.size > 0:
            sparse_candidates = self.bm25_index.search(query, top_k=sparse_top_k)
            if expr and sparse_candidates:
                import re
                m = re.search(r'video_id\s*==\s*"([^"]+)"', expr)
                if m:
                    vid = m.group(1)
                    sparse_candidates = [c for c in sparse_candidates if c["video_id"] == vid]

        # ---- Stage 3: RRF across dense and sparse ----
        if sparse_candidates:
            fused = self._rrf_merge_dense_sparse(dense_candidates, sparse_candidates)
        else:
            fused = dense_candidates
            for c in fused:
                c["rrf_fusion_score"] = c.get("final_score", 0.0)

        top_for_rerank = fused[:fusion_top_k]

        # ---- Stage 4: cross-encoder ----
        has_text = bool(top_for_rerank) and self.cross_encoder is not None \
            and self.cross_encoder.has_any_text(top_for_rerank)
        run_cross_encoder = use_cross_encoder and self.cross_encoder is not None and has_text

        if use_cross_encoder and self.cross_encoder is not None and top_for_rerank and not run_cross_encoder:
            logger.info(
                f"Stage 4 skipped: none of {len(top_for_rerank)} candidates carry OCR/ASR/object "
                f"text. Ranking falls back to dense+sparse fusion. If this is frequent, the "
                f"cross-encoder is contributing nothing and per-frame metadata needs populating."
            )

        if run_cross_encoder:
            reranked = self.cross_encoder.rerank(query, top_for_rerank, top_k=cross_encoder_top_k)
            for c in reranked:
                c["tier"] = TIER_CROSS_ENCODED
                c["_raw_score"] = float(
                    c.get("cross_encoder_prob", 1.0 / (1.0 + np.exp(-c["cross_encoder_score"])))
                )
            remaining = fused[fusion_top_k:]
            for c in remaining:
                # Previously: rrf_fusion_score - 1e6. Tiers make that hack
                # unnecessary; a lower tier already sorts below, and the
                # score stays inside [0, 1] so nothing displays as -1e6.
                c["tier"] = TIER_FUSED
                c["_raw_score"] = float(c.get("rrf_fusion_score", 0.0))
            final_ranked = list(reranked) + list(remaining)
        else:
            final_ranked = fused
            for c in final_ranked:
                c["tier"] = TIER_FUSED
                c["_raw_score"] = float(c.get("rrf_fusion_score", 0.0))

        # ---- Stage 5: frame neighbour expansion ----
        should_expand = expand_neighbors and (
            settings.FRAME_EXPANSION_FOR_UI or final_top_k > 20
        )
        if should_expand:
            sources = final_ranked[: settings.FRAME_EXPANSION_MAX_SOURCES]
            neighbors = self._expand_neighbors(sources, return_neighbors_only=True)
            final_ranked = _merge_expansion_preserving_rank(final_ranked[:final_top_k], neighbors)

        # ---- Stage 7 (part 1): band scores and attach confidence ----
        # Done before dedup so dedup sees a coherent, non-negative ordering.
        final_ranked = scoring.finalize(final_ranked, score_field="final_score")
        for c in final_ranked:
            c["final_rerank_score"] = c["final_score"]   # backwards-compatible alias

        # ---- Stage 6: deduplication ----
        if dedup:
            final_ranked = deduplicate_results(final_ranked, top_k=final_top_k)

        out = final_ranked[:final_top_k]
        self.last_audit = scoring.audit(out)
        if self.last_audit.flat_pool:
            logger.warning(
                f"Flat confidence pool for query {query!r}: the retriever did not "
                f"discriminate between candidates. conf={self.last_audit.conf_max:.3f}"
            )
        return out

    # ------------------------------------------------------------------ #

    def diagnostics(self) -> dict:
        """Everything needed to explain the last query's scores."""
        d = {"audit": self.last_audit.as_dict() if self.last_audit else None}
        if self.cross_encoder is not None and hasattr(self.cross_encoder, "stats_summary"):
            d["cross_encoder"] = self.cross_encoder.stats_summary()
        return d
