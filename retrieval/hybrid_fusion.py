"""
Full Textual KIS Retrieval Funnel.

    Dense Vector Search (TULIP+SigLIP2) -> BM25 Metadata Search
    -> Hybrid Fusion (RRF) -> Cross-Encoder Re-ranker
    -> Frame Neighbor Expansion (+-5, +-10 frames)

This module is the single entry point the API should call for a KIS text
query. It does NOT own the models — pass in already-initialized components
so they're loaded once at server startup, not per-request.

Scoring strategy notes (matches the AIC scoring rule:
Final = (R@1+R@5+R@20+R@50+R@100)/5):
  - Stage 1-3 optimize for RECALL: cast a wide net (dense_top_k + sparse_top_k
    candidates), so the true answer is *somewhere* in the pool before we
    spend the expensive cross-encoder pass.
  - Stage 4 optimizes for PRECISION AT RANK 1: the cross-encoder only runs
    on the much smaller `fusion_top_k` survivors, re-ordering them so the
    single best match lands at position 1.
  - Stage 5 hedges against off-by-a-few-frames ground truth: expands each
    surviving top result to its +-5/+-10 frame neighbors so the exact
    annotated frame is very likely present somewhere in the submitted
    top-100, even if the keyframe extractor picked a slightly different
    frame than the one BTC's ground truth points to.
"""

from typing import List, Dict, Optional
import numpy as np
from loguru import logger

from config import settings
from .reranker import HybridReranker
from .bm25_search import BM25Index
from .cross_encoder_reranker import CrossEncoderReranker
from .result_dedup import deduplicate_results


def _frame_key(c: dict) -> str:
    return f"{c['video_id']}_{c['frame_idx']}"


def _decay_score(score: float, decay: float) -> float:
    """
    Multiplicatively worsen a score by `decay` (0 < decay < 1), so a Stage-5
    expanded neighbor always ranks strictly below the real result it came
    from — regardless of the score's sign.

    Naively doing `score * decay` breaks for negative scores (e.g. raw
    cross-encoder logits, or z-score-normalized fusion scores): multiplying
    a negative number by something <1 makes it LESS negative, i.e. BETTER,
    which inverts the intended guarantee and lets synthetic neighbors
    outrank genuine results. Dividing negative scores by `decay` instead
    correctly makes them more negative (worse).
    """
    return score * decay if score >= 0 else score / decay


def _merge_expansion_preserving_rank(
    base_results: List[dict],
    expanded_neighbors: List[dict],
) -> List[dict]:
    """
    Insert synthetic neighbors immediately after their source hit without
    letting them outrank higher-scoring real results above the source.
    """
    neighbor_by_source: Dict[str, List[dict]] = {}
    real_keys = {_frame_key(r) for r in base_results}

    for n in expanded_neighbors:
        if not n.get("is_expanded_neighbor"):
            continue
        key = _frame_key(n)
        if key in real_keys:
            continue
        src_key = f"{n['video_id']}_{n.get('expanded_from_frame')}"
        if n.get("expanded_from_frame") is None:
            continue
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
        src_key = f"{r['video_id']}_{r['frame_idx']}"
        for n in neighbor_by_source.get(src_key, []):
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
        """
        vector_store: MilvusVectorStore or LocalVectorStore (must implement
            .hybrid_search(clip_query, siglip2_query, top_k, expr))
        bm25_index: BM25Index, optional. If None or empty, pipeline falls
            back to dense-only retrieval — never errors out.
        dense_reranker: HybridReranker (RRF+CombMNZ over CLIP/TULIP+SigLIP2).
            Created if not passed in.
        cross_encoder: CrossEncoderReranker, optional. If None, Stage 4 is
            skipped and fusion-ranked order is kept — useful for a fast
            "no cross-encoder" mode during early dev/testing on Kaggle.
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.dense_reranker = dense_reranker or HybridReranker()
        self.cross_encoder = cross_encoder

    # ------------------------------------------------------------------ #
    # Stage 3: second-level RRF fusion between dense and sparse result sets
    # ------------------------------------------------------------------ #

    def _rrf_merge_dense_sparse(
        self,
        dense_candidates: List[dict],
        sparse_candidates: List[dict],
        k: int = 60,
    ) -> List[dict]:
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
    # Stage 5: frame neighbor expansion
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
            base_score = r.get("final_rerank_score", r.get("rrf_fusion_score", 0.0))
            for w in windows:
                for sign in (-1, 1):
                    neighbor_idx = r["frame_idx"] + sign * w
                    if neighbor_idx < 0:
                        continue
                    key = f"{r['video_id']}_{neighbor_idx}"
                    if key in expanded:
                        continue  # real retrieved result takes priority over a synthetic neighbor
                    neighbor = {
                        **r,
                        "frame_idx": neighbor_idx,
                        "final_rerank_score": _decay_score(base_score, decay),
                        "is_expanded_neighbor": True,
                        "expanded_from_frame": r["frame_idx"],
                    }
                    expanded[key] = neighbor
                    neighbors_only.append(neighbor)

        if return_neighbors_only:
            return neighbors_only

        out = list(expanded.values())
        out.sort(key=lambda c: c.get("final_rerank_score", c.get("rrf_fusion_score", 0.0)), reverse=True)
        return out

    # ------------------------------------------------------------------ #
    # Public entry point
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
        """
        Args mirror the 5 funnel stages. All top-k defaults come from
        settings so you can tune them centrally without touching call sites.

        Returns a list of candidate dicts, best first, each carrying:
          clip_score, siglip2_score, final_score        (Stage 1, dense fusion)
          bm25_score                                     (Stage 2, if matched)
          rrf_fusion_score                               (Stage 3)
          cross_encoder_score / cross_encoder_prob       (Stage 4, if run)
          final_rerank_score                             (what it's sorted by)
          is_expanded_neighbor, expanded_from_frame       (Stage 5, if synthetic)
        """
        if dedup is None:
            dedup = settings.RESULT_DEDUP_ENABLED
        dense_top_k = dense_top_k or settings.DENSE_TOP_K
        sparse_top_k = sparse_top_k or settings.SPARSE_TOP_K
        fusion_top_k = fusion_top_k or settings.FUSION_TOP_K
        cross_encoder_top_k = cross_encoder_top_k or settings.CROSS_ENCODER_TOP_K
        final_top_k = final_top_k or settings.FINAL_TOP_K

        # ---- Stage 1: Dense retrieval + first-level RRF/CombMNZ fusion ----
        dense_candidates = self.vector_store.hybrid_search(
            clip_query=clip_query_emb, siglip2_query=siglip2_query_emb,
            top_k=dense_top_k, expr=expr,
        )
        dense_candidates = self.dense_reranker.rerank(dense_candidates, top_k=dense_top_k)

        # ---- Stage 2: Sparse BM25 retrieval ----
        sparse_candidates: List[dict] = []
        if use_bm25 and self.bm25_index is not None and self.bm25_index.size > 0:
            sparse_candidates = self.bm25_index.search(query, top_k=sparse_top_k)
            if expr and sparse_candidates:
                # BM25Index has no expr filtering built in; apply video_filter here.
                # expr is expected in the form 'video_id == "xyz"' set by the API layer.
                import re
                m = re.search(r'video_id\s*==\s*"([^"]+)"', expr)
                if m:
                    vid = m.group(1)
                    sparse_candidates = [c for c in sparse_candidates if c["video_id"] == vid]

        # ---- Stage 3: Second-level RRF fusion (dense vs sparse) ----
        if sparse_candidates:
            fused = self._rrf_merge_dense_sparse(dense_candidates, sparse_candidates)
        else:
            fused = dense_candidates
            for c in fused:
                c["rrf_fusion_score"] = c.get("final_score", 0.0)

        top_for_rerank = fused[:fusion_top_k]

        # ---- Stage 4: Cross-Encoder re-ranking (maximize R@1) ----
        run_cross_encoder = (
            use_cross_encoder
            and self.cross_encoder is not None
            and bool(top_for_rerank)
            and self.cross_encoder.has_any_text(top_for_rerank)
        )
        if use_cross_encoder and self.cross_encoder is not None and top_for_rerank and not run_cross_encoder:
            logger.debug(
                "Cross-Encoder stage skipped: no candidate in this batch has "
                "OCR/ASR/object text (metadata is empty). Populate per-frame "
                "metadata during ingestion for this stage to add value."
            )
        if run_cross_encoder:
            reranked = self.cross_encoder.rerank(query, top_for_rerank, top_k=cross_encoder_top_k)
            for c in reranked:
                c["final_rerank_score"] = c["cross_encoder_score"]
            remaining = fused[fusion_top_k:]
            for c in remaining:
                # Push everything the cross-encoder didn't see behind the
                # reranked slice, but preserve relative fusion order.
                c["final_rerank_score"] = c.get("rrf_fusion_score", 0.0) - 1e6
            final_ranked = reranked + remaining
        else:
            final_ranked = fused
            for c in final_ranked:
                c["final_rerank_score"] = c.get("rrf_fusion_score", 0.0)

        # ---- Stage 5: Frame neighbor expansion ----
        should_expand = expand_neighbors and (
            settings.FRAME_EXPANSION_FOR_UI or final_top_k > 20
        )
        if should_expand:
            sources = final_ranked[: settings.FRAME_EXPANSION_MAX_SOURCES]
            neighbors = self._expand_neighbors(sources, return_neighbors_only=True)
            final_ranked = _merge_expansion_preserving_rank(final_ranked[:final_top_k], neighbors)

        # ---- Stage 6: Result deduplication ----
        if dedup:
            final_ranked = deduplicate_results(final_ranked, top_k=final_top_k)

        return final_ranked[:final_top_k]
