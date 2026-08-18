"""
BM25 Sparse Search Module — Metadata-based Retrieval.

Complements dense vector search (TULIP/SigLIP2) with exact keyword matching
over per-frame text metadata: OCR text, ASR transcript snippets, and
detected-object labels (Faster R-CNN Objects JSON provided by BTC).

This is Stage 2 of the Textual KIS funnel:
    Dense Vector Search -> [BM25 Metadata Search] -> Hybrid Fusion (RRF)
    -> Cross-Encoder Re-ranker -> Frame Neighbor Expansion

Design notes for AIC 2026 scale (70-80GB, millions of keyframes):
- rank_bm25's BM25Okapi is pure-Python and holds the full corpus in RAM.
  For >200k documents this becomes slow to build/query. If your dataset
  crosses that threshold, swap this for Elasticsearch/OpenSearch (see
  `ElasticBM25Index` stub at the bottom) without changing the calling code
  in hybrid_fusion.py — both expose the same `.search()` interface.
- The index is picklable and persisted to disk so it survives Kaggle
  session restarts without needing to re-tokenize the whole corpus.
"""

import os
import re
import pickle
from typing import List, Dict, Optional, Callable
import numpy as np
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

from config import settings


# Keep Vietnamese diacritics (Latin Extended Additional block) when tokenizing.
_TOKEN_RE = re.compile(r"[^\w\u00C0-\u1EF9]+", re.UNICODE)

# Minimal Vietnamese + English stopword list. Extend as needed; keeping it
# small on purpose — over-aggressive stopword removal hurts short OCR queries.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "va", "va_", "la", "cua", "va", "nhung", "mot", "cac", "nhung", "va",
    "và", "là", "của", "một", "các", "những", "và", "có", "được", "này",
}


def simple_tokenize(text: str) -> List[str]:
    """Lowercase + split on non-word chars, keep Vietnamese diacritics, drop stopwords."""
    if not text:
        return []
    text = text.lower().strip()
    tokens = [t for t in _TOKEN_RE.split(text) if t and t not in _STOPWORDS]
    return tokens


def vietnamese_word_segment_tokenize(text: str) -> List[str]:
    """
    Optional: word-segmented tokenizer using underthesea/pyvi for proper
    Vietnamese compound-word handling (e.g. "quần_áo" instead of "quần", "áo").
    Falls back to simple_tokenize if the library isn't installed.
    Enable via settings.BM25_USE_VN_SEGMENTATION = True.
    """
    if not text:
        return []
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text.lower(), format="text")
        return simple_tokenize(segmented)
    except ImportError:
        logger.warning("underthesea not installed, falling back to simple_tokenize")
        return simple_tokenize(text)


def build_frame_document_text(metadata: dict) -> str:
    """
    Combine all textual signals available for one keyframe into a single
    BM25 document. Called both when building the index and (by the
    cross-encoder module) when re-ranking, so keep this the single
    source of truth for "what text represents this frame".

    Expected metadata keys (all optional, fill in whatever BTC's JSON has):
        - ocr_text: str
        - asr_transcript: str (segment near this frame's timestamp, not whole video)
        - objects: list[str] or list[dict] with a 'label'/'class' key (Faster R-CNN)
        - caption: str (if you generate captions separately)
        - video_title / video_description: str
    """
    parts = []
    if metadata.get("ocr_text"):
        parts.append(str(metadata["ocr_text"]))
    if metadata.get("asr_transcript"):
        parts.append(str(metadata["asr_transcript"]))
    objects = metadata.get("objects")
    if objects:
        if isinstance(objects, list):
            labels = []
            for o in objects:
                if isinstance(o, dict):
                    labels.append(str(o.get("label") or o.get("class") or o.get("name") or ""))
                else:
                    labels.append(str(o))
            parts.append(", ".join(l for l in labels if l))
        else:
            parts.append(str(objects))
    if metadata.get("caption"):
        parts.append(str(metadata["caption"]))
    if metadata.get("video_title"):
        parts.append(str(metadata["video_title"]))
    return " | ".join(p for p in parts if p)


class BM25Index:
    """
    In-memory BM25 index over per-frame text documents.
    One document = one keyframe (video_id, frame_idx).
    """

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        if BM25Okapi is None:
            raise ImportError(
                "rank_bm25 not installed. Run: pip install rank-bm25 --break-system-packages"
            )
        use_vn = getattr(settings, "BM25_USE_VN_SEGMENTATION", False)
        self.tokenizer = tokenizer or (vietnamese_word_segment_tokenize if use_vn else simple_tokenize)

        self._bm25: Optional[BM25Okapi] = None
        self._doc_keys: List[str] = []          # "video_id_frame_idx"
        self._doc_meta: List[dict] = []          # {'video_id', 'frame_idx', 'metadata': {...}}
        self._tokenized_corpus: List[List[str]] = []
        self._inverted_index: Dict[str, set] = {}  # token -> set(doc indices), for overlap filtering

    # ---------------------------------------------------------------- #
    # Build
    # ---------------------------------------------------------------- #

    def add_documents(self, records: List[dict]) -> None:
        """
        records: list of dicts, each with:
            - 'video_id': str
            - 'frame_idx': int
            - 'metadata': dict (passed to build_frame_document_text)
        Call `.build()` after adding all documents (or in batches for very
        large corpora — rebuilding BM25Okapi is O(n), so batch adds then
        one final build() rather than build()-ing after every add).
        """
        for r in records:
            text = build_frame_document_text(r.get("metadata", {}))
            self._doc_keys.append(f"{r['video_id']}_{r['frame_idx']}")
            self._doc_meta.append({
                "video_id": r["video_id"],
                "frame_idx": r["frame_idx"],
                "metadata": r.get("metadata", {}),
            })
            self._tokenized_corpus.append(self.tokenizer(text))

    def build(self) -> None:
        if not self._tokenized_corpus:
            logger.warning("BM25Index.build() called with 0 documents — index stays empty")
            self._bm25 = None
            self._inverted_index = {}
            return
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        # Inverted index: token -> doc indices. Used to filter candidates by
        # genuine term overlap, rather than by score sign. rank_bm25's IDF
        # floor (eps * average_idf) can itself go negative on tiny/degenerate
        # corpora (e.g. a term present in every single document), which would
        # make a naive `score > 0` filter wrongly drop real matches. This
        # never happens in practice on a real multi-thousand-frame corpus,
        # but the overlap-based filter is correct either way and just as fast.
        self._inverted_index = {}
        for i, tokens in enumerate(self._tokenized_corpus):
            for t in set(tokens):
                self._inverted_index.setdefault(t, set()).add(i)

        n_empty = sum(1 for t in self._tokenized_corpus if not t)
        logger.info(
            f"BM25 index built: {len(self._doc_keys)} documents "
            f"({n_empty} with no extractable text — dense retrieval only for those)"
        )

    # ---------------------------------------------------------------- #
    # Search
    # ---------------------------------------------------------------- #

    def search(self, query: str, top_k: int = 100) -> List[dict]:
        """
        Returns list of dicts: {'video_id', 'frame_idx', 'bm25_score', 'metadata'}
        Sorted by bm25_score descending. Empty list if index is empty or
        the query has no matchable tokens (e.g. pure OOV query — dense
        retrieval carries the query in that case, which is expected).
        """
        if self._bm25 is None or not self._doc_keys:
            return []
        tokens = self.tokenizer(query)
        if not tokens:
            return []

        # Restrict scoring to documents that actually share at least one
        # query token (via the inverted index), rather than filtering by
        # `score > 0` — BM25 score sign isn't a reliable relevance signal
        # on tiny/degenerate corpora (see note in build()).
        candidate_set: set = set()
        for t in tokens:
            candidate_set |= self._inverted_index.get(t, set())
        if not candidate_set:
            return []

        scores = self._bm25.get_scores(tokens)
        candidate_idx = np.fromiter(candidate_set, dtype=np.int64)
        top_idx = candidate_idx[np.argsort(-scores[candidate_idx])][:top_k]

        results = []
        for idx in top_idx:
            meta = self._doc_meta[idx]
            results.append({
                "video_id": meta["video_id"],
                "frame_idx": meta["frame_idx"],
                "bm25_score": float(scores[idx]),
                "metadata": meta["metadata"],
            })
        return results

    # ---------------------------------------------------------------- #
    # Persistence (survives Kaggle session restarts)
    # ---------------------------------------------------------------- #

    def save(self, path: Optional[str] = None) -> None:
        path = path or settings.BM25_INDEX_PATH or str(settings.CACHE_DIR / "bm25_index.pkl")
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "doc_keys": self._doc_keys,
                "doc_meta": self._doc_meta,
                "tokenized_corpus": self._tokenized_corpus,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 index saved to {path} ({len(self._doc_keys)} docs)")

    def load(self, path: Optional[str] = None) -> bool:
        path = path or settings.BM25_INDEX_PATH or str(settings.CACHE_DIR / "bm25_index.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._doc_keys = data["doc_keys"]
        self._doc_meta = data["doc_meta"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        logger.info(f"BM25 index loaded from {path}: {len(self._doc_keys)} documents")
        return True

    # ---------------------------------------------------------------- #
    # Maintenance
    # ---------------------------------------------------------------- #

    def delete_by_video(self, video_id: str) -> None:
        keep = [i for i, m in enumerate(self._doc_meta) if m.get("video_id") != video_id]
        self._doc_keys = [self._doc_keys[i] for i in keep]
        self._doc_meta = [self._doc_meta[i] for i in keep]
        self._tokenized_corpus = [self._tokenized_corpus[i] for i in keep]
        self.build()

    @property
    def size(self) -> int:
        return len(self._doc_keys)


class ElasticBM25Index:
    """
    Drop-in replacement for BM25Index backed by Elasticsearch/OpenSearch,
    for when the in-memory rank_bm25 index becomes too slow to build/query
    (rule of thumb: >200k-300k documents on a single Kaggle CPU).

    Exposes the same .search(query, top_k) -> List[dict] interface so
    HybridFusionPipeline doesn't need to change. NOT implemented here —
    stub left as a clear extension point. Ask if you want this built out;
    it needs a running ES/OpenSearch instance (Docker or a hosted free tier).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ElasticBM25Index is a stub. Use BM25Index (rank_bm25) for now, "
            "or ask for the Elasticsearch implementation if your corpus outgrows it."
        )
