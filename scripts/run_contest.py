#!/usr/bin/env python3
"""
Batch contest runner for TKIS / Q&A / TRAKE queries.

Usage:
    python scripts/run_contest.py --queries-dir queries/ --output-dir submission/

Reads all ``*.txt`` files from ``--queries-dir``, detects mode from the
file suffix (``-kis``, ``-qa``, ``-trake``), runs the appropriate
pipeline, and writes one ``<filename>.csv`` per query to ``--output-dir``.

When the real video library is ready, run the full indexing pipeline
(keyframe extraction → embeddings) before using this script.  The search
components (HybridFusionPipeline, embedding_engine) are loaded from the
API layer so the same models are shared.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.query_parser import detect_mode, parse_trake, split_qa, TKIS, QA, TRAKE
from retrieval.answer_expander import expand_rows, format_csv_line, SubmissionRow
from retrieval.frame_sampler import sample_video_windows
from retrieval.trake_aligner import align_video_events, rank_videos
import numpy as np


def _read_query(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _write_csv(path: Path, lines: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ------------------------------------------------------------------ #
# Pipeline stubs — replaced with real implementations when models load
# ------------------------------------------------------------------ #

def _tkis_pipeline(
    query_text: str,
    search_fn,
    embedding_fn,
    max_rows: int = 100,
) -> List[str]:
    """Global search → top-100 (video, frame) → CSV lines."""
    clip_emb, siglip2_emb = embedding_fn(query_text)
    results = search_fn(query_text, clip_emb, siglip2_emb, top_k=max_rows)
    rows = [
        SubmissionRow(
            line=format_csv_line(r["video_id"], r["frame_idx"]),
            score=r.get("final_rerank_score", r.get("rrf_fusion_score", r.get("final_score", 0.0))),
        )
        for r in results
    ]
    ranked = expand_rows(rows, max_rows=max_rows)
    return [r.line for r in ranked]


def _qa_pipeline(
    query_text: str,
    search_fn,
    embedding_fn,
    vlm_fn,
    top_k: int = 20,
    max_rows: int = 100,
) -> List[str]:
    """Parse description/question → search top-K videos → VLM → CSV."""
    from config import settings
    parsed = split_qa(query_text)
    clip_emb, siglip2_emb = embedding_fn(parsed.description)
    results = search_fn(parsed.description, clip_emb, siglip2_emb, top_k=top_k)

    rows: List[SubmissionRow] = []
    for r in results[:top_k]:
        vid = r["video_id"]
        kf_idx = r.get("frame_idx", 0)
        score = r.get("final_rerank_score", r.get("rrf_fusion_score", r.get("final_score", 0.0)))
        frame_count = r.get("frame_count", 2000)
        sampled = sample_video_windows(
            video_id=vid,
            keyframes=[{"frame_idx": kf_idx, "score": score}],
            frame_count=frame_count,
            delta_ratio=settings.DELTA_RATIO,
            frame_step=settings.FRAME_STEP,
        )
        vlm_out = vlm_fn(parsed.question, sampled, vid, mode="qa")
        if vlm_out and vlm_out.get("answer"):
            for sf in sampled:
                rows.append(SubmissionRow(
                    line=format_csv_line(sf.video_id, sf.frame_idx, answer=vlm_out["answer"]),
                    score=sf.score,
                ))

    ranked = expand_rows(rows, max_rows=max_rows)
    return [r.line for r in ranked]


def _trake_pipeline(
    query_text: str,
    search_fn,
    embedding_fn,
    event_search_fn,
    top_k: int = 50,
    max_rows: int = 100,
) -> List[str]:
    """Parse events → search top-K videos → align → rank → CSV."""
    from config import settings
    parsed = parse_trake(query_text)
    clip_emb, siglip2_emb = embedding_fn(parsed.main)
    results = search_fn(parsed.main, clip_emb, siglip2_emb, top_k=top_k)

    alignments = []
    for r in results[:top_k]:
        vid = r["video_id"]
        kf_idx = r.get("frame_idx", 0)
        score = r.get("final_rerank_score", r.get("rrf_fusion_score", r.get("final_score", 0.0)))
        frame_count = r.get("frame_count", 2000)
        sampled = sample_video_windows(
            video_id=vid,
            keyframes=[{"frame_idx": kf_idx, "score": score}],
            frame_count=frame_count,
            delta_ratio=settings.DELTA_RATIO,
            frame_step=settings.FRAME_STEP,
        )
        candidate_dicts = [{"video_id": vid, "frame_idx": sf.frame_idx, "score": sf.score} for sf in sampled]
        alignment = align_video_events(
            video_id=vid,
            candidate_frames=candidate_dicts,
            event_texts=parsed.events,
            query_score=score,
            event_search_fn=event_search_fn,
        )
        alignments.append(alignment)

    ranked = rank_videos(alignments, max_rows=max_rows)
    rows = []
    for a in ranked:
        frames = [ef.frame_idx for ef in a.event_frames if ef.frame_idx >= 0]
        if len(frames) == len(parsed.events):
            rows.append(SubmissionRow(
                line=format_csv_line(a.video_id, frames=frames),
                score=a.total_score,
            ))

    ranked_rows = expand_rows(rows, max_rows=max_rows)
    return [r.line for r in ranked_rows]


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def _make_embedding_fn(engine):
    """Wrap engine.encode_text to return tuple (clip_emb, siglip2_emb)."""
    def embedding_fn(text):
        embs = engine.encode_text(text)
        return embs["clip"], embs["siglip2"]
    return embedding_fn


def _make_event_search_fn(store, engine):
    """Create event_search_fn for TRAKE alignment.

    Returns: fn(event_text, candidates) -> [(frame_idx, score), ...]
    """
    def event_search_fn(event_text, candidates):
        embs = engine.encode_text(event_text)
        clip_emb = embs["clip"]
        siglip2_emb = embs["siglip2"]

        results = []
        for cand in candidates:
            fi = cand["frame_idx"]
            vid = cand.get("video_id", candidates[0].get("video_id", ""))
            frame_embs = store.get_frame_embeddings(video_id=vid, frame_indices=[fi])
            if not frame_embs:
                continue
            fe = frame_embs[0]
            clip_sim = float(np.dot(clip_emb, fe["clip_emb"]))
            siglip2_sim = float(np.dot(siglip2_emb, fe["siglip2_emb"]))
            avg_sim = (clip_sim + siglip2_sim) / 2.0
            results.append((fi, avg_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    return event_search_fn


def _load_search_backend(engine):
    """Prefer Milvus + the full retrieval funnel (BM25 + cross-encoder) so the
    contest media-info metadata (title/keywords/date) engages; fall back to
    PersistentVectorStore when Milvus is unavailable.

    Returns (store, search_fn) where ``search_fn(query_text, clip_emb,
    siglip2_emb, top_k)`` returns candidate dicts best-first.
    """
    from config import settings
    from retrieval.reranker import HybridReranker
    from retrieval.bm25_search import BM25Index, build_bm25_index_from_store
    from retrieval.hybrid_fusion import HybridFusionPipeline
    from retrieval import media_meta

    try:
        from retrieval.milvus_store import MilvusVectorStore

        store = MilvusVectorStore()

        bm25 = BM25Index()
        if not (bm25.load() and bm25.size > 0):
            build_bm25_index_from_store(store, bm25)
            if bm25.size > 0:
                bm25.save()

        cross_encoder = None
        if settings.CROSS_ENCODER_ENABLED:
            try:
                from retrieval.cross_encoder_reranker import CrossEncoderReranker
                cross_encoder = CrossEncoderReranker()
            except Exception as e:
                print(f"  [warn] cross-encoder unavailable: {e}")

        pipeline = HybridFusionPipeline(
            vector_store=store,
            bm25_index=bm25,
            dense_reranker=HybridReranker(),
            cross_encoder=cross_encoder,
        )

        def search_fn(query_text, clip_emb, siglip2_emb, top_k=100):
            results = pipeline.search(
                query=query_text,
                clip_query_emb=clip_emb,
                siglip2_query_emb=siglip2_emb,
                final_top_k=top_k,
                use_bm25=True,
                use_cross_encoder=True,
                expand_neighbors=False,
                dedup=False,
            )
            # Enrich each hit with the real video frame count (from media-info
            # length) so Q&A/TRAKE sample realistic delta windows.
            for r in results:
                if "frame_count" not in r:
                    fc = media_meta.video_frame_count(r.get("video_id", ""))
                    if fc:
                        r["frame_count"] = fc
            return results

        print(f"  Milvus + full funnel ready (BM25={bm25.size} docs, "
              f"cross_encoder={'yes' if cross_encoder else 'no'})")
        return store, search_fn

    except Exception as e:
        print(f"  [warn] Milvus/funnel unavailable ({e}); "
              f"falling back to PersistentVectorStore")
        from retrieval.persistent_store import PersistentVectorStore
        store = PersistentVectorStore(str(settings.DATA_DIR / "persistent_store"))
        print(f"Loaded: {len(store._ids)} frames in persistent store")
        return store, store.hybrid_search


def main():
    parser = argparse.ArgumentParser(description="Run contest queries (TKIS/Q&A/TRAKE)")
    parser.add_argument("--queries-dir", required=True, help="Directory with query .txt files")
    parser.add_argument("--output-dir", default="submission", help="Output CSV directory")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K videos for Q&A")
    parser.add_argument("--trake-videos", type=int, default=50, help="Top-K videos for TRAKE")
    parser.add_argument("--max-rows", type=int, default=100, help="Max CSV rows per query")
    args = parser.parse_args()

    queries_dir = Path(args.queries_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load models ---
    print("Loading models...", flush=True)

    from embedding.hybrid_engine import HybridEmbeddingEngine
    from retrieval.vqa_llm import answer_from_vlm

    engine = HybridEmbeddingEngine()
    store, search_fn = _load_search_backend(engine)
    embedding_fn = _make_embedding_fn(engine)
    vlm_fn = answer_from_vlm
    event_search_fn = _make_event_search_fn(store, engine)

    txt_files = sorted(queries_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {queries_dir}")
        return

    print(f"Found {len(txt_files)} query files.")
    for txt_path in txt_files:
        mode = detect_mode(txt_path.name)
        query_text = _read_query(txt_path)
        csv_path = output_dir / f"{txt_path.stem}.csv"

        print(f"  [{mode.upper():5s}] {txt_path.name} → {csv_path.name}")

        if mode == TKIS:
            lines = _tkis_pipeline(query_text, search_fn, embedding_fn, args.max_rows)
        elif mode == QA:
            lines = _qa_pipeline(query_text, search_fn, embedding_fn, vlm_fn, args.top_k, args.max_rows)
        elif mode == TRAKE:
            lines = _trake_pipeline(query_text, search_fn, embedding_fn, event_search_fn, args.trake_videos, args.max_rows)
        else:
            lines = []

        _write_csv(csv_path, lines)
        print(f"    → {len(lines)} rows")

    print(f"\nDone. Output in {output_dir}/")


if __name__ == "__main__":
    main()
