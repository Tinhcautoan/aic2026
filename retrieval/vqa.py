"""
Video Question Answering helpers (retrieval + metadata based).

Uses in-video frame retrieval and per-frame text metadata (OCR/ASR/objects)
to produce a short text answer when no dedicated VQA generative model is loaded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from retrieval.bm25_search import build_frame_document_text


def normalize_video_id(video: str) -> str:
    """Map 'foo.mp4' or path/to/foo.mp4 -> 'foo'."""
    return Path((video or "").strip()).stem


def _score_text_overlap(question: str, text: str) -> float:
    q_tokens = set(re.findall(r"[\w\u00C0-\u1EF9]+", question.lower()))
    t_tokens = set(re.findall(r"[\w\u00C0-\u1EF9]+", text.lower()))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _pick_answer_snippet(question: str, text: str, max_len: int = 256) -> str:
    if not text:
        return ""
    chunks = [c.strip() for c in re.split(r"[|\n;]+", text) if c.strip()]
    if not chunks:
        chunks = [text.strip()]
    chunks.sort(key=lambda c: _score_text_overlap(question, c), reverse=True)
    answer = chunks[0]
    if len(answer) > max_len:
        answer = answer[: max_len - 3].rstrip() + "..."
    return answer


def answer_from_retrieval_results(
    question: str,
    results: List[dict],
    cross_encoder=None,
) -> Dict[str, Any]:
    """
    Build a VQA answer from ranked retrieval hits inside one video.

    Returns dict with answer, evidence frame, timestamp, and debug info.
    """
    if not results:
        return {
            "answer": "",
            "evidence_video_id": "",
            "evidence_frame_idx": None,
            "evidence_timestamp": None,
            "confidence": 0.0,
            "source": "none",
        }

    candidates = [dict(r) for r in results if build_frame_document_text(r.get("metadata") or {})]
    if not candidates:
        return {
            "answer": "",
            "evidence_video_id": "",
            "evidence_frame_idx": None,
            "evidence_timestamp": None,
            "confidence": 0.0,
            "source": "none",
        }
    ranked = candidates

    if cross_encoder is not None and candidates and cross_encoder.has_any_text(candidates):
        try:
            ranked = cross_encoder.rerank(question, candidates, top_k=min(5, len(candidates)))
        except Exception:
            ranked = candidates

    best = ranked[0]
    meta = best.get("metadata") or {}
    doc_text = build_frame_document_text(meta)
    answer = _pick_answer_snippet(question, doc_text)

    if not answer:
        # Fallback: use retrieval score only — no textual evidence indexed yet.
        answer = ""

    score = best.get("cross_encoder_score", best.get("final_rerank_score", best.get("final_score", 0.0)))
    return {
        "answer": answer,
        "evidence_video_id": best.get("video_id", ""),
        "evidence_frame_idx": best.get("frame_idx"),
        "evidence_timestamp": best.get("timestamp"),
        "confidence": float(score) if score is not None else 0.0,
        "source": "metadata" if answer else "retrieval_only",
    }
