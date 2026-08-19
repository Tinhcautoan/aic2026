"""
TRAKE alignment: for each event in the query, find the best-matching
frame within the pre-sampled candidate pool for each video, then rank
videos by combined score.

This module does NOT embed anything itself — it receives an
`event_search_fn` (dependency-injected) that maps event text to
scored candidate frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple


@dataclass
class TrakeEventFrame:
    event_text: str
    frame_idx: int
    score: float


@dataclass
class TrakeVideoAlignment:
    video_id: str
    event_frames: List[TrakeEventFrame]
    total_score: float


def align_video_events(
    video_id: str,
    candidate_frames: List[dict],
    event_texts: List[str],
    query_score: float,
    event_search_fn: Callable[[str, List[dict]], List[Tuple[int, float]]],
) -> TrakeVideoAlignment:
    """
    Find the best candidate frame for each event using event_search_fn.

    Args:
        candidate_frames: sampled frames from the delta window (each with
            at least ``frame_idx`` and ``score``).
        event_texts: ordered event descriptions extracted by query_parser.
        query_score: the main-query retrieval score (used in total_score).
        event_search_fn: ``fn(event_text, candidates) -> [(frame_idx, score), ...]``
            Must return results sorted best-first.  Typically backed by
            TULIP / SigLIP2 cosine similarity against the candidate embeddings.

    total_score is computed as ``query_score * mean(event_scores)`` so that
    both query relevance and per-event alignment contribute.
    """
    event_frames: List[TrakeEventFrame] = []
    event_scores: List[float] = []
    used_frames: set = set()

    for event_text in event_texts:
        results = event_search_fn(event_text, candidate_frames)
        results = [(fi, sc) for fi, sc in results if fi not in used_frames]
        if results:
            best_idx, best_score = results[0]
            used_frames.add(best_idx)
            event_frames.append(TrakeEventFrame(event_text, best_idx, best_score))
            event_scores.append(best_score)
        else:
            event_frames.append(TrakeEventFrame(event_text, -1, 0.0))
            event_scores.append(0.0)

    avg_event_score = sum(event_scores) / max(1, len(event_scores))
    total_score = query_score * avg_event_score

    return TrakeVideoAlignment(
        video_id=video_id,
        event_frames=event_frames,
        total_score=total_score,
    )


def rank_videos(
    alignments: List[TrakeVideoAlignment],
    max_rows: int = 100,
) -> List[TrakeVideoAlignment]:
    """Sort video alignments best-first and cap at max_rows."""
    ranked = sorted(alignments, key=lambda a: a.total_score, reverse=True)
    return ranked[:max_rows]
