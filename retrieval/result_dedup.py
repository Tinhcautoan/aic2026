"""
Stage 6: deduplicate search results for UI diversity.

Applies temporal NMS (same video, nearby frames) and optional scene bucketing
(same video, nearby timestamps) while preserving score order. Backfills from
the original pool when strict dedup yields fewer than top_k results.
"""

from __future__ import annotations

from typing import List, Optional

from config import settings


def _result_score(c: dict) -> float:
    return c.get("final_rerank_score", c.get("final_score", c.get("clip_score", 0.0)))


def _scene_bucket(video_id: str, timestamp: float, window_sec: float) -> tuple:
    if window_sec <= 0:
        return (video_id, int(timestamp))
    return (video_id, int(timestamp // window_sec))


def _passes_temporal_nms(
    candidate: dict,
    selected: List[dict],
    min_frame_gap: int,
) -> bool:
    """Return False if candidate is too close in frame index to an already-selected hit."""
    vid = candidate["video_id"]
    frame_idx = candidate["frame_idx"]
    for s in selected:
        if s["video_id"] != vid:
            continue
        if abs(s["frame_idx"] - frame_idx) < min_frame_gap:
            return False
    return True


def _passes_scene_bucket(
    candidate: dict,
    scene_buckets: set,
    scene_window_sec: float,
) -> bool:
    """Return False if this video/scene bucket was already used."""
    if scene_window_sec <= 0:
        return True
    bucket = _scene_bucket(candidate["video_id"], candidate.get("timestamp", 0.0), scene_window_sec)
    return bucket not in scene_buckets


def deduplicate_results(
    results: List[dict],
    top_k: int,
    min_frame_gap: Optional[int] = None,
    scene_window_sec: Optional[float] = None,
    use_scene_bucket: bool = True,
) -> List[dict]:
    """
    Remove near-duplicate frames while keeping the highest-scoring representative.

    Args:
        results: Ranked candidates (best first).
        top_k: Target number of results to return.
        min_frame_gap: Minimum frame distance within the same video.
        scene_window_sec: Max one result per (video, timestamp bucket).
        use_scene_bucket: Apply scene bucketing when True.

    Returns:
        Deduplicated list, length <= top_k (backfilled from pool if needed).
    """
    if not results or top_k <= 0:
        return []

    min_frame_gap = (
        settings.RESULT_DEDUP_MIN_FRAME_GAP if min_frame_gap is None else min_frame_gap
    )
    scene_window_sec = (
        settings.RESULT_DEDUP_SCENE_WINDOW_SEC if scene_window_sec is None else scene_window_sec
    )

    selected: List[dict] = []
    scene_buckets: set = set()
    selected_keys: set = set()

    def _try_add(candidate: dict, enforce_dedup: bool) -> bool:
        key = f"{candidate['video_id']}_{candidate['frame_idx']}"
        if key in selected_keys:
            return False
        if enforce_dedup:
            if not _passes_temporal_nms(candidate, selected, min_frame_gap):
                return False
            if use_scene_bucket and not _passes_scene_bucket(candidate, scene_buckets, scene_window_sec):
                return False
        selected.append(candidate)
        selected_keys.add(key)
        if use_scene_bucket and scene_window_sec > 0:
            scene_buckets.add(
                _scene_bucket(candidate["video_id"], candidate.get("timestamp", 0.0), scene_window_sec)
            )
        return True

    # Pass 1: strict dedup in score order
    for c in results:
        if len(selected) >= top_k:
            break
        _try_add(c, enforce_dedup=True)

    # Pass 2: backfill without dedup rules so UI is not short on results
    if len(selected) < top_k:
        for c in results:
            if len(selected) >= top_k:
                break
            _try_add(c, enforce_dedup=False)

    return selected[:top_k]
