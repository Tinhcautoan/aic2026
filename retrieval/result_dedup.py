"""
Stage 6: deduplicate results.

FIX
---
Stage 5 creates synthetic neighbours at +-FRAME_EXPANSION_WINDOWS (default
+-5 frames) to hedge against the annotated frame being a few frames off from
the keyframe we indexed. Stage 6 then removed every frame within
RESULT_DEDUP_MIN_FRAME_GAP (default 15) of an already-selected hit. Since
the source hit always scores higher and is selected first, every neighbour
failed temporal NMS and was dropped. The 8-second scene bucket removed the
rest. Stage 5 was, in practice, a no-op.

Neighbours are now exempt from both NMS and scene bucketing: they exist to
cover frame indices, not to add visual variety, so the diversity rules that
apply to real hits do not apply to them. They also never claim a scene
bucket, which would otherwise let a synthetic frame block a genuine hit
later in the list.
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


def _passes_temporal_nms(candidate: dict, selected: List[dict], min_frame_gap: int) -> bool:
    vid = candidate["video_id"]
    frame_idx = candidate["frame_idx"]
    for s in selected:
        if s["video_id"] != vid:
            continue
        if abs(s["frame_idx"] - frame_idx) < min_frame_gap:
            return False
    return True


def _passes_scene_bucket(candidate: dict, scene_buckets: set, scene_window_sec: float) -> bool:
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
    exempt_neighbors: Optional[bool] = None,
) -> List[dict]:
    """Drop near-duplicate frames, keeping the highest-scoring representative.

    Args:
        exempt_neighbors: when True, candidates flagged `is_expanded_neighbor`
            bypass NMS and scene bucketing. Defaults to
            settings.RESULT_DEDUP_EXEMPT_NEIGHBORS.
    """
    if not results or top_k <= 0:
        return []

    min_frame_gap = settings.RESULT_DEDUP_MIN_FRAME_GAP if min_frame_gap is None else min_frame_gap
    scene_window_sec = (
        settings.RESULT_DEDUP_SCENE_WINDOW_SEC if scene_window_sec is None else scene_window_sec
    )
    if exempt_neighbors is None:
        exempt_neighbors = bool(getattr(settings, "RESULT_DEDUP_EXEMPT_NEIGHBORS", True))

    selected: List[dict] = []
    scene_buckets: set = set()
    selected_keys: set = set()

    def _is_neighbor(c: dict) -> bool:
        return exempt_neighbors and bool(c.get("is_expanded_neighbor"))

    def _try_add(candidate: dict, enforce_dedup: bool) -> bool:
        key = f"{candidate['video_id']}_{candidate['frame_idx']}"
        if key in selected_keys:
            return False

        neighbor = _is_neighbor(candidate)
        if enforce_dedup and not neighbor:
            if not _passes_temporal_nms(candidate, selected, min_frame_gap):
                return False
            if use_scene_bucket and not _passes_scene_bucket(candidate, scene_buckets, scene_window_sec):
                return False

        selected.append(candidate)
        selected_keys.add(key)

        # A synthetic neighbour must not claim a scene bucket; doing so would
        # let it suppress a genuine hit that appears later in the ranking.
        if not neighbor and use_scene_bucket and scene_window_sec > 0:
            scene_buckets.add(
                _scene_bucket(candidate["video_id"], candidate.get("timestamp", 0.0), scene_window_sec)
            )
        return True

    for c in results:
        if len(selected) >= top_k:
            break
        _try_add(c, enforce_dedup=True)

    if len(selected) < top_k:
        for c in results:
            if len(selected) >= top_k:
                break
            _try_add(c, enforce_dedup=False)

    return selected[:top_k]
