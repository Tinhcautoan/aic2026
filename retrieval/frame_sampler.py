"""
Frame Sampler: select candidate frames in a delta window around keyframes.

Given a keyframe index and the video's total frame count, this module
samples every N-th frame within ±delta of the keyframe.  This feeds
the downstream VLM / embedding alignment with a focused, dense
candidate set instead of uniformly sampling the whole video.

Delta is proportional to frame count (default 5%), so short and long
videos both get a meaningful window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from config import settings


@dataclass
class SampledFrame:
    video_id: str
    frame_idx: int
    score: float
    is_keyframe: bool


def sample_window(
    video_id: str,
    keyframe_idx: int,
    keyframe_score: float,
    frame_count: int,
    delta_ratio: float = 0.05,
    frame_step: int = 5,
) -> List[SampledFrame]:
    """
    Sample every `frame_step`-th frame in [keyframe_idx-delta, keyframe_idx+delta].

    The keyframe itself is always included (even if it doesn't fall on
    a step boundary) and marked `is_keyframe=True`.  Indices are clamped
    to [0, frame_count - 1].
    """
    if frame_count <= 0:
        return []

    delta = int(max(1, frame_count * delta_ratio))

    lo = max(0, keyframe_idx - delta)
    hi = min(frame_count - 1, keyframe_idx + delta)

    frames: List[SampledFrame] = []
    seen = set()

    # Step through the window
    start = lo + (-(lo % frame_step)) % frame_step  # round up to next multiple
    for idx in range(start, hi + 1, frame_step):
        frames.append(SampledFrame(
            video_id=video_id,
            frame_idx=idx,
            score=keyframe_score,
            is_keyframe=(idx == keyframe_idx),
        ))
        seen.add(idx)

    # Always include the keyframe if it wasn't already on a step boundary
    if keyframe_idx not in seen:
        frames.append(SampledFrame(
            video_id=video_id,
            frame_idx=keyframe_idx,
            score=keyframe_score,
            is_keyframe=True,
        ))
        frames.sort(key=lambda f: f.frame_idx)

    return frames


def sample_video_windows(
    video_id: str,
    keyframes: List[Dict],
    frame_count: int,
    delta_ratio: float = 0.05,
    frame_step: int = 5,
) -> List[SampledFrame]:
    """
    Sample frames across multiple keyframes for a single video, merging
    and deduplicating the resulting windows.

    Each keyframe dict must contain at least ``frame_idx`` and ``score``.
    """
    all_frames: List[SampledFrame] = []
    seen: set = set()

    for kf in keyframes:
        window = sample_window(
            video_id=video_id,
            keyframe_idx=kf["frame_idx"],
            keyframe_score=kf.get("score", 0.0),
            frame_count=frame_count,
            delta_ratio=delta_ratio,
            frame_step=frame_step,
        )
        for f in window:
            if f.frame_idx not in seen:
                all_frames.append(f)
                seen.add(f.frame_idx)

    all_frames.sort(key=lambda f: f.frame_idx)
    return all_frames
