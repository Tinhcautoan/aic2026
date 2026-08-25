"""
TRAKE alignment: pick one frame per event, in strictly increasing temporal
order, maximising total similarity.

WHY THIS WAS REWRITTEN
----------------------
The previous implementation took the argmax candidate for each event
independently, excluding frames already used. Nothing forced frame indices
to increase. A video whose event 3 fires before event 1 scored identically
to a correctly ordered one -- but the AIC submission format requires
`<video>,<Frame ID_1>,...,<Frame ID_N>` in true temporal order, so an
out-of-order row scores zero.

This is now a dynamic program (the DANTE formulation):

    dp[t][j] = s_t(F[j]) + max{ dp[t-1][k] - lambda*(F[j]-F[k]) : F[k] < F[j] }

solved with a running prefix maximum, so it is O(N*M) rather than O(N*M^2).
With lambda != 0 the prefix maximum tracks (dp + lambda*F) and subtracts
lambda*F[j] at use time, which keeps the linear-time property.

Strictly increasing indices also make the old `used_frames` bookkeeping
unnecessary -- a frame cannot be reused if every step must move forward.

Preserved from the old behaviour, because callers and tests depend on it:
  * `total_score = query_score * mean(event_scores)`
  * an event with no candidates keeps `frame_idx = -1`, `score = 0.0`
  * `rank_videos` is unchanged
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from loguru import logger

try:
    from config import settings
except Exception:  # pragma: no cover
    settings = None


def _penalty() -> float:
    return float(getattr(settings, "TRAKE_ORDER_PENALTY", 0.0)) if settings else 0.0


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
    degraded: bool = False


def _solve_monotone_chain(
    event_scores: List[Dict[int, float]],
    penalty: float = 0.0,
) -> Optional[List[Tuple[int, float]]]:
    """Best strictly increasing frame assignment, one frame per event.

    `event_scores[t]` maps frame_idx -> similarity for event t. Every event
    passed in must have at least one candidate; empty events are handled by
    the caller, which removes them from the chain so they do not block the
    ordering of their neighbours.

    Returns [(frame_idx, score), ...] aligned with the input, or None when no
    strictly increasing assignment exists at all.
    """
    n = len(event_scores)
    if n == 0:
        return []

    frames: List[int] = sorted({f for sc in event_scores for f in sc})
    m = len(frames)
    if m == 0:
        return None
    pos = {f: j for j, f in enumerate(frames)}

    NEG = -math.inf
    dp = [[NEG] * m for _ in range(n)]
    back = [[-1] * m for _ in range(n)]

    for f, s in event_scores[0].items():
        dp[0][pos[f]] = float(s)

    for t in range(1, n):
        best_val = NEG          # max over k < j of (dp[t-1][k] + penalty*F[k])
        best_idx = -1
        for j in range(m):
            # Everything with index < j is already folded into the prefix max.
            if best_idx >= 0:
                prev = best_val - penalty * frames[j]
                s = event_scores[t].get(frames[j])
                if s is not None and prev > NEG:
                    dp[t][j] = float(s) + prev
                    back[t][j] = best_idx
            # Now admit j itself into the prefix for the next iteration.
            if dp[t - 1][j] > NEG:
                cand = dp[t - 1][j] + penalty * frames[j]
                if cand > best_val:
                    best_val = cand
                    best_idx = j

    end = max(range(m), key=lambda j: dp[n - 1][j])
    if dp[n - 1][end] == NEG:
        return None

    chain: List[Tuple[int, float]] = []
    j = end
    for t in range(n - 1, -1, -1):
        f = frames[j]
        chain.append((f, float(event_scores[t][f])))
        j = back[t][j]
        if j < 0 and t > 0:
            return None
    chain.reverse()
    return chain


def _greedy_fallback(
    event_texts: Sequence[str],
    scored: List[List[Tuple[int, float]]],
) -> List[TrakeEventFrame]:
    """Original behaviour, used only when no ordered chain exists."""
    out: List[TrakeEventFrame] = []
    used: set = set()
    for text, results in zip(event_texts, scored):
        pick = next(((fi, sc) for fi, sc in results if fi not in used), None)
        if pick is None:
            out.append(TrakeEventFrame(text, -1, 0.0))
        else:
            used.add(pick[0])
            out.append(TrakeEventFrame(text, pick[0], pick[1]))
    return out


def align_video_events(
    video_id: str,
    candidate_frames: List[dict],
    event_texts: List[str],
    query_score: float,
    event_search_fn: Callable[[str, List[dict]], List[Tuple[int, float]]],
    order_penalty: Optional[float] = None,
) -> TrakeVideoAlignment:
    """Align events to frames under a strict temporal-order constraint."""
    penalty = _penalty() if order_penalty is None else float(order_penalty)

    scored: List[List[Tuple[int, float]]] = [
        list(event_search_fn(text, candidate_frames) or []) for text in event_texts
    ]

    # Events with no candidates are gaps: they take frame -1 and are lifted
    # out of the chain so they cannot break their neighbours' ordering.
    chain_positions = [i for i, r in enumerate(scored) if r]
    degraded = False

    if not chain_positions:
        event_frames = [TrakeEventFrame(t, -1, 0.0) for t in event_texts]
    else:
        maps: List[Dict[int, float]] = []
        for i in chain_positions:
            best: Dict[int, float] = {}
            for fi, sc in scored[i]:
                if fi not in best or sc > best[fi]:
                    best[fi] = float(sc)
            maps.append(best)

        solution = _solve_monotone_chain(maps, penalty=penalty)

        if solution is None:
            degraded = True
            logger.warning(
                f"TRAKE: no strictly increasing assignment for video {video_id} "
                f"({len(chain_positions)} events with candidates); falling back to greedy."
            )
            event_frames = _greedy_fallback(event_texts, scored)
        else:
            event_frames = [TrakeEventFrame(t, -1, 0.0) for t in event_texts]
            for slot, (frame_idx, score) in zip(chain_positions, solution):
                event_frames[slot] = TrakeEventFrame(event_texts[slot], frame_idx, score)

    scores = [ef.score for ef in event_frames]
    avg = sum(scores) / max(1, len(scores))
    return TrakeVideoAlignment(
        video_id=video_id,
        event_frames=event_frames,
        total_score=query_score * avg,
        degraded=degraded,
    )


def rank_videos(
    alignments: List[TrakeVideoAlignment],
    max_rows: int = 100,
) -> List[TrakeVideoAlignment]:
    return sorted(alignments, key=lambda a: a.total_score, reverse=True)[:max_rows]


def violates_order(alignment: TrakeVideoAlignment) -> bool:
    """True when the produced frames are not strictly increasing.

    Cheap invariant check -- needs no ground truth. Should be False for every
    alignment the DP produced; only a degraded fallback can trip it.
    """
    idx = [ef.frame_idx for ef in alignment.event_frames if ef.frame_idx >= 0]
    return any(b <= a for a, b in zip(idx, idx[1:]))
