"""
TRAKE on the GPU index: rank videos by their sub-events, then align one frame per
event in strictly increasing time order.

This is the AIC-HQT method with the fusion layer removed (PE-Core only):

  A. Sub-event ranking  -- score every event separately, take each video's best
     frame per event, average over events ("mean of max"). HQT measured this at
     R@1 31.7 vs 25.0 for one holistic sentence on Charades-STA. Because every
     frame sits in the GPU matrix the scores are exact for ALL videos; the
     "pool floor" imputation HQT needed for a truncated vector-DB pool is gone.
  B. Whole-video candidates -- the alignment sees every indexed frame of the
     video (HQT's TRAKE_DELTA_RATIO = 1.0), not a window around one anchor.
  C. Monotone DP -- dp[t][j] = S[j,t] + max{dp[t-1][k] : frame_k <= frame_j - gap}
     solved with a running prefix maximum, O(E*M). An out-of-order row would
     score zero at submission, so the chain is strictly increasing by construction.

The final video order is the mean cosine of the aligned chain. (HQT multiplies it by
sigmoid(mean logit); with PE-Core, which has no logit_bias, that factor saturates to
~1.0 for every realistic candidate, so it does not change the order.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from retrieval.frame_index import FrameIndex

__all__ = ["align_chain", "rank_trake", "TrakeVideo"]

_NEG = -1e30


def align_chain(S: np.ndarray, frames: np.ndarray, min_gap: int = 1) -> Optional[Tuple[List[int], float]]:
    """Best strictly-increasing assignment of one row per event.

    S       [M, E] similarity of each of the M candidate frames to each of the E events.
    frames  [M] ascending frame indices of those rows.
    min_gap minimum distance, in frame indices, between consecutive events (1 = strictly
            increasing; larger stops the chain from collapsing onto one instant).

    Returns ([row per event], total score) or None when no ordered chain exists.
    """
    M, E = S.shape
    if E == 0:
        return [], 0.0
    if M < E:
        return None
    gap = max(1, int(min_gap))
    # lim[j] = number of rows k with frames[k] <= frames[j] - gap  (all such k have k < j)
    lim = np.searchsorted(frames, frames - gap, side="right")
    dp = np.full((E, M), _NEG, dtype=np.float64)
    back = np.full((E, M), -1, dtype=np.int64)
    dp[0] = S[:, 0]
    ar = np.arange(M)
    for t in range(1, E):
        prev = dp[t - 1]
        cm = np.maximum.accumulate(prev)                                  # best dp[t-1][k] for k <= j
        am = np.maximum.accumulate(np.where(prev >= cm, ar, 0))           # ... and where it was attained
        ok = lim > 0
        jm = np.clip(lim - 1, 0, M - 1)
        best = cm[jm]
        feasible = ok & (best > _NEG / 2)
        dp[t] = np.where(feasible, S[:, t] + best, _NEG)
        back[t] = np.where(feasible, am[jm], -1)
    end = int(np.argmax(dp[E - 1]))
    if dp[E - 1, end] <= _NEG / 2:
        return None
    rows = [end]
    for t in range(E - 1, 0, -1):
        rows.append(int(back[t, rows[-1]]))
    rows.reverse()
    return rows, float(dp[E - 1, end])


def _alternatives(col: np.ndarray, ms: np.ndarray, n: int, min_ms: int, keep_row: int) -> List[int]:
    """Up to n rows with the highest score for one event, at least `min_ms` apart.
    The chain's own pick always comes first."""
    order = np.argsort(-col, kind="stable")
    picked = [keep_row]
    for r in order:
        if len(picked) >= n:
            break
        r = int(r)
        if r == keep_row:
            continue
        if all(abs(int(ms[r]) - int(ms[p])) >= min_ms for p in picked):
            picked.append(r)
    return picked


@dataclass
class TrakeVideo:
    video_id: str
    fps: float
    score: float                 # mean cosine of the aligned chain
    stage_a: float               # mean over events of the video's best cosine (unordered)
    frames: List[dict] = field(default_factory=list)   # per event: frame_idx, ms, score, alts[]


@torch.inference_mode()
def rank_trake(index: FrameIndex, Q: torch.Tensor, events: Sequence[str], top_k: int = 20,
               shortlist: int = 30, alt_n: int = 8, alt_min_sec: float = 1.5,
               min_gap_sec: float = 0.0) -> List[TrakeVideo]:
    """Q: [E, D] normalised event embeddings (any dtype). Returns up to top_k videos, best first."""
    E = int(Q.shape[0])
    if E == 0:
        return []
    Qd = Q.to(index.device, dtype=index.emb.dtype)
    S = index.emb @ Qd.T                                       # [N, E] cosines, every frame x every event

    # ---- A. per-video best frame per event, averaged over events -------------------------
    V = len(index.video_ids)
    best = torch.full((V, E), float("-inf"), device=index.device, dtype=S.dtype)
    best.scatter_reduce_(0, index.vid_t[:, None].expand(-1, E), S, reduce="amax", include_self=True)
    stage_a = best.float().mean(dim=1)
    too_short = torch.from_numpy(index.frame_counts < E).to(index.device)     # cannot hold E ordered frames
    stage_a = torch.where(too_short, torch.full_like(stage_a, float("-inf")), stage_a)
    k = min(max(shortlist, top_k), V)
    top_vals, top_v = torch.topk(stage_a, k)
    top_vals, top_v = top_vals.cpu().numpy(), top_v.cpu().numpy()
    S_cpu = S.float().cpu().numpy()                            # [N, E]; small (N*E floats)

    # ---- B + C. align inside each shortlisted video -------------------------------------
    out: List[TrakeVideo] = []
    for v, a in zip(top_v, top_vals):
        if not np.isfinite(a):
            continue
        s, e = index.slice_of(int(v))
        fr, ms = index.frames[s:e], index.ms[s:e]
        Sv = S_cpu[s:e]
        fps = float(index.fps[int(v)])
        gap = max(1, int(round(min_gap_sec * fps))) if min_gap_sec > 0 else 1
        sol = align_chain(Sv, fr, gap)
        if sol is None and gap > 1:
            sol = align_chain(Sv, fr, 1)                       # a wide gap can be infeasible on a short clip
        if sol is None:
            continue
        rows, total = sol
        frames_out = []
        for t, r in enumerate(rows):
            alts = _alternatives(Sv[:, t], ms, alt_n, int(alt_min_sec * 1000), r)
            frames_out.append({
                "frame_idx": int(fr[r]), "ms": int(ms[r]), "score": float(Sv[r, t]),
                "alts": [{"frame_idx": int(fr[x]), "ms": int(ms[x]), "score": float(Sv[x, t])} for x in alts],
            })
        out.append(TrakeVideo(index.video_ids[int(v)], fps, total / E, float(a), frames_out))
    out.sort(key=lambda x: (x.score, x.stage_a), reverse=True)
    return out[:top_k]
