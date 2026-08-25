"""
Unified scoring and confidence layer.

WHY THIS MODULE EXISTS
----------------------
Before this module, `final_rerank_score` carried four mutually incompatible
scales depending on which branch of the funnel a candidate came out of:

    cross-encoder ran        -> sigmoid(logit)            in (0, 1)
    cross-encoder skipped    -> raw RRF sum               <= 2/(k+1) = 0.0328
    beyond fusion_top_k      -> rrf_fusion_score - 1e6    ~ -999999.97
    no BM25 hit at all       -> dense reranker score      in [0, 1]

The UI read that single field as "confidence", so whenever the cross-encoder
was skipped (the common case, because it needs OCR/ASR text that most frames
do not have) the reported confidence collapsed to <= 0.033. That is the
"confidence is always tiny" symptom. It was never a model quality problem.

THE FIX: two numbers, not one.
------------------------------
  final_score : ordering key only. Banded by tier so a cross-encoded
                candidate always outranks a merely-fused one, but kept
                inside [0, 1] so nothing ever displays -1e6.

  confidence  : a calibrated, cross-query-comparable number in [0, 1]
                derived from how far the candidate separates from the rest
                of the retrieved pool. Never used for ordering.

CALIBRATION
-----------
Raw cosine from a CLIP-family model is not a probability. CLIP ViT-L/14
puts correct matches around 0.20-0.35 cosine; that looks "low" but is
normal. Every open_clip model ships the temperature it was trained with as
`logit_scale` (log-space), and SigLIP models additionally ship `logit_bias`.
`calibrated_similarity` applies them so downstream code works with the
model's own logit space instead of raw cosine.

HONEST LIMITATION
-----------------
`confidence` is a *relative* score: monotone in evidence, comparable across
queries, and bounded. It is NOT P(top-1 is correct). Turning it into a true
probability needs a few hundred labelled queries plus an isotonic or Platt
fit -- see `fit_isotonic` at the bottom, which is the hook for that.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from config import settings
except Exception:  # pragma: no cover - allows standalone import in tests
    settings = None


_EPS = 1e-8

# ---------------------------------------------------------------------------
# Tiers. Lower is better. A candidate never crosses tiers because of its
# score; tiers encode *how much evidence we gathered*, not how good it looks.
# ---------------------------------------------------------------------------
TIER_CROSS_ENCODED = 0   # survived to stage 4 and was jointly scored
TIER_FUSED = 1           # dense (+ sparse) fusion only
TIER_EXPANDED = 2        # synthetic frame-neighbour, no evidence of its own

_TIER_BAND = {
    TIER_CROSS_ENCODED: (0.60, 1.00),
    TIER_FUSED:         (0.20, 0.60),
    TIER_EXPANDED:      (0.00, 0.20),
}


def _cfg(name: str, default):
    return getattr(settings, name, default) if settings is not None else default


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    # split form avoids overflow for large |x|
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def minmax(scores: Sequence[float]) -> np.ndarray:
    """Scale to [0, 1]. A degenerate pool maps to 0.5, never to 0 -- mapping a
    flat pool to zero would silently annihilate one branch of a fusion."""
    a = np.asarray(scores, dtype=np.float64)
    if a.size == 0:
        return a
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < _EPS:
        return np.full_like(a, 0.5)
    return (a - lo) / (hi - lo)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    a = np.asarray(scores, dtype=np.float64) / max(temperature, _EPS)
    a = a - a.max()
    e = np.exp(a)
    return e / (e.sum() + _EPS)


def zscore(scores: Sequence[float]) -> np.ndarray:
    a = np.asarray(scores, dtype=np.float64)
    if a.size == 0:
        return a
    sd = float(a.std())
    if sd < _EPS:
        return np.zeros_like(a)
    return (a - float(a.mean())) / sd


# ---------------------------------------------------------------------------
# Model-native calibration
# ---------------------------------------------------------------------------

def get_logit_params(model) -> tuple:
    """Pull (logit_scale, logit_bias) out of an open_clip model.

    Returns floats. `logit_scale` is stored in log space by open_clip, so it
    is exponentiated here. SigLIP-family models also carry `logit_bias`;
    CLIP-family models do not, in which case bias is 0.0.

    Falls back to (100.0, 0.0) -- CLIP's trained temperature -- if the model
    exposes neither, so callers never have to branch.
    """
    scale, bias = None, 0.0
    try:
        import torch
        ls = getattr(model, "logit_scale", None)
        if ls is not None:
            v = ls.detach() if hasattr(ls, "detach") else ls
            scale = float(torch.exp(v).item() if hasattr(v, "item") else np.exp(float(v)))
        lb = getattr(model, "logit_bias", None)
        if lb is not None:
            v = lb.detach() if hasattr(lb, "detach") else lb
            bias = float(v.item() if hasattr(v, "item") else float(v))
    except Exception:
        pass
    if scale is None or not np.isfinite(scale) or scale <= 0:
        scale = 100.0
    return scale, bias


def calibrated_similarity(cosine, logit_scale: float = 100.0, logit_bias: float = 0.0):
    """Raw cosine -> the model's own logit space.

    This is the single most important line for the 'scores look tiny'
    complaint: 0.28 cosine on CLIP ViT-L/14 becomes a logit of 28, which is
    a strong match, not a weak one.
    """
    return np.asarray(cosine, dtype=np.float64) * float(logit_scale) + float(logit_bias)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def pool_confidence(scores: Sequence[float]) -> np.ndarray:
    """Confidence in [0, 1] for every candidate in one query's pool.

    The top candidate's confidence answers: *how far does the best hit stand
    out from everything else this query retrieved?* Two orthogonal signals:

        sep    = (s_top - mean) / std     how high above the pool it sits
        margin = (s_top - s_2nd) / std    how clearly it beats the runner-up

    A flat pool (the retriever found nothing discriminating) gives sep and
    margin near zero, so confidence lands near sigmoid(-BIAS) ~ 0.18 --
    correctly low. A dominant hit gives high sep and margin and approaches
    1.0. Lower-ranked candidates are scaled down from the top confidence by
    their relative score, so confidence is monotone with rank by
    construction and never exceeds the leader's.

    Constants are tunable in settings but need no fitting; they only set how
    aggressively separation converts to confidence.
    """
    a = np.asarray(scores, dtype=np.float64)
    n = a.size
    if n == 0:
        return a
    if n == 1:
        return np.array([float(_cfg("CONFIDENCE_SINGLETON", 0.5))])

    w_sep = float(_cfg("CONFIDENCE_SEP_WEIGHT", 0.6))
    w_margin = float(_cfg("CONFIDENCE_MARGIN_WEIGHT", 0.9))
    bias = float(_cfg("CONFIDENCE_BIAS", 1.5))
    gamma = float(_cfg("CONFIDENCE_RANK_GAMMA", 1.0))

    order = np.argsort(-a)
    s_sorted = a[order]
    sd = float(a.std())
    if sd < _EPS:
        # Every candidate scored identically: the retriever discriminated
        # nothing. Report the floor rather than a comfortable-looking number.
        return np.full(n, float(sigmoid(np.array([-bias]))[0]))

    sep = (s_sorted[0] - float(a.mean())) / sd
    margin = (s_sorted[0] - s_sorted[1]) / sd
    conf_top = float(sigmoid(np.array([w_sep * sep + w_margin * margin - bias]))[0])

    lo, hi = float(s_sorted[-1]), float(s_sorted[0])
    rel = np.ones(n) if hi - lo < _EPS else (a - lo) / (hi - lo)
    conf = conf_top * np.power(np.clip(rel, 0.0, 1.0), gamma)
    return np.clip(conf, 0.0, 1.0)


def assign_confidence(
    candidates: List[dict],
    score_key: str = "dense_logit",
    fallback_keys: Sequence[str] = ("fusion_score", "final_score", "clip_score"),
    out_key: str = "confidence",
) -> List[dict]:
    """Attach `confidence` to every candidate, in place.

    Confidence is always computed from ONE common signal shared by every
    candidate -- by default the calibrated dense logit -- never from
    `final_score`, which is deliberately banded by tier and would therefore
    report a cliff at every tier boundary.
    """
    if not candidates:
        return candidates

    key = None
    for cand_key in (score_key, *fallback_keys):
        if all(c.get(cand_key) is not None for c in candidates):
            key = cand_key
            break
    if key is None:
        for c in candidates:
            c[out_key] = 0.0
            c["confidence_basis"] = "unavailable"
        return candidates

    conf = pool_confidence([float(c[key]) for c in candidates])
    for c, v in zip(candidates, conf):
        c[out_key] = float(v)
        c["confidence_basis"] = key
    return candidates


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def band_score(raw_scores: Sequence[float], tier: int) -> np.ndarray:
    """Map a tier's raw scores into that tier's reserved [lo, hi] band.

    Replaces the old `score - 1e6` trick. Ordering across tiers is preserved
    because the bands do not overlap, and nothing leaves [0, 1], so any UI
    that renders the number as a bar or a percentage stays sane.
    """
    lo, hi = _TIER_BAND.get(tier, (0.0, 1.0))
    return lo + minmax(raw_scores) * (hi - lo)


def sort_key(c: dict):
    """Total order: better tier first, then higher score, then stable by id."""
    return (
        int(c.get("tier", TIER_FUSED)),
        -float(c.get("final_score", 0.0)),
        str(c.get("video_id", "")),
        int(c.get("frame_idx", 0)),
    )


def finalize(candidates: List[dict], score_field: str = "final_score") -> List[dict]:
    """Band every tier, attach confidence, sort. The single exit point that
    guarantees the invariant: 0 <= final_score <= 1 and 0 <= confidence <= 1."""
    if not candidates:
        return candidates

    by_tier: Dict[int, List[dict]] = {}
    for c in candidates:
        by_tier.setdefault(int(c.get("tier", TIER_FUSED)), []).append(c)

    for tier, group in by_tier.items():
        raw = [float(c.get("_raw_score", c.get(score_field, 0.0))) for c in group]
        banded = band_score(raw, tier)
        for c, v in zip(group, banded):
            c[score_field] = float(v)

    assign_confidence(candidates)
    candidates.sort(key=sort_key)
    return candidates


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class ScoreAudit:
    """What the scoring layer saw. Log this when a result looks wrong."""
    n_candidates: int = 0
    tiers: Dict[int, int] = field(default_factory=dict)
    score_min: float = 0.0
    score_max: float = 0.0
    conf_min: float = 0.0
    conf_max: float = 0.0
    conf_basis: str = ""
    flat_pool: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def audit(candidates: List[dict]) -> ScoreAudit:
    if not candidates:
        return ScoreAudit()
    scores = [float(c.get("final_score", 0.0)) for c in candidates]
    confs = [float(c.get("confidence", 0.0)) for c in candidates]
    tiers: Dict[int, int] = {}
    for c in candidates:
        t = int(c.get("tier", TIER_FUSED))
        tiers[t] = tiers.get(t, 0) + 1
    return ScoreAudit(
        n_candidates=len(candidates),
        tiers=tiers,
        score_min=min(scores), score_max=max(scores),
        conf_min=min(confs), conf_max=max(confs),
        conf_basis=str(candidates[0].get("confidence_basis", "")),
        flat_pool=(max(confs) - min(confs)) < 1e-6,
    )


# ---------------------------------------------------------------------------
# Hook for real probability calibration, once labels exist
# ---------------------------------------------------------------------------

def fit_isotonic(scores: Sequence[float], labels: Sequence[int]):
    """Fit score -> P(correct) on labelled data.

    Not used by the pipeline. This is the documented upgrade path from
    'relative confidence' to an actual probability. Needs roughly 200+
    labelled (score, is_correct) pairs before the fit means anything.
    Returns a callable, or None if scikit-learn is unavailable.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception:
        return None
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64))
    return lambda s: float(ir.predict([float(s)])[0])
