"""
Proves the confidence bug and the fix, with no data and no models.

Reconstructs exactly what the old code produced for `final_rerank_score`
under each of its four branches, then shows what the new scoring layer
produces for the identical candidate pool.

Run:  python -m scripts.verify_scoring_fix
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from retrieval import scoring as S

RRF_K = 60


def make_pool(n=100, seed=0, separation=3.0):
    """A realistic dense pool: one clear winner, a long mediocre tail."""
    rng = np.random.default_rng(seed)
    cos = np.clip(rng.normal(0.18, 0.03, n), 0.02, 0.5)
    cos[0] = 0.18 + separation * 0.03          # the correct frame
    cos = np.sort(cos)[::-1]
    return cos


def old_scores(cos, cross_encoder_ran: bool, fusion_top_k=50):
    """Replicates the previous final_rerank_score assignment."""
    n = len(cos)
    out = []
    for i in range(n):
        if cross_encoder_ran:
            if i < fusion_top_k:
                # sigmoid of a bge logit; roughly this range in practice
                out.append(float(1 / (1 + np.exp(-(3.0 - 0.12 * i)))))
            else:
                rrf = 1.0 / (RRF_K + i + 1) + 1.0 / (RRF_K + i + 1)
                out.append(rrf - 1e6)
        else:
            # raw RRF, the common path. Caps at 2/(k+1).
            out.append(1.0 / (RRF_K + i + 1) + 1.0 / (RRF_K + i + 1))
    return np.array(out)


def new_scores(cos, cross_encoder_ran: bool, fusion_top_k=100, logit_scale=100.0):
    cands = []
    for i, c in enumerate(cos):
        tier = S.TIER_CROSS_ENCODED if (cross_encoder_ran and i < fusion_top_k) else S.TIER_FUSED
        raw = (float(1 / (1 + np.exp(-(3.0 - 0.12 * i))))
               if tier == S.TIER_CROSS_ENCODED
               else 1.0 / (RRF_K + i + 1) * 2)
        cands.append({
            "video_id": "V1", "frame_idx": i * 25,
            "tier": tier, "_raw_score": raw,
            "dense_logit": float(c) * logit_scale,   # the calibration step
        })
    S.finalize(cands)
    return cands


def show(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    for label, v in rows:
        print(f"  {label:<34} {v}")


print("=" * 74)
print("CONFIDENCE BUG — BEFORE vs AFTER")
print("=" * 74)

cos = make_pool(n=100, separation=3.0)
print(f"\nPool: 100 candidates, top cosine {cos[0]:.3f}, median {np.median(cos):.3f}")
print("(0.27 cosine on CLIP ViT-L/14 is a STRONG match — that is the point.)")

for ce in (False, True):
    label = "cross-encoder RAN" if ce else "cross-encoder SKIPPED (the common path)"
    old = old_scores(cos, ce)
    new = new_scores(cos, ce)

    show(f"OLD — {label}", [
        ("displayed top-1 'confidence'", f"{old[0]:.6f}"),
        ("displayed rank-60 'confidence'", f"{old[59]:.6f}"),
        ("min displayed value", f"{old.min():.6f}"),
        ("theoretical max of raw RRF", f"{2/(RRF_K+1):.6f}"),
    ])
    show(f"NEW — {label}", [
        ("top-1 confidence", f"{new[0]['confidence']:.4f}"),
        ("rank-60 confidence", f"{new[59]['confidence']:.4f}"),
        ("top-1 final_score (ordering)", f"{new[0]['final_score']:.4f}"),
        ("min final_score", f"{min(c['final_score'] for c in new):.4f}"),
        ("confidence basis", new[0]["confidence_basis"]),
    ])

print("\n" + "=" * 74)
print("CONFIDENCE TRACKS EVIDENCE, NOT WHICH BRANCH RAN")
print("=" * 74)
def controlled_pool(margin_sigmas, n=100, tail_mean=0.18, tail_sd=0.03):
    """Fixed tail, winner planted a known number of sigmas above the tail max.

    Using a random pool here would be misleading: the maximum of 100 gaussian
    draws already sits ~2.5 sigma above the mean, so even a 'no winner' pool
    looks separated. This isolates the planted margin.
    """
    tail = np.linspace(tail_mean + 1.5 * tail_sd, tail_mean - 2.0 * tail_sd, n - 1)
    top = tail[0] + margin_sigmas * tail_sd
    return np.concatenate([[top], tail])

print(f"\n  {'winner margin (sigma)':<24}{'top-1 confidence':<20}{'reading'}")
for margin, reading in [(0.0, "no winner: top ties the tail"),
                        (0.5, "weak"),
                        (1.5, "plausible"),
                        (3.0, "good"),
                        (6.0, "unambiguous")]:
    v = new_scores(controlled_pool(margin), cross_encoder_ran=False)[0]["confidence"]
    print(f"  {margin:<24.1f}{v:<20.4f}{reading}")

print("\n  Same pool, both branches — confidence must barely move:")
for ce in (False, True):
    v = new_scores(make_pool(separation=3.0), ce)[0]["confidence"]
    print(f"    cross-encoder {'ran    ' if ce else 'skipped'} -> {v:.4f}")

print("\n" + "=" * 74)
print("INVARIANTS")
print("=" * 74)
ok = True
for ce in (False, True):
    new = new_scores(make_pool(separation=3.0), ce)
    checks = [
        ("final_score within [0,1]", all(0.0 <= c["final_score"] <= 1.0 for c in new)),
        ("confidence within [0,1]", all(0.0 <= c["confidence"] <= 1.0 for c in new)),
        ("no -1e6 sentinel", all(c["final_score"] > -1 for c in new)),
        ("confidence monotone with rank", all(
            new[i]["confidence"] >= new[i + 1]["confidence"] - 1e-9 for i in range(len(new) - 1))),
        ("tiers never interleave", [c["tier"] for c in new] == sorted(c["tier"] for c in new)),
    ]
    for name, passed in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  (cross-encoder {'ran' if ce else 'skipped'})")

flat = S.pool_confidence([0.5] * 50)[0]
print(f"  [{'PASS' if flat < 0.25 else 'FAIL'}] flat pool reports low confidence ({flat:.4f})")
ok &= flat < 0.25

print("\n" + ("ALL INVARIANTS PASS" if ok else "SOME INVARIANTS FAILED"))
sys.exit(0 if ok else 1)
