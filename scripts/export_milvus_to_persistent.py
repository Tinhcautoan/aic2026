#!/usr/bin/env python3
"""
Export all frame embeddings from Milvus Lite to ``data/persistent_store/`` as pickles.

IMPORTANT: stop the running server (``run.py``) first — Milvus Lite keeps an
exclusive lock on ``milvus.db`` while any process has it open, so the script
cannot open the DB while the server is running.

Run this AFTER re-embedding with native 1152-dim SigLIP2 so the pickle fallback
store matches the Milvus schema (clip=768, siglip2=1152).  The written format
is identical to what ``PersistentVectorStore._save_to_disk`` produces, so the
existing fallback path (``PersistentVectorStore`` / ``ingest_from_persistent_store``)
works unchanged:

    metadata.pkl    -> {'ids': ["video_id:frame_idx", ...],
                        'metadata': {id: {video_id, frame_idx, timestamp, ...}}}
    embeddings.pkl  -> {'clip': [np.float32 768-dim, ...], 'siglip2': [np.float32 1152-dim, ...]}

Usage:
    python scripts/export_milvus_to_persistent.py                # all frames
    python scripts/export_milvus_to_persistent.py --video-id L21_V001
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Export Milvus frame embeddings -> persistent_store pickles"
    )
    parser.add_argument("--video-id", default=None, help="Only export one video (default: all)")
    parser.add_argument(
        "--output-dir",
        default=str(settings.DATA_DIR / "persistent_store"),
        help="Output directory (default: data/persistent_store)",
    )
    args = parser.parse_args()

    from retrieval.milvus_store import MilvusVectorStore

    try:
        store = MilvusVectorStore()
    except Exception as e:
        print(
            f"Could not open Milvus ({e}).\n"
            f"Make sure the server (run.py) is STOPPED — Milvus Lite locks "
            f"milvus.db while it is open.",
            file=sys.stderr,
        )
        sys.exit(1)

    fields = ["id", "video_id", "frame_idx", "timestamp",
              "clip_embedding", "siglip2_embedding", "metadata"]
    expr = f'video_id == "{args.video_id}"' if args.video_id else ""

    t0 = time.time()
    rows = store.client.query(
        collection_name=store.collection_name,
        output_fields=fields,
        filter=expr,
        limit=300000,
    )
    print(f"Read {len(rows)} rows from Milvus in {time.time() - t0:.1f}s")
    if not rows:
        print("Nothing to export.")
        store.close()
        return

    ids = []
    metadata = {}
    clip_list = []
    siglip2_list = []
    for r in rows:
        vid = r.get("video_id", "")
        fid = f"{vid}:{r.get('frame_idx', 0)}"  # matches PersistentVectorStore id format
        ids.append(fid)
        meta = {
            "video_id": vid,
            "frame_idx": r.get("frame_idx", 0),
            "timestamp": float(r.get("timestamp", 0.0)),
        }
        meta.update(r.get("metadata") or {})
        metadata[fid] = meta
        clip_list.append(np.asarray(r.get("clip_embedding"), dtype=np.float32))
        siglip2_list.append(np.asarray(r.get("siglip2_embedding"), dtype=np.float32))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_file = out_dir / "metadata.pkl"
    emb_file = out_dir / "embeddings.pkl"

    t1 = time.time()
    with open(meta_file, "wb") as f:
        pickle.dump({"ids": ids, "metadata": metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(emb_file, "wb") as f:
        pickle.dump({"clip": clip_list, "siglip2": siglip2_list}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(ids)} frames to {out_dir} in {time.time() - t1:.1f}s")
    if clip_list:
        print(f"  clip dim: {clip_list[0].shape[0]} | siglip2 dim: {siglip2_list[0].shape[0]}")

    # Self-verify: reload with the real PersistentVectorStore class.
    from retrieval.persistent_store import PersistentVectorStore

    ps = PersistentVectorStore(str(out_dir))
    n = len(ps._ids)
    print(f"Verify: PersistentVectorStore loaded {n} frames")
    if n and clip_list:
        q = clip_list[0]
        hits = ps.search_clip(q, top_k=3)
        print(f"Verify: search_clip returns {len(hits)} hits; "
              f"top = {hits[0]['video_id']}:{hits[0]['frame_idx']} "
              f"(score {hits[0]['clip_score']:.3f})")
    store.close()


if __name__ == "__main__":
    main()
