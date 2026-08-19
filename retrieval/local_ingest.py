"""
Ingest existing local data into Milvus Lite on startup.

Runs only when the Milvus collection is empty, in priority order:
  1. persistent_store — already-embedded frames (pickle), inserted directly.
  2. keyframes        — images re-embedded with TULIP + SigLIP2.
  3. videos           — full keyframe-extraction + embedding pipeline.
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger

from config import settings


def _group_keyframes() -> dict:
    """Return {video_name: [(frame_idx, Path), ...]} from data/keyframes."""
    kf_dir = settings.KEYFRAME_DIR
    if not kf_dir.exists():
        return {}

    kf_re = re.compile(r"^(.+?)_frame_(\d+)\.jpg$")
    grouped = defaultdict(list)

    # Flat format: data/keyframes/{name}_frame_{idx}.jpg
    for fp in sorted(kf_dir.glob("*.jpg")):
        m = kf_re.match(fp.name)
        if m:
            grouped[m.group(1)].append((int(m.group(2)), fp))

    # Subdirectory format: data/keyframes/{name}/{name}_frame_{idx}.jpg
    for sub_dir in kf_dir.iterdir():
        if not sub_dir.is_dir():
            continue
        for fp in sorted(sub_dir.glob("*.jpg")):
            m = kf_re.match(fp.name)
            if m:
                grouped[sub_dir.name].append((int(m.group(2)), fp))

    return dict(grouped)


def _ingest_keyframes(milvus_store, engine) -> int:
    grouped = _group_keyframes()
    if not grouped:
        return 0

    batch_size = 64
    total = 0
    for vname in sorted(grouped):
        frames = sorted(set(grouped[vname]), key=lambda x: x[0])
        images = []
        for _, p in frames:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                logger.warning(f"Could not load keyframe {p.name}: {e}")
        if not images:
            continue

        frame_indices = [fi for fi, _ in frames[:len(images)]]

        all_clip, all_sig = [], []
        for bi in range(0, len(images), batch_size):
            embs = engine.encode_images_batch(images[bi:bi + batch_size])
            all_clip.append(embs["clip"])
            all_sig.append(embs["siglip2"])
        clip_embs = np.concatenate(all_clip, axis=0)
        sig_embs = np.concatenate(all_sig, axis=0)

        milvus_store.insert_frames(
            video_id=vname,
            frame_indices=frame_indices,
            timestamps=[float(fi) for fi in frame_indices],
            clip_embeddings=clip_embs,
            siglip2_embeddings=sig_embs,
        )
        total += len(images)
        logger.info(f"Ingested {len(images)} keyframes for video '{vname}'")

    return total


def _ingest_videos(milvus_store, pipeline) -> int:
    video_dir = settings.VIDEO_DIR
    if not video_dir.exists():
        return 0
    video_files = sorted(video_dir.glob("*.mp4"))
    if not video_files:
        return 0

    total = 0
    for vp in video_files:
        try:
            res = pipeline.process_video(video_path=str(vp))
            total += int(res.get("keyframes", 0) or 0)
        except Exception as e:
            logger.warning(f"Failed to process video {vp.name}: {e}")
    return total


def ingest_local_data_if_empty(milvus_store, embedding_engine=None, pipeline=None) -> dict:
    """Populate Milvus Lite from existing local data if the collection is empty."""
    if milvus_store is None:
        return {"status": "no_store"}

    if not milvus_store.is_empty():
        n = milvus_store.get_collection_stats().get("num_entities", 0)
        logger.info(f"Milvus already populated ({n} entities); skipping local ingest")
        return {"status": "already_populated", "num_entities": n}

    summary = {"status": "ingested", "persistent_store": 0, "keyframes": 0, "videos": 0}

    # 1. Migrate from persistent store (fast — embeddings already computed)
    try:
        summary["persistent_store"] = milvus_store.ingest_from_persistent_store()
    except Exception as e:
        logger.warning(f"Persistent-store migration failed: {e}")

    # 2. Re-embed keyframes (needs the embedding engine)
    if milvus_store.is_empty() and embedding_engine is not None:
        try:
            summary["keyframes"] = _ingest_keyframes(milvus_store, embedding_engine)
        except Exception as e:
            logger.warning(f"Keyframe ingestion failed: {e}")

    # 3. Process videos (needs the full pipeline)
    if milvus_store.is_empty() and pipeline is not None:
        try:
            summary["videos"] = _ingest_videos(milvus_store, pipeline)
        except Exception as e:
            logger.warning(f"Video ingestion failed: {e}")

    return summary
