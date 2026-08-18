#!/usr/bin/env python3
"""
Local pipeline setup: copy videos, extract keyframes, generate embeddings.

Usage:
    python scripts/setup_local.py --videos-dir /path/to/videos
    python scripts/setup_local.py --videos-dir /path/to/videos --skip-keyframes
    python scripts/setup_local.py --videos-dir /path/to/videos --skip-embeddings
"""

import argparse
import shutil
import re
import time
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings


def main():
    parser = argparse.ArgumentParser(description="Local pipeline setup")
    parser.add_argument("--videos-dir", required=True, help="Directory with .mp4 videos")
    parser.add_argument("--skip-keyframes", action="store_true", help="Skip keyframe extraction")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    data_dir = settings.PROJECT_ROOT / "data"
    video_dir = data_dir / "videos"
    keyframe_dir = data_dir / "keyframes"
    store_dir = data_dir / "persistent_store"

    # Step 1: Copy videos
    print("=" * 60)
    print("Step 1: Copying videos...")
    print("=" * 60)
    video_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.stem for f in video_dir.glob("*.mp4"))
    copied = 0
    for f in videos_dir.glob("*.mp4"):
        if f.stem not in existing:
            shutil.copy2(f, video_dir / f.name)
            copied += 1
    total = len(list(video_dir.glob("*.mp4")))
    print(f"  Copied {copied} new videos -> {video_dir} ({total} total)\n")

    # Step 2: Extract keyframes
    if not args.skip_keyframes:
        print("=" * 60)
        print("Step 2: Extracting keyframes...")
        print("=" * 60)
        if keyframe_dir.exists() and len(list(keyframe_dir.glob("*.jpg"))) > 100:
            print(f"  Keyframes already exist in {keyframe_dir} ({len(list(keyframe_dir.glob('*.jpg')))} files) -- skipping\n")
        else:
            from preprocessing.pipeline import PreprocessingPipeline
            pipeline = PreprocessingPipeline()
            video_files = sorted(video_dir.glob("*.mp4"))
            video_paths = [str(v) for v in video_files]
            print(f"  Processing {len(video_paths)} videos...")
            t0 = time.time()
            results = pipeline.process_batch(video_paths, save_keyframes=True, max_workers=4)
            elapsed = time.time() - t0
            ok = sum(1 for r in results if r.get("status") == "success")
            print(f"  Done: {ok}/{len(video_paths)} videos in {elapsed:.1f}s\n")
    else:
        print("Step 2: SKIPPED (keyframes)\n")

    # Step 3: Generate embeddings
    if not args.skip_embeddings:
        print("=" * 60)
        print("Step 3: Generating embeddings...")
        print("=" * 60)
        from retrieval.persistent_store import PersistentVectorStore
        store = PersistentVectorStore(str(store_dir))
        if len(store._ids) > 10:
            print(f"  Embeddings already exist ({len(store._ids)} frames) -- skipping\n")
        else:
            _generate_embeddings(keyframe_dir, store_dir)
    else:
        print("Step 3: SKIPPED (embeddings)\n")

    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"""
Next steps:
  1. Run contest queries:
     python scripts/run_contest.py --queries-dir queries/ --output-dir submission/

  2. Query file naming (auto-detects mode):
     video_001-kis.txt   -> TKIS (retrieval only)
     video_001-qa.txt    -> Q&A (retrieval + VLM answer)
     video_001-trake.txt -> TRAKE (multi-event alignment)
""")


def _generate_embeddings(keyframe_dir: Path, store_dir: Path):
    from PIL import Image
    from embedding.hybrid_engine import HybridEmbeddingEngine
    from retrieval.persistent_store import PersistentVectorStore

    print("  Loading models...")
    engine = HybridEmbeddingEngine()
    store = PersistentVectorStore(str(store_dir))

    kf_re = re.compile(r'^(.+?)_frame_(\d+)\.jpg$')
    grouped = defaultdict(list)
    for fp in sorted(keyframe_dir.glob("*.jpg")):
        m = kf_re.match(fp.name)
        if m:
            grouped[m.group(1)].append((int(m.group(2)), fp))

    video_names = sorted(grouped.keys())
    total_frames = sum(len(v) for v in grouped.values())
    print(f"  Found {len(video_names)} videos, {total_frames} keyframes")

    batch_size = 64
    total_done = 0
    t0 = time.time()

    for vi, vname in enumerate(video_names):
        frames = sorted(grouped[vname], key=lambda x: x[0])
        images = []
        for _, p in frames:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"    Warning: {p.name}: {e}")

        if not images:
            continue

        frame_indices = [fi for fi, _ in frames[:len(images)]]
        timestamps = [float(fi) for fi in frame_indices]

        all_clip = []
        all_siglip2 = []
        for bi in range(0, len(images), batch_size):
            batch = images[bi:bi + batch_size]
            embs = engine.encode_images_batch(batch)
            all_clip.append(embs["clip"])
            all_siglip2.append(embs["siglip2"])

        import numpy as np
        clip_embs = np.concatenate(all_clip, axis=0)
        siglip2_embs = np.concatenate(all_siglip2, axis=0)

        store.insert_frames(
            video_id=vname,
            frame_indices=frame_indices,
            clip_embeddings=clip_embs,
            siglip2_embeddings=siglip2_embs,
            timestamps=timestamps,
        )

        total_done += len(images)
        elapsed = time.time() - t0
        fps = total_done / elapsed if elapsed > 0 else 0
        print(f"  [{vi+1}/{len(video_names)}] {vname}: {len(images)} frames ({total_done} total, {fps:.1f} fps)")

    elapsed = time.time() - t0
    print(f"\n  Done: {total_done} frames in {elapsed:.1f}s ({total_done/elapsed:.1f} fps)")
    print(f"  Store: {store.get_collection_stats()}\n")


if __name__ == "__main__":
    main()
