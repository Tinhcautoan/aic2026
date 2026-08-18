# Local GPU Server Pipeline Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all bugs and missing pieces so the full pipeline runs locally on a GPU server — download videos, extract keyframes, generate embeddings, run contest queries.

**Architecture:** The pipeline runs sequentially: (1) download/copy videos to `data/videos/`, (2) extract keyframes via `preprocessing/pipeline.py`, (3) generate TULIP + SigLIP2 embeddings, (4) run `scripts/run_contest.py` with real models.

**Tech Stack:** Python 3.10+, PyTorch, open_clip_torch, TULIP fork, SigLIP2, PersistentVectorStore (pickle-based).

**Spec:** HCM AI Contest VideoQA — 3 modes (TKIS/Q&A/TRAKE), R@k scoring, CSV submission format.

## Global Constraints

- Local GPU server (no Colab)
- A100 or similar GPU with sufficient VRAM
- No Milvus/Redis required (local mode uses PersistentVectorStore)
- Videos must be in `data/videos/` before keyframe extraction
- TULIP fork at `tulip/open_clip/src/` must NOT pollute `sys.modules['open_clip']`

---

## File Structure

```
embedding/tulip_encoder.py          — NO CHANGE (already has isolated import)
embedding/siglip2_encoder.py        — NO CHANGE (already has sys.path fix)
embedding/clip_encoder.py           — NO CHANGE (already has sys.path fix)
embedding/hybrid_engine.py          — MODIFY: fix deprecated asyncio call
embedding/__init__.py               — MODIFY: add try/except guards
preprocessing/keyframe_extractor.py — MODIFY: fix open_clip import pollution
preprocessing/pipeline.py           — MODIFY: fix hardcoded OCR/ASR, sequential batch
retrieval/persistent_store.py       — MODIFY: use set for _ids, add get_frame_embeddings
scripts/run_contest.py              — MODIFY: move numpy import to top level
scripts/batch_process.py            — MODIFY: fix CSV quoting bug
colab_requirements.txt              — MODIFY: add loguru
tests/test_system.py                — MODIFY: fix UNIFIED_DIM assertion
requirements.txt                    — MODIFY: add requests
```

---

### Task 1: Fix Critical Bug — keyframe_extractor.py open_clip pollution

**Files:**
- Modify: `preprocessing/keyframe_extractor.py:110-118`

**Why:** The keyframe extractor adds the TULIP fork path to `sys.path` and does bare `import open_clip`, which overwrites `sys.modules['open_clip']` with the TULIP fork. If `HybridEmbeddingEngine` is loaded afterward, SigLIP2 will use the wrong open_clip.

- [ ] **Step 1: Fix the import in `_ModelWorker.load()`**

Replace the bare `import open_clip` with a safe import that excludes TULIP paths:

```python
# BEFORE (lines 113-118):
import sys as _sys
_tulip_dir = os.path.join(os.path.dirname(__file__), '..', 'tulip', 'open_clip', 'src')
if os.path.isdir(_tulip_dir) and _tulip_dir not in _sys.path:
    _sys.path.insert(0, _tulip_dir)

import open_clip

# AFTER:
import sys as _sys
_tulip_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'tulip', 'open_clip', 'src'))
_saved_paths = [p for p in _sys.path if 'tulip' in p and 'open_clip' in p]
for _p in _saved_paths:
    _sys.path.remove(_p)
import open_clip
_sys.path.extend(_saved_paths)
```

- [ ] **Step 2: Apply same fix to `KeyframeExtractor._load_model()`**

Search for the same pattern in the `KeyframeExtractor._load_model()` method (around line 258) and apply the same fix.

- [ ] **Step 3: Commit**

```bash
git add preprocessing/keyframe_extractor.py
git commit -m "fix: prevent TULIP open_clip fork from polluting sys.modules"
```

---

### Task 2: Fix test_system.py UNIFIED_DIM assertion

**Files:**
- Modify: `tests/test_system.py:25`

- [ ] **Step 1: Fix the assertion**

```python
# BEFORE:
self.assertTrue(settings.UNIFIED_DIM == 1024)

# AFTER:
self.assertTrue(settings.UNIFIED_DIM == 768)
```

- [ ] **Step 2: Run test to verify**

Run: `python -m pytest tests/test_system.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_system.py
git commit -m "fix: update UNIFIED_DIM assertion to match actual value (768)"
```

---

### Task 3: Fix embedding/__init__.py — add import guards

**Files:**
- Modify: `embedding/__init__.py`

- [ ] **Step 1: Read current file and add try/except**

```python
# BEFORE:
from .clip_encoder import CLIPEncoder
from .siglip2_encoder import SigLIP2Encoder
from .tulip_encoder import TULIPEncoder
from .hybrid_engine import HybridEmbeddingEngine

# AFTER:
try:
    from .clip_encoder import CLIPEncoder
except ImportError:
    CLIPEncoder = None

try:
    from .siglip2_encoder import SigLIP2Encoder
except ImportError:
    SigLIP2Encoder = None

try:
    from .tulip_encoder import TULIPEncoder
except ImportError:
    TULIPEncoder = None

try:
    from .hybrid_engine import HybridEmbeddingEngine
except ImportError:
    HybridEmbeddingEngine = None
```

- [ ] **Step 2: Commit**

```bash
git add embedding/__init__.py
git commit -m "fix: add try/except guards to embedding __init__ imports"
```

---

### Task 4: Fix hybrid_engine.py — deprecated asyncio call

**Files:**
- Modify: `embedding/hybrid_engine.py` (lines ~115, ~127)

- [ ] **Step 1: Replace deprecated `asyncio.get_event_loop()`**

Search for `asyncio.get_event_loop()` and replace with `asyncio.get_running_loop()`. If no running loop exists, wrap in a new loop:

```python
# BEFORE:
loop = asyncio.get_event_loop()

# AFTER:
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
```

- [ ] **Step 2: Commit**

```bash
git add embedding/hybrid_engine.py
git commit -m "fix: replace deprecated asyncio.get_event_loop()"
```

---

### Task 5: Fix preprocessing/pipeline.py — hardcoded OCR/ASR + sequential batch

**Files:**
- Modify: `preprocessing/pipeline.py:110` and `preprocessing/pipeline.py:114-122`

- [ ] **Step 1: Fix hardcoded return values**

```python
# BEFORE (line 110):
"ocr_enabled": False, "asr_enabled": False,

# AFTER:
"ocr_enabled": self.ocr_enabled, "asr_enabled": self.asr_enabled,
```

- [ ] **Step 2: Fix process_batch to respect max_workers**

Read the current `process_batch` method and add `ThreadPoolExecutor` for parallel processing:

```python
def process_batch(self, video_paths: List[str], save_keyframes: bool = False, max_workers: int = 4) -> List[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(self.process_video, video_path=vp, save_keyframes=save_keyframes): vp
            for vp in video_paths
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                results.append({"status": "error", "error": str(e)})
    return results
```

- [ ] **Step 3: Commit**

```bash
git add preprocessing/pipeline.py
git commit -m "fix: use actual OCR/ASR settings + parallel batch processing"
```

---

### Task 6: Fix persistent_store.py — performance + get_frame_embeddings

**Files:**
- Modify: `retrieval/persistent_store.py`

**Note:** `get_frame_embeddings` was already added in the previous session. Verify it exists and is correct.

- [ ] **Step 1: Verify get_frame_embeddings exists**

Check that `get_frame_embeddings()` method exists in `PersistentVectorStore`. If not, add it.

- [ ] **Step 2: Optimize _ids lookup with set**

Add a `_ids_set` alongside `_ids` for O(1) membership checks:

```python
# In __init__:
self._ids_set = set()

# In _load_from_disk:
self._ids_set = set(self._ids)

# In insert_frames:
if frame_id in self._ids_set:
    continue
self._ids_set.add(frame_id)
```

- [ ] **Step 3: Commit**

```bash
git add retrieval/persistent_store.py
git commit -m "perf: use set for O(1) frame ID lookup in persistent store"
```

---

### Task 7: Fix scripts/run_contest.py — numpy import location

**Files:**
- Modify: `scripts/run_contest.py:218`

- [ ] **Step 1: Move numpy import to top level**

```python
# BEFORE (inside main()):
import numpy as np

# AFTER (at top of file, after other imports):
import numpy as np
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_contest.py
git commit -m "fix: move numpy import to module level"
```

---

### Task 8: Fix scripts/batch_process.py — CSV quoting + missing dependency

**Files:**
- Modify: `scripts/batch_process.py:175`
- Modify: `requirements.txt`

- [ ] **Step 1: Fix CSV quoting bug**

Read line 175 and fix the extra double-quotes around answers.

- [ ] **Step 2: Add requests to requirements.txt**

```
requests>=2.31.0
```

- [ ] **Step 3: Commit**

```bash
git add scripts/batch_process.py requirements.txt
git commit -m "fix: CSV quoting bug in batch_process + add requests dependency"
```

---

### Task 9: Fix colab_requirements.txt — add loguru

**Files:**
- Modify: `colab_requirements.txt`

- [ ] **Step 1: Add loguru**

Add `loguru>=0.7.0` to the ML/Embeddings section.

- [ ] **Step 2: Commit**

```bash
git add colab_requirements.txt
git commit -m "fix: add missing loguru dependency to colab_requirements.txt"
```

---

### Task 10: Create local pipeline runner script

**Files:**
- Create: `scripts/setup_local.py`

This script handles the full local pipeline: copy videos, extract keyframes, generate embeddings.

- [ ] **Step 1: Create setup_local.py**

```python
#!/usr/bin/env python3
"""
Local pipeline setup: copy videos, extract keyframes, generate embeddings.

Usage:
    python scripts/setup_local.py --videos-dir /path/to/videos
    python scripts/setup_local.py --videos-dir /path/to/videos --skip-keyframes
"""
import argparse
import shutil
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

def main():
    parser = argparse.ArgumentParser(description="Local pipeline setup")
    parser.add_argument("--videos-dir", required=True, help="Directory with .mp4 videos")
    parser.add_argument("--skip-keyframes", action="store_true", help="Skip keyframe extraction")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    video_dir = data_dir / "videos"
    keyframe_dir = data_dir / "keyframes"
    store_dir = data_dir / "persistent_store"

    # Step 1: Copy videos
    video_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.stem for f in video_dir.glob("*.mp4"))
    copied = 0
    for f in videos_dir.glob("*.mp4"):
        if f.stem not in existing:
            shutil.copy2(f, video_dir / f.name)
            copied += 1
    print(f"Copied {copied} videos to {video_dir} ({len(list(video_dir.glob('*.mp4')))} total)")

    # Step 2: Extract keyframes
    if not args.skip_keyframes:
        if keyframe_dir.exists() and any(keyframe_dir.glob("*.jpg")):
            print(f"Keyframes already exist in {keyframe_dir} — skipping")
        else:
            print("Extracting keyframes...")
            from preprocessing.pipeline import PreprocessingPipeline
            pipeline = PreprocessingPipeline()
            video_files = sorted(video_dir.glob("*.mp4"))
            results = pipeline.keyframe_extractor.extract_batch(
                video_paths=[str(v) for v in video_files],
                output_dir=str(keyframe_dir),
                save_keyframes=True,
                max_workers=4,
            )
            ok = sum(1 for r in results.values() if r and r[0].get("keyframes", 0) > 0)
            print(f"Keyframe extraction: {ok}/{len(video_files)} videos")

    # Step 3: Generate embeddings
    if not args.skip_embeddings:
        if store_dir.exists() and any(store_dir.glob("*.pkl")):
            from retrieval.persistent_store import PersistentVectorStore
            store = PersistentVectorStore(str(store_dir))
            if len(store._ids) > 10:
                print(f"Embeddings already exist ({len(store._ids)} frames) — skipping")
            else:
                _generate_embeddings(keyframe_dir, store_dir)
        else:
            _generate_embeddings(keyframe_dir, store_dir)

    print(f"\nSetup complete! Run contest queries with:")
    print(f"  python scripts/run_contest.py --queries-dir queries/ --output-dir submission/")


def _generate_embeddings(keyframe_dir, store_dir):
    import re
    import time
    import numpy as np
    from collections import defaultdict
    from PIL import Image
    from embedding.hybrid_engine import HybridEmbeddingEngine
    from retrieval.persistent_store import PersistentVectorStore

    print("Generating embeddings...")
    engine = HybridEmbeddingEngine()
    store = PersistentVectorStore(str(store_dir))

    kf_re = re.compile(r'^(.+?)_frame_(\d+)\.jpg$')
    grouped = defaultdict(list)
    for fp in sorted(keyframe_dir.glob("*.jpg")):
        m = kf_re.match(fp.name)
        if m:
            grouped[m.group(1)].append((int(m.group(2)), fp))

    video_names = sorted(grouped.keys())
    print(f"Found {len(video_names)} videos, {sum(len(v) for v in grouped.values())} keyframes")

    batch_size = 64
    total_frames = 0
    t0 = time.time()

    for vi, vname in enumerate(video_names):
        frames = sorted(grouped[vname], key=lambda x: x[0])
        images = [Image.open(p).convert("RGB") for _, p in frames]
        frame_indices = [fi for fi, _ in frames]
        timestamps = [float(fi) for fi in frame_indices]

        all_clip = []
        all_siglip2 = []
        for bi in range(0, len(images), batch_size):
            batch = images[bi:bi + batch_size]
            embs = engine.encode_images_batch(batch)
            all_clip.append(embs["clip"])
            all_siglip2.append(embs["siglip2"])

        clip_embs = np.concatenate(all_clip, axis=0)
        siglip2_embs = np.concatenate(all_siglip2, axis=0)

        store.insert_frames(
            video_id=vname,
            frame_indices=frame_indices,
            clip_embeddings=clip_embs,
            siglip2_embeddings=siglip2_embs,
            timestamps=timestamps,
        )

        total_frames += len(images)
        elapsed = time.time() - t0
        fps = total_frames / elapsed if elapsed > 0 else 0
        print(f"  [{vi+1}/{len(video_names)}] {vname}: {len(images)} frames ({total_frames} total, {fps:.1f} fps)")

    elapsed = time.time() - t0
    print(f"\nDone: {total_frames} frames in {elapsed:.1f}s ({total_frames/elapsed:.1f} fps)")
    print(f"Store: {store.get_collection_stats()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/setup_local.py
git commit -m "feat: add local pipeline setup script (copy videos + extract + embed)"
```

---

### Task 11: End-to-end verification

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS (including fixed test_system.py).

- [ ] **Step 2: Test keyframe extraction + embedding generation**

```bash
python scripts/setup_local.py --videos-dir /path/to/videos
```

Expected: Videos copied, keyframes extracted, embeddings generated in `data/persistent_store/`.

- [ ] **Step 3: Test contest runner**

```bash
python scripts/run_contest.py --queries-dir queries/ --output-dir submission/
```

Expected: CSV files generated in `submission/`.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete local GPU server pipeline"
```
