"""
Keyframe Extraction Module using feature differencing.
Extracts distinctive keyframes while eliminating visual redundancy.

Algorithm:
- Computes relative change in features between consecutive frames
- feature_diff = ||f_current - f_prev|| / ||f_prev||
- Threshold of 0.6 empirically determined as optimal

Optimized for A100 80GB:
- Batch forward passes (32-64 frames per pass)
- Multi-threaded frame prefetching
- Parallel multi-video processing
"""

import os
import time
import threading
import queue
from pathlib import Path
from typing import List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from config import settings


# ────────────────────────────────────────────────────────────────────── #
#  Frame Prefetcher — reads frames from disk on a background thread    #
# ────────────────────────────────────────────────────────────────────── #

class _FramePrefetcher:
    """Producer thread: reads frames from cv2.VideoCapture into a queue."""

    def __init__(self, video_path: str, sample_rate: int, max_queue: int = 256):
        self._video_path = video_path
        self._sample_rate = sample_rate
        self._max_queue = max_queue
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        cap = cv2.VideoCapture(self._video_path)
        frame_idx = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                self._q.put(None)  # sentinel
                break
            if frame_idx % self._sample_rate == 0:
                self._q.put((frame_idx, frame))
            frame_idx += 1
        cap.release()

    def get_batch(self, batch_size: int) -> list:
        """Return up to batch_size (frame_idx, frame) tuples. None sentinel = EOF."""
        batch = []
        for _ in range(batch_size):
            try:
                item = self._q.get(timeout=10)
            except queue.Empty:
                break
            if item is None:
                batch.append(None)
                break
            batch.append(item)
        return batch

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ────────────────────────────────────────────────────────────────────── #
#  KeyframeExtractor                                                    #
# ────────────────────────────────────────────────────────────────────── #

class KeyframeExtractor:
    """Extracts keyframes from videos using feature differencing."""

    def __init__(
        self,
        threshold: Optional[float] = None,
        min_delta: Optional[int] = None,
        sample_rate: Optional[int] = None,
        image_size: Optional[int] = None,
        batch_size: int = 64,
    ):
        self.threshold: float = threshold if threshold is not None else settings.KEYFRAME_THRESHOLD
        self.min_delta: int = min_delta if min_delta is not None else settings.KEYFRAME_MIN_DELTA
        self.sample_rate: int = sample_rate if sample_rate is not None else settings.FRAME_SAMPLE_RATE
        self.image_size: int = image_size if image_size is not None else settings.IMAGE_SIZE
        self.batch_size: int = batch_size

        self._model: Any = None
        self._preprocess_fn: Any = None
        self._device: str = "cpu"
        self._model_lock = threading.Lock()

        logger.info(
            f"KeyframeExtractor initialized: threshold={self.threshold}, "
            f"min_delta={self.min_delta}, sample_rate={self.sample_rate}, "
            f"batch_size={self.batch_size}"
        )

    # ── Model loading ──────────────────────────────────────────────── #

    def _load_model(self) -> None:
        """Lazy-load the feature model (TULIP or CLIP)."""
        if self._model is not None:
            return

        import sys as _sys
        _tulip_dir = os.path.join(os.path.dirname(__file__), '..', 'tulip', 'open_clip', 'src')
        if os.path.isdir(_tulip_dir) and _tulip_dir not in _sys.path:
            _sys.path.insert(0, _tulip_dir)

        import open_clip
        import torch

        self._device = (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )
        model_name = settings.TULIP_MODEL_NAME if settings.USE_TULIP else settings.CLIP_MODEL_NAME
        pretrained = settings.TULIP_PRETRAINED if settings.USE_TULIP else settings.CLIP_PRETRAINED
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=self._device,
        )
        if settings.USE_FP16 and "cuda" in self._device:
            model = model.half()
        model.eval()
        self._model = model
        self._preprocess_fn = preprocess
        logger.info(f"Feature model loaded on {self._device} ({model_name})")

    # ── Feature computation ────────────────────────────────────────── #

    @staticmethod
    def _compute_feature_diff(current_features: np.ndarray, prev_features: np.ndarray) -> float:
        prev_norm = float(np.linalg.norm(prev_features))
        if prev_norm < 1e-8:
            return 0.0
        return float(np.linalg.norm(current_features - prev_features)) / prev_norm

    def _extract_features_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract features from a batch of frames. Returns (N, dim) array."""
        import torch

        tensors = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            t = self._preprocess_fn(pil_image)
            tensors.append(t)

        batch = torch.stack(tensors).to(self._device)
        if settings.USE_FP16 and "cuda" in self._device:
            batch = batch.half()

        with torch.no_grad():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)

    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Single-frame fallback (kept for compatibility)."""
        return self._extract_features_batch([frame])[0]

    # ── Single video extraction (batched + prefetched) ─────────────── #

    def extract_from_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        save_keyframes: bool = False,
    ) -> List[dict]:
        """Extract keyframes from a video file using batch inference + prefetch."""
        self._load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = Path(video_path).stem
        cap.release()

        logger.info(
            f"Extracting keyframes from '{video_name}': "
            f"{total_frames} frames @ {fps:.2f} FPS (batch={self.batch_size})"
        )

        if save_keyframes and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        keyframes: List[dict] = []
        prev_features: Optional[np.ndarray] = None
        last_keyframe_idx = -self.min_delta

        start_time = time.time()

        with _FramePrefetcher(video_path, self.sample_rate) as prefetcher:
            while True:
                batch_items = prefetcher.get_batch(self.batch_size)
                if not batch_items:
                    break

                # Filter out sentinel
                valid = [(idx, fr) for idx, fr in batch_items if fr is not None]
                if not valid:
                    break

                frame_indices = [item[0] for item in valid]
                frames = [item[1] for item in valid]

                # Batch GPU forward pass (one call for N frames)
                try:
                    all_features = self._extract_features_batch(frames)
                except Exception as e:
                    logger.warning(f"Batch inference failed, falling back to single: {e}")
                    all_features = np.stack([self._extract_features(fr) for fr in frames])

                # Sequential diff computation on CPU (fast)
                for local_i, frame_idx in enumerate(frame_indices):
                    features = all_features[local_i]
                    frame = frames[local_i]

                    if prev_features is None:
                        prev_features = features
                        info = self._save_keyframe(frame, frame_idx, frame_idx / fps, video_name, output_dir, save_keyframes)
                        info["features"] = features
                        keyframes.append(info)
                        last_keyframe_idx = frame_idx
                        continue

                    feat_diff = self._compute_feature_diff(features, prev_features)

                    if feat_diff > self.threshold and (frame_idx - last_keyframe_idx) >= self.min_delta:
                        info = self._save_keyframe(frame, frame_idx, frame_idx / fps, video_name, output_dir, save_keyframes)
                        info["features"] = features
                        keyframes.append(info)
                        last_keyframe_idx = frame_idx
                        prev_features = features

                # Check if we hit sentinel (EOF) in this batch
                if any(item is None for item in batch_items):
                    break

        elapsed = time.time() - start_time
        reduction = (1 - len(keyframes) / max(total_frames / self.sample_rate, 1)) * 100

        logger.info(
            f"Extracted {len(keyframes)} keyframes from {total_frames} frames "
            f"({reduction:.1f}% reduction) in {elapsed:.2f}s "
            f"({total_frames / elapsed:.0f} fps)"
        )
        return keyframes

    def _save_keyframe(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        video_name: str,
        output_dir: Optional[str],
        save: bool,
    ) -> dict:
        info: dict = {"frame_idx": frame_idx, "timestamp": timestamp, "path": None}
        if save and output_dir:
            filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            info["path"] = filepath
        return info

    # ── Multi-video extraction (parallel) ──────────────────────────── #

    def extract_batch(
        self,
        video_paths: List[str],
        output_dir: Optional[str] = None,
        save_keyframes: bool = False,
        max_workers: int = 2,
    ) -> dict:
        """Extract keyframes from multiple videos in parallel.

        max_workers=2 is safe: each video uses ~2GB VRAM for inference,
        leaving plenty of room for the embedding models later.
        """
        results: dict = {}
        total = len(video_paths)

        def _process_one(vp: str) -> tuple:
            try:
                kfs = self.extract_from_video(
                    video_path=vp, output_dir=output_dir, save_keyframes=save_keyframes
                )
                return (vp, kfs, None)
            except Exception as e:
                logger.error(f"Failed: {vp}: {e}")
                return (vp, [], str(e))

        logger.info(f"Processing {total} videos with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_one, vp): vp for vp in video_paths}
            done_count = 0
            for future in as_completed(futures):
                vp, kfs, err = future.result()
                results[vp] = kfs
                done_count += 1
                vid_name = Path(vp).stem
                status = f"{len(kfs)} keyframes" if not err else f"FAILED: {err}"
                logger.info(f"  [{done_count}/{total}] {vid_name} — {status}")

        ok = sum(1 for v in results.values() if v)
        logger.info(f"Batch done: {ok}/{total} videos succeeded")
        return results
