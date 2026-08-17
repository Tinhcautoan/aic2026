"""
Keyframe Extraction Module using feature differencing.
Extracts distinctive keyframes while eliminating visual redundancy.

Algorithm:
- Computes relative change in features between consecutive frames
- feature_diff = ||f_current - f_prev|| / ||f_prev||
- Threshold of 0.6 empirically determined as optimal
"""

import os
import time
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Any
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from config import settings


class KeyframeExtractor:
    """Extracts keyframes from videos using feature differencing."""

    def __init__(
        self,
        threshold: Optional[float] = None,
        min_delta: Optional[int] = None,
        sample_rate: Optional[int] = None,
        image_size: Optional[int] = None,
    ):
        self.threshold: float = threshold if threshold is not None else settings.KEYFRAME_THRESHOLD
        self.min_delta: int = min_delta if min_delta is not None else settings.KEYFRAME_MIN_DELTA
        self.sample_rate: int = sample_rate if sample_rate is not None else settings.FRAME_SAMPLE_RATE
        self.image_size: int = image_size if image_size is not None else settings.IMAGE_SIZE

        self._model: Any = None
        self._preprocess_fn: Any = None
        self._device: str = "cpu"

        logger.info(
            f"KeyframeExtractor initialized: threshold={self.threshold}, "
            f"min_delta={self.min_delta}, sample_rate={self.sample_rate}"
        )

    def _load_model(self) -> None:
        """Lazy-load the feature model (TULIP or CLIP)."""
        if self._model is not None:
            return

        import sys as _sys
        _tulip_dir = os.path.join(os.path.dirname(__file__), '..', 'tulip', 'open_clip', 'src')
        if os.path.isdir(_tulip_dir) and _tulip_dir not in _sys.path:
            _sys.path.insert(0, _tulip_dir)

        import open_clip  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]

        self._device = (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )
        model_name = settings.TULIP_MODEL_NAME if settings.USE_TULIP else settings.CLIP_MODEL_NAME
        pretrained = settings.TULIP_PRETRAINED if settings.USE_TULIP else settings.CLIP_PRETRAINED
        model, _, preprocess = open_clip.create_model_and_transforms(  # type: ignore[arg-type]
            model_name=model_name,
            pretrained=pretrained,
            device=self._device,
        )
        if settings.USE_FP16 and "cuda" in self._device:
            model = model.half()  # type: ignore[assignment,union-attr]
        model.eval()  # type: ignore[union-attr]
        self._model = model
        self._preprocess_fn = preprocess  # type: ignore[assignment]
        logger.info(f"Feature model loaded for keyframe extraction on {self._device} ({model_name})")

    def _compute_feature_diff(
        self,
        current_features: np.ndarray,
        prev_features: np.ndarray,
    ) -> float:
        """Compute normalized Euclidean norm of feature difference."""
        prev_norm = float(np.linalg.norm(prev_features))
        if prev_norm < 1e-8:
            return 0.0
        diff_norm = float(np.linalg.norm(current_features - prev_features))
        return diff_norm / prev_norm

    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract features from a video frame (BGR -> RGB -> PIL -> model)."""
        import torch

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        image_tensor: Any = self._preprocess_fn(pil_image).unsqueeze(0).to(self._device)
        if settings.USE_FP16 and "cuda" in self._device:
            image_tensor = image_tensor.half()

        with torch.no_grad():
            features: Any = self._model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32).flatten()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def extract_from_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        save_keyframes: bool = False,
    ) -> List[dict]:
        """Extract keyframes from a video file."""
        self._load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = Path(video_path).stem

        logger.info(
            f"Extracting keyframes from '{video_name}': "
            f"{total_frames} frames @ {fps:.2f} FPS"
        )

        keyframes: List[dict] = []
        prev_features: Optional[np.ndarray] = None
        last_keyframe_idx = -self.min_delta
        frame_idx = 0

        if save_keyframes and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sample_rate != 0:
                frame_idx += 1
                continue

            try:
                features = self._extract_features(frame)
            except Exception as e:
                logger.warning(f"Failed to extract features at frame {frame_idx}: {e}")
                frame_idx += 1
                continue

            if prev_features is None:
                prev_features = features
                info = self._save_keyframe(frame, frame_idx, frame_idx / fps, video_name, output_dir, save_keyframes)
                info["features"] = features
                keyframes.append(info)
                last_keyframe_idx = frame_idx
                frame_idx += 1
                continue

            feat_diff = self._compute_feature_diff(features, prev_features)

            if feat_diff > self.threshold and (frame_idx - last_keyframe_idx) >= self.min_delta:
                info = self._save_keyframe(frame, frame_idx, frame_idx / fps, video_name, output_dir, save_keyframes)
                info["features"] = features
                keyframes.append(info)
                last_keyframe_idx = frame_idx
                prev_features = features

            frame_idx += 1

        cap.release()

        elapsed = time.time() - start_time
        reduction = (1 - len(keyframes) / max(total_frames / self.sample_rate, 1)) * 100

        logger.info(
            f"Extracted {len(keyframes)} keyframes from {total_frames} frames "
            f"({reduction:.1f}% reduction) in {elapsed:.2f}s"
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
        """Save a keyframe image and return its metadata."""
        info: dict = {"frame_idx": frame_idx, "timestamp": timestamp, "path": None}
        if save and output_dir:
            filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            info["path"] = filepath
        return info

    def extract_batch(
        self,
        video_paths: List[str],
        output_dir: Optional[str] = None,
        save_keyframes: bool = False,
    ) -> dict:
        """Extract keyframes from multiple videos."""
        results: dict = {}
        for vp in video_paths:
            try:
                results[vp] = self.extract_from_video(video_path=vp, output_dir=output_dir, save_keyframes=save_keyframes)
            except Exception as e:
                logger.error(f"Failed to extract from {vp}: {e}")
                results[vp] = []
        return results