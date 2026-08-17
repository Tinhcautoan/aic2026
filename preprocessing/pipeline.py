"""
Full Preprocessing Pipeline: Keyframe Extraction -> OCR -> ASR -> Embedding -> Store.
Orchestrates the end-to-end ingestion of video data.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from PIL import Image
from loguru import logger

from config import settings
from .keyframe_extractor import KeyframeExtractor


class PreprocessingPipeline:
    """
    Structured preprocessing pipeline for video ingestion:
    1. Keyframe extraction (feature-difference based)
    2. Dual-model embedding generation (TULIP/CLIP + SigLIP2)
    3. Vector store insertion
    """

    def __init__(
        self,
        embedding_engine: Any = None,
        milvus_store: Any = None,
        ocr_enabled: Optional[bool] = None,
        asr_enabled: Optional[bool] = None,
    ):
        self.keyframe_extractor = KeyframeExtractor()
        self.embedding_engine = embedding_engine
        self.milvus_store = milvus_store
        self.ocr_enabled = ocr_enabled if ocr_enabled is not None else settings.OCR_ENABLED
        self.asr_enabled = asr_enabled if asr_enabled is not None else settings.ASR_ENABLED
        logger.info(f"Pipeline initialized: OCR={self.ocr_enabled}, ASR={self.asr_enabled}")

    def process_video(
        self,
        video_path: str,
        video_id: Optional[str] = None,
        save_keyframes: bool = False,
        keyframe_output_dir: Optional[str] = None,
    ) -> dict:
        """Process a single video through the full pipeline."""
        vid = video_id or Path(video_path).stem
        logger.info(f"Processing: {vid} ({video_path})")
        start_time = time.time()

        # Step 1: Extract keyframes
        kf_dir = keyframe_output_dir or (str(settings.KEYFRAME_DIR / vid) if save_keyframes else None)
        keyframes = self.keyframe_extractor.extract_from_video(
            video_path=video_path,
            output_dir=kf_dir,
            save_keyframes=save_keyframes,
        )

        if not keyframes:
            logger.warning(f"No keyframes extracted from {vid}")
            return {"video_id": vid, "keyframes": 0, "status": "no_keyframes"}

        # Step 2: Load keyframe images for embedding
        import cv2
        images = []
        for kf in keyframes:
            if kf.get("path"):
                images.append(Image.open(kf["path"]).convert("RGB"))
            else:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, kf["frame_idx"])
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images.append(Image.fromarray(frame_rgb))
                else:
                    images.append(Image.new("RGB", (224, 224)))

        # Step 3: Generate embeddings
        engine = self.embedding_engine
        if engine is None:
            from embedding.hybrid_engine import HybridEmbeddingEngine
            engine = HybridEmbeddingEngine()
        logger.info(f"Generating embeddings for {len(keyframes)} keyframes...")
        hybrid_embs = engine.encode_images_batch(images)

        # Step 4: Insert into store
        store = self.milvus_store
        if store is None:
            from retrieval.local_store import LocalVectorStore
            store = LocalVectorStore()
        logger.info(f"Inserting {len(keyframes)} frames...")

        frame_indices = [kf["frame_idx"] for kf in keyframes]
        timestamps = [kf["timestamp"] for kf in keyframes]
        metadata_list = [{} for _ in keyframes]
        store.insert_frames(
            video_id=vid, frame_indices=frame_indices, timestamps=timestamps,
            clip_embeddings=hybrid_embs["clip"], siglip2_embeddings=hybrid_embs["siglip2"],
            metadata_list=metadata_list,
        )

        elapsed = time.time() - start_time
        logger.info(f"Done: {vid} — {len(keyframes)} keyframes in {elapsed:.1f}s")
        return {
            "video_id": vid, "video_path": video_path,
            "keyframes": len(keyframes), "embeddings": len(keyframes) * 2,
            "ocr_enabled": False, "asr_enabled": False,
            "processing_time": elapsed, "status": "success",
        }

    def process_batch(self, video_paths: List[str], save_keyframes: bool = False, max_workers: int = 4) -> List[dict]:
        results = []
        for vp in video_paths:
            try:
                results.append(self.process_video(video_path=vp, save_keyframes=save_keyframes))
            except Exception as e:
                logger.error(f"Failed: {vp}: {e}")
                results.append({"video_path": vp, "status": "error", "error": str(e)})
        return results

    def cleanup(self):
        if self.milvus_store and hasattr(self.milvus_store, 'close'):
            self.milvus_store.close()
        logger.info("Pipeline cleanup complete")