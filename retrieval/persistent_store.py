"""
File-based persistent vector store for local mode.
Saves keyframes and embeddings to disk using pickle.
Survives server restarts — no need to re-process videos.
"""

import pickle
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from config import settings

class PersistentVectorStore:
    """
    Local file-based vector store with persistence.
    Stores embeddings and metadata on disk in pickle format.
    """

    def __init__(self, storage_dir: str = None):
        self.storage_dir = Path(storage_dir or settings.PROJECT_ROOT / "data" / "persistent_store")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._metadata_file = self.storage_dir / "metadata.pkl"
        self._embeddings_file = self.storage_dir / "embeddings.pkl"
        
        self._ids = []
        self._ids_set = set()
        self._embeddings_clip = []
        self._embeddings_siglip2 = []
        self._metadata = {}  # {id: {video_id, frame_idx, timestamp}}
        
        self.name = "persistent_store"
        self._load_from_disk()
        
        if self._ids:
            logger.info(f"Loaded {len(self._ids)} frames from persistent storage")

    def _load_from_disk(self):
        """Load existing data from disk."""
        try:
            if self._metadata_file.exists() and self._embeddings_file.exists():
                with open(self._metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self._ids = data.get('ids', [])
                    self._ids_set = set(self._ids)
                    self._metadata = data.get('metadata', {})
                
                with open(self._embeddings_file, 'rb') as f:
                    emb_data = pickle.load(f)
                    self._embeddings_clip = emb_data.get('clip', [])
                    self._embeddings_siglip2 = emb_data.get('siglip2', [])

                if settings.USE_C2LIP:
                    self._embeddings_siglip2 = self._load_c2lip_embeddings(
                        fallback=self._embeddings_siglip2, expected_len=len(self._ids)
                    )

                logger.info(f"Restored {len(self._ids)} frames from {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to load persistent store: {e}")
            self._ids = []
            self._metadata = {}
            self._embeddings_clip = []
            self._embeddings_siglip2 = []

    def _load_c2lip_embeddings(self, fallback: list, expected_len: int) -> list:
        """Precomputed C2LIP vectors for the frames already in this store,
        positionally aligned to self._ids (verified against this store's own
        siglip2 vectors before this feature was wired up -- same source, same
        order). Falls back to the original siglip2 vectors on any mismatch so
        a stale/incomplete pkl can't silently misalign the corpus."""
        path = settings.C2LIP_EMBEDDINGS_PKL
        c2lip_embs = None
        last_err = None
        # This is a ~1GB read and this machine's disk has shown transient
        # "[Errno 22] Invalid argument" failures under memory pressure right
        # after loading multiple GB-scale models -- retry a few times before
        # accepting the (silently wrong) fallback.
        for attempt in range(5):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                c2lip_embs = data.get('c2lip')
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(2)
        if last_err is not None:
            logger.error(f"USE_C2LIP is True but {path!r} could not be loaded after 5 "
                         f"attempts ({last_err}); falling back to siglip2 vectors from "
                         f"persistent_store.")
            return fallback

        if c2lip_embs is None or len(c2lip_embs) != expected_len:
            logger.error(
                f"USE_C2LIP is True but {path!r} has "
                f"{'no' if c2lip_embs is None else len(c2lip_embs)} 'c2lip' vectors, "
                f"expected {expected_len} (matching this store's ids). "
                f"Falling back to siglip2 vectors from persistent_store."
            )
            return fallback

        logger.info(f"Using {len(c2lip_embs)} precomputed C2LIP vectors from {path!r} "
                    f"for the siglip2 slot.")
        return c2lip_embs

    def _save_to_disk(self):
        """Save current data to disk."""
        try:
            # Save metadata
            with open(self._metadata_file, 'wb') as f:
                pickle.dump({
                    'ids': self._ids,
                    'metadata': self._metadata
                }, f)
            
            # Save embeddings
            with open(self._embeddings_file, 'wb') as f:
                pickle.dump({
                    'clip': self._embeddings_clip,
                    'siglip2': self._embeddings_siglip2
                }, f)
            
            logger.debug(f"Saved {len(self._ids)} frames to persistent storage")
        except Exception as e:
            logger.error(f"Failed to save persistent store: {e}")

    def insert_frames(
        self,
        video_id: str,
        frame_indices: List[int],
        clip_embeddings: np.ndarray,
        siglip2_embeddings: np.ndarray,
        timestamps: List[float],
        metadata_list: List[Dict] = None,
    ):
        """Insert keyframes with embeddings."""
        if len(frame_indices) == 0:
            return
        
        if metadata_list is None:
            metadata_list = [{} for _ in frame_indices]
        
        for frame_idx, clip_emb, siglip2_emb, ts, meta in zip(
            frame_indices, clip_embeddings, siglip2_embeddings, timestamps, metadata_list
        ):
            frame_id = f"{video_id}:{frame_idx}"
            
            # Skip if already exists
            if frame_id in self._ids_set:
                logger.warning(f"Frame {frame_id} already exists, skipping")
                continue
            
            self._ids.append(frame_id)
            self._ids_set.add(frame_id)
            self._metadata[frame_id] = {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "timestamp": float(ts),
                **meta  # Merge additional metadata
            }
            self._embeddings_clip.append(clip_emb)
            self._embeddings_siglip2.append(siglip2_emb)
        
        # Save after each insert
        self._save_to_disk()

    def search_clip(self, query_emb: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Search using CLIP embeddings."""
        if not self._embeddings_clip:
            return []
        
        # Compute similarities
        similarities = np.dot(self._embeddings_clip, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            frame_id = self._ids[idx]
            meta = self._metadata[frame_id]
            results.append({
                "id": frame_id,
                "video_id": meta["video_id"],
                "frame_idx": meta["frame_idx"],
                "timestamp": meta["timestamp"],
                "clip_score": float(similarities[idx]),
            })
        return results

    def search_siglip2(self, query_emb: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Search using SigLIP2 embeddings."""
        if not self._embeddings_siglip2:
            return []
        
        similarities = np.dot(self._embeddings_siglip2, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            frame_id = self._ids[idx]
            meta = self._metadata[frame_id]
            results.append({
                "id": frame_id,
                "video_id": meta["video_id"],
                "frame_idx": meta["frame_idx"],
                "timestamp": meta["timestamp"],
                "siglip2_score": float(similarities[idx]),
            })
        return results

    def hybrid_search(
        self,
        clip_query: np.ndarray,
        siglip2_query: np.ndarray,
        top_k: int = 20,
        expr: Optional[str] = None,
    ) -> List[Dict]:
        """
        Hybrid search: average CLIP + SigLIP2 scores.
        If expr is provided (e.g., 'video_id == "video123"'), filter results.
        """
        if not self._embeddings_clip or not self._embeddings_siglip2:
            return []
        
        clip_sims = np.dot(self._embeddings_clip, clip_query)
        siglip2_sims = np.dot(self._embeddings_siglip2, siglip2_query)
        
        # Average scores
        avg_sims = (clip_sims + siglip2_sims) / 2.0
        top_indices = np.argsort(avg_sims)[::-1]
        
        results = []
        for idx in top_indices:
            frame_id = self._ids[idx]
            meta = self._metadata[frame_id]
            
            # Apply expr filter if provided (simple string matching for video_id)
            if expr:
                # Parse expr like 'video_id == "video123"'
                if 'video_id' in expr and '==' in expr:
                    # Extract video_id from expr
                    parts = expr.split('==')
                    if len(parts) == 2:
                        target_video_id = parts[1].strip().strip('"').strip("'")
                        if meta['video_id'] != target_video_id:
                            continue
                else:
                    # If expr format is unexpected, skip filtering
                    pass
            
            results.append({
                "id": frame_id,
                "video_id": meta["video_id"],
                "frame_idx": meta["frame_idx"],
                "timestamp": meta["timestamp"],
                "clip_score": float(clip_sims[idx]),
                "siglip2_score": float(siglip2_sims[idx]),
                "final_score": float(avg_sims[idx]),
            })
            
            if len(results) >= top_k:
                break
        
        return results

    def get_frame_embeddings(self, video_id: str, frame_indices: List[int] = None) -> List[dict]:
        """Get embeddings for specific frames in a video.

        If frame_indices is None, returns all frames for the video.
        Returns list of dicts with 'frame_idx', 'clip_emb', 'siglip2_emb'.
        """
        results = []
        for i, frame_id in enumerate(self._ids):
            meta = self._metadata.get(frame_id, {})
            if meta.get("video_id") != video_id:
                continue
            fi = meta.get("frame_idx")
            if frame_indices is not None and fi not in frame_indices:
                continue
            results.append({
                "frame_idx": fi,
                "clip_emb": self._embeddings_clip[i],
                "siglip2_emb": self._embeddings_siglip2[i],
            })
        return results

    def get_collection_stats(self) -> Dict:
        """Return collection statistics."""
        return {
            "name": self.name,
            "num_entities": len(self._ids),
            "storage": str(self.storage_dir),
        }

    def close(self):
        """Cleanup."""
        self._save_to_disk()
        logger.info("Persistent store closed and saved")

    def clear(self):
        """Clear all data and delete files."""
        self._ids = []
        self._ids_set = set()
        self._metadata = {}
        self._embeddings_clip = []
        self._embeddings_siglip2 = []
        
        self._metadata_file.unlink(missing_ok=True)
        self._embeddings_file.unlink(missing_ok=True)
        logger.info("Persistent store cleared")
