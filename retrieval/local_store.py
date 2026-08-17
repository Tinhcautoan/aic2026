"""
Local In-Memory Vector Store — replaces Milvus when Docker is unavailable.
Uses numpy for cosine similarity search. Implements the same interface as MilvusVectorStore.
"""

import numpy as np
from typing import List, Optional
from loguru import logger
from config import settings

class LocalVectorStore:
    """
    In-memory vector store that mimics the MilvusVectorStore API.
    Uses numpy cosine similarity for search.
    """

    def __init__(self):
        self.dim = settings.UNIFIED_DIM
        self._ids = []
        self._video_ids = []
        self._frame_indices = []
        self._timestamps = []
        self._clip_embeddings = []
        self._siglip2_embeddings = []
        self._metadata = []
        self._next_id = 1
        logger.info("LocalVectorStore initialized (in-memory fallback)")

    def insert_frames(
        self,
        video_id: str,
        frame_indices: List[int],
        timestamps: List[float],
        clip_embeddings: np.ndarray,
        siglip2_embeddings: np.ndarray,
        metadata_list: Optional[List[dict]] = None,
    ) -> List[int]:
        n = len(frame_indices)
        if metadata_list is None:
            metadata_list = [{}] * n

        ids = list(range(self._next_id, self._next_id + n))
        self._next_id += n

        self._ids.extend(ids)
        self._video_ids.extend([video_id] * n)
        self._frame_indices.extend(frame_indices)
        self._timestamps.extend(timestamps)
        self._clip_embeddings.append(clip_embeddings)
        self._siglip2_embeddings.append(siglip2_embeddings)
        self._metadata.extend(metadata_list)

        logger.info(f"Inserted {n} frames for video '{video_id}' (local store)")
        return ids

    def _get_embedding_matrix(self, field: str = "clip") -> np.ndarray:
        """Build the full embedding matrix for search."""
        if field == "clip":
            if not self._clip_embeddings:
                return np.array([]).reshape(0, self.dim)
            return np.vstack(self._clip_embeddings).astype(np.float32)
        else:
            if not self._siglip2_embeddings:
                return np.array([]).reshape(0, self.dim)
            return np.vstack(self._siglip2_embeddings).astype(np.float32)

    def _cosine_search(self, query: np.ndarray, emb_matrix: np.ndarray, top_k: int) -> List[dict]:
        """Search using cosine similarity."""
        query = query.astype(np.float32).reshape(1, -1)
        # Normalize
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        emb_norm = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
        # Cosine similarity
        scores = np.dot(query_norm, emb_norm.T)[0]
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self._ids):
                results.append({
                    "id": self._ids[idx],
                    "video_id": self._video_ids[idx],
                    "frame_idx": self._frame_indices[idx],
                    "timestamp": self._timestamps[idx],
                    "distance": float(1.0 - scores[idx]),  # cosine distance
                    "score": float(scores[idx]),  # cosine similarity
                    "metadata": self._metadata[idx] if idx < len(self._metadata) else {},
                })
        return results

    def search_clip(self, query_embedding: np.ndarray, top_k: int = 10, expr: Optional[str] = None) -> List[dict]:
        emb_matrix = self._get_embedding_matrix("clip")
        if emb_matrix.shape[0] == 0:
            return []
        return self._cosine_search(query_embedding, emb_matrix, top_k)

    def search_siglip2(self, query_embedding: np.ndarray, top_k: int = 10, expr: Optional[str] = None) -> List[dict]:
        emb_matrix = self._get_embedding_matrix("siglip2")
        if emb_matrix.shape[0] == 0:
            return []
        return self._cosine_search(query_embedding, emb_matrix, top_k)

    def hybrid_search(self, clip_query: np.ndarray, siglip2_query: np.ndarray, top_k: int = 10, expr: Optional[str] = None) -> List[dict]:
        clip_results = self.search_clip(clip_query, top_k=top_k, expr=expr)
        siglip2_results = self.search_siglip2(siglip2_query, top_k=top_k, expr=expr)

        merged = {}
        for r in clip_results:
            key = f"{r['video_id']}_{r['frame_idx']}"
            merged[key] = {
                "id": r["id"], "video_id": r["video_id"], "frame_idx": r["frame_idx"],
                "timestamp": r["timestamp"], "clip_score": r["score"], "siglip2_score": 0.0,
                "metadata": r["metadata"],
            }
        for r in siglip2_results:
            key = f"{r['video_id']}_{r['frame_idx']}"
            if key in merged:
                merged[key]["siglip2_score"] = r["score"]
            else:
                merged[key] = {
                    "id": r["id"], "video_id": r["video_id"], "frame_idx": r["frame_idx"],
                    "timestamp": r["timestamp"], "clip_score": 0.0, "siglip2_score": r["score"],
                    "metadata": r["metadata"],
                }
        return list(merged.values())

    def get_collection_stats(self) -> dict:
        return {"name": "local_store", "num_entities": len(self._ids)}

    def delete_by_video(self, video_id: str) -> bool:
        # Remove all entries with matching video_id
        indices_to_keep = [i for i, vid in enumerate(self._video_ids) if vid != video_id]
        self._ids = [self._ids[i] for i in indices_to_keep]
        self._video_ids = [self._video_ids[i] for i in indices_to_keep]
        self._frame_indices = [self._frame_indices[i] for i in indices_to_keep]
        self._timestamps = [self._timestamps[i] for i in indices_to_keep]
        self._metadata = [self._metadata[i] for i in indices_to_keep]
        # Rebuild embedding matrices
        if self._clip_embeddings:
            new_clip = np.vstack(self._clip_embeddings)[indices_to_keep]
            self._clip_embeddings = [new_clip] if len(new_clip) > 0 else []
        if self._siglip2_embeddings:
            new_siglip = np.vstack(self._siglip2_embeddings)[indices_to_keep]
            self._siglip2_embeddings = [new_siglip] if len(new_siglip) > 0 else []
        return True

    def close(self):
        logger.info("LocalVectorStore closed")
