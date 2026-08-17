"""
Milvus Vector Database Integration for storing and retrieving video frame embeddings.
Configured with HNSW indexing for high-recall approximate nearest-neighbor search.
"""

import asyncio
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
from loguru import logger
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
)

from config import settings

class MilvusVectorStore:
    """
    Milvus vector store for storing CLIP and SigLIP2 embeddings
    in separate partitions within the same collection.
    """

    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.dim = settings.UNIFIED_DIM
        self.collection = None

        self._connect()
        self._init_collection()

    def _connect(self):
        """Establish connection to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _init_collection(self):
        """Initialize or load the Milvus collection with HNSW index."""
        if utility.has_collection(self.collection_name):
            col = Collection(self.collection_name)
            # Kiểm tra xem dimension trong Milvus có khớp với settings.UNIFIED_DIM không
            existing_dim = None
            for field in col.schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    existing_dim = field.params.get("dim")
                    break
            
            if existing_dim and existing_dim != self.dim:
                logger.warning(
                    f"Collection dimension mismatch! (Milvus stored: {existing_dim}, Config expected: {self.dim}). Recreating collection..."
                )
                utility.drop_collection(self.collection_name)
                self._create_collection()
            else:
                self.collection = col
                self.collection.load()
                logger.info(f"Loaded existing collection: {self.collection_name}")
        else:
            self._create_collection()
            
        self._ensure_index()

    def _create_collection(self):
        """Create a new Milvus collection with the appropriate schema."""
        # Define schema fields
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Auto-generated primary key",
            ),
            FieldSchema(
                name="video_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Source video identifier",
            ),
            FieldSchema(
                name="frame_idx",
                dtype=DataType.INT64,
                description="Frame index within the video",
            ),
            FieldSchema(
                name="timestamp",
                dtype=DataType.FLOAT,
                description="Timestamp in seconds",
            ),
            FieldSchema(
                name="clip_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
                description="CLIP embedding vector (1024-dim)",
            ),
            FieldSchema(
                name="siglip2_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
                description="SigLIP2 projected embedding vector (1024-dim)",
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="Optional metadata (OCR text, ASR transcript, etc.)",
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=" Video Frame Embeddings (CLIP + SigLIP2)",
        )

        self.collection = Collection(name=self.collection_name, schema=schema)
        logger.info(f"Created collection: {self.collection_name}")

    def _ensure_index(self):
        """Create HNSW indexes for both embedding fields if they don't exist."""
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": settings.MILVUS_M,
                "efConstruction": settings.MILVUS_EF_CONSTRUCTION,
            },
        }

        # Check and create CLIP embedding index
        clip_index_exists = any(
            idx.field_name == "clip_embedding"
            for idx in self.collection.indexes
        )
        if not clip_index_exists:
            self.collection.create_index(
                field_name="clip_embedding",
                index_params=index_params,
            )
            logger.info("Created HNSW index on clip_embedding")

        # Check and create SigLIP2 embedding index
        siglip2_index_exists = any(
            idx.field_name == "siglip2_embedding"
            for idx in self.collection.indexes
        )
        if not siglip2_index_exists:
            self.collection.create_index(
                field_name="siglip2_embedding",
                index_params=index_params,
            )
            logger.info("Created HNSW index on siglip2_embedding")

        self.collection.load()

    def insert_frames(
        self,
        video_id: str,
        frame_indices: List[int],
        timestamps: List[float],
        clip_embeddings: np.ndarray,
        siglip2_embeddings: np.ndarray,
        metadata_list: Optional[List[dict]] = None,
    ) -> List[int]:
        """
        Insert multiple frame embeddings into Milvus.

        Args:
            video_id: Source video identifier
            frame_indices: List of frame indices
            timestamps: List of timestamps in seconds
            clip_embeddings: (N, 1024) CLIP embeddings
            siglip2_embeddings: (N, 1024) SigLIP2 embeddings
            metadata_list: Optional list of metadata dicts (OCR, ASR, etc.)

        Returns:
            List of inserted primary key IDs
        """
        n = len(frame_indices)
        if metadata_list is None:
            metadata_list = [{}] * n

        entities = [
            [video_id] * n,  # video_id
            frame_indices,  # frame_idx
            timestamps,  # timestamp
            clip_embeddings.tolist(),  # clip_embedding
            siglip2_embeddings.tolist(),  # siglip2_embedding
            metadata_list,  # metadata
        ]

        try:
            insert_result = self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"Inserted {n} frames for video '{video_id}'")
            return insert_result.primary_keys
        except MilvusException as e:
            logger.error(f"Failed to insert frames: {e}")
            raise

    def search_clip(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        expr: Optional[str] = None,
    ) -> List[dict]:
        """
        Search using CLIP embedding.

        Args:
            query_embedding: (1024,) query vector from CLIP
            top_k: Number of top results
            expr: Optional boolean expression for filtering

        Returns:
            List of result dicts with id, video_id, frame_idx, timestamp, distance, metadata
        """
        return self._search(
            anns_field="clip_embedding",
            query_embedding=query_embedding,
            top_k=top_k,
            expr=expr,
        )

    def search_siglip2(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        expr: Optional[str] = None,
    ) -> List[dict]:
        """
        Search using SigLIP2 embedding.

        Args:
            query_embedding: (1024,) query vector from SigLIP2
            top_k: Number of top results
            expr: Optional boolean expression for filtering

        Returns:
            List of result dicts with id, video_id, frame_idx, timestamp, distance, metadata
        """
        return self._search(
            anns_field="siglip2_embedding",
            query_embedding=query_embedding,
            top_k=top_k,
            expr=expr,
        )

    def _search(
        self,
        anns_field: str,
        query_embedding: np.ndarray,
        top_k: int,
        expr: Optional[str] = None,
    ) -> List[dict]:
        """Internal search method."""
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": max(64, top_k * 2)},
        }

        output_fields = [
            "video_id",
            "frame_idx",
            "timestamp",
            "metadata",
        ]

        try:
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field=anns_field,
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=output_fields,
            )

            if not results or len(results) == 0:
                return []

            # Format results
            formatted = []
            for hit in results[0]:
                formatted.append({
                    "id": hit.id,
                    "video_id": hit.entity.get("video_id", ""),
                    "frame_idx": hit.entity.get("frame_idx", 0),
                    "timestamp": hit.entity.get("timestamp", 0.0),
                    # IMPORTANT: with metric_type="COSINE", Milvus returns the
                    # raw cosine SIMILARITY in `hit.distance` (higher = more
                    # similar) -- NOT a 0..2 "cosine distance" the way L2/IP
                    # metrics' `distance` field works. `1.0 - hit.distance`
                    # was inverting every score (best matches scored lowest,
                    # worst matches scored highest), which only mattered once
                    # downstream RRF/CombMNZ re-sorted candidates by this
                    # value -- Milvus's own internal search order was fine,
                    # but re-ranking on the inverted score scrambled it.
                    "distance": hit.distance,
                    "score": float(hit.distance),  # cosine similarity, higher = better
                    "metadata": hit.entity.get("metadata", {}),
                })
            return formatted

        except MilvusException as e:
            logger.error(f"Search failed: {e}")
            return []

    def hybrid_search(
        self,
        clip_query: np.ndarray,
        siglip2_query: np.ndarray,
        top_k: int = 10,
        expr: Optional[str] = None,
    ) -> List[dict]:
        """
        Perform hybrid search: query both CLIP and SigLIP2 embeddings,
        then return candidates with scores from both models.

        Args:
            clip_query: (1024,) CLIP query embedding
            siglip2_query: (1024,) SigLIP2 query embedding
            top_k: Number of top results per model
            expr: Optional boolean expression for filtering

        Returns:
            List of candidate dicts with clip_score and siglip2_score
        """
        clip_results = self.search_clip(clip_query, top_k=top_k, expr=expr)
        siglip2_results = self.search_siglip2(siglip2_query, top_k=top_k, expr=expr)

        # Merge results by frame key (video_id + frame_idx)
        merged = {}
        for r in clip_results:
            key = f"{r['video_id']}_{r['frame_idx']}"
            merged[key] = {
                "id": r["id"],
                "video_id": r["video_id"],
                "frame_idx": r["frame_idx"],
                "timestamp": r["timestamp"],
                "clip_score": r["score"],
                "siglip2_score": 0.0,
                "metadata": r["metadata"],
            }

        for r in siglip2_results:
            key = f"{r['video_id']}_{r['frame_idx']}"
            if key in merged:
                merged[key]["siglip2_score"] = r["score"]
            else:
                merged[key] = {
                    "id": r["id"],
                    "video_id": r["video_id"],
                    "frame_idx": r["frame_idx"],
                    "timestamp": r["timestamp"],
                    "clip_score": 0.0,
                    "siglip2_score": r["score"],
                    "metadata": r["metadata"],
                }

        return list(merged.values())

    def get_collection_stats(self) -> dict:
        """Get collection statistics."""
        if self.collection is None:
            return {"num_entities": 0}
        return {
            "name": self.collection_name,
            "num_entities": self.collection.num_entities,
        }

    def delete_by_video(self, video_id: str) -> bool:
        """Delete all frames belonging to a specific video."""
        try:
            expr = f'video_id == "{video_id}"'
            self.collection.delete(expr)
            self.collection.flush()
            logger.info(f"Deleted frames for video '{video_id}'")
            return True
        except MilvusException as e:
            logger.error(f"Failed to delete video frames: {e}")
            return False

    def close(self):
        """Release Milvus resources."""
        if self.collection:
            self.collection.release()
        connections.disconnect("default")
        logger.info("Milvus connection closed")