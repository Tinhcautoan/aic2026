"""
Milvus Vector Database Integration (Milvus Lite).

Stores CLIP/TULIP + SigLIP2 embeddings in a single collection using the
file-based Milvus Lite backend (no Docker, no server, no port 19530).

Uses the `MilvusClient` API because Milvus Lite does NOT support the
ORM-style `Collection` / `connections` API.
"""

from typing import List, Optional
import numpy as np
from loguru import logger
from pymilvus import MilvusClient, DataType

from config import settings


class MilvusVectorStore:
    """Milvus Lite vector store for CLIP and SigLIP2 frame embeddings."""

    def __init__(self):
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.dim = settings.UNIFIED_DIM
        self.client = None

        self._init_client()
        self._init_collection()

    def _init_client(self):
        """Open the Milvus Lite file-based client."""
        uri = str(settings.MILVUS_DB_PATH)
        logger.info(f"Initializing Milvus Lite at {uri}")
        self.client = MilvusClient(uri=uri)

    def _init_collection(self):
        """Create the collection if missing; recreate on dimension mismatch."""
        if self.client.has_collection(self.collection_name):
            existing_dim = self._get_existing_dim()
            if existing_dim and existing_dim != self.dim:
                logger.warning(
                    f"Collection dimension mismatch (stored {existing_dim}, "
                    f"expected {self.dim}); recreating collection..."
                )
                self.client.drop_collection(self.collection_name)
                self._create_collection()
            else:
                logger.info(f"Loaded existing collection: {self.collection_name}")
        else:
            self._create_collection()

        # Create the vector indexes if missing — search fails with
        # "No index found" otherwise. (Milvus Lite may ignore index_params
        # passed to create_collection, so create indexes explicitly here.)
        self._ensure_index()

        # MilvusClient does NOT auto-load a collection after create/reopen — a
        # fresh collection is in 'released' state and search would fail with
        # "call load() before search". Load it into memory here.
        self.client.load_collection(self.collection_name)

    def _get_existing_dim(self) -> Optional[int]:
        try:
            desc = self.client.describe_collection(self.collection_name)
            for f in desc.get("fields", []):
                dim = (f.get("params") or {}).get("dim")
                if dim:
                    return int(dim)
        except Exception as e:
            logger.warning(f"Could not inspect collection schema: {e}")
        return None

    def _build_index_params(self, fields, index_type):
        ip = self.client.prepare_index_params()
        for field in fields:
            kwargs = dict(
                field_name=field,
                index_type=index_type,
                metric_type=settings.MILVUS_METRIC_TYPE,
            )
            if index_type == "HNSW":
                kwargs["params"] = {
                    "M": settings.MILVUS_M,
                    "efConstruction": settings.MILVUS_EF_CONSTRUCTION,
                }
            ip.add_index(**kwargs)
        return ip

    def _create_collection(self):
        """Create the collection schema + index at creation time (most reliable in Milvus Lite)."""
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("video_id", DataType.VARCHAR, max_length=256)
        schema.add_field("frame_idx", DataType.INT64)
        schema.add_field("timestamp", DataType.FLOAT)
        schema.add_field("clip_embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("siglip2_embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("metadata", DataType.JSON)

        index_params = self._build_index_params(
            ["clip_embedding", "siglip2_embedding"], settings.MILVUS_INDEX_TYPE,
        )
        self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)
        logger.info(f"Created collection: {self.collection_name}")

    def _ensure_index(self):
        """Create vector indexes if missing (Milvus search requires an index).

        Each attempt is VERIFIED via list_indexes because Milvus Lite can
        silently ignore index creation. Falls back HNSW -> AUTOINDEX -> FLAT.
        """
        def _indexed():
            try:
                return set(self.client.list_indexes(self.collection_name))
            except Exception:
                return set()

        missing = [f for f in ("clip_embedding", "siglip2_embedding") if f not in _indexed()]
        if not missing:
            return

        for index_type in (settings.MILVUS_INDEX_TYPE, "AUTOINDEX", "FLAT"):
            index_params = self._build_index_params(missing, index_type)
            try:
                self.client.create_index(self.collection_name, index_params=index_params)
                if all(f in _indexed() for f in missing):
                    logger.info(f"Created {index_type} indexes on {missing} for {self.collection_name}")
                    return
                logger.warning(f"{index_type} index creation not verified; trying next")
            except Exception as e:
                logger.warning(f"{index_type} index creation failed: {e}")

        logger.error(
            f"Could not create any index on {self.collection_name}. "
            f"Delete the milvus.db file and restart so the collection is recreated "
            f"fresh (index is created before data insertion)."
        )

    def insert_frames(
        self,
        video_id: str,
        frame_indices: List[int],
        timestamps: List[float],
        clip_embeddings: np.ndarray,
        siglip2_embeddings: np.ndarray,
        metadata_list: Optional[List[dict]] = None,
    ) -> List[int]:
        """Insert frame embeddings into Milvus Lite."""
        n = len(frame_indices)
        if n == 0:
            return []
        if metadata_list is None:
            metadata_list = [{} for _ in range(n)]

        clip_list = (
            clip_embeddings.tolist()
            if hasattr(clip_embeddings, "tolist") else list(clip_embeddings)
        )
        sig_list = (
            siglip2_embeddings.tolist()
            if hasattr(siglip2_embeddings, "tolist") else list(siglip2_embeddings)
        )

        data = [
            {
                "video_id": video_id,
                "frame_idx": int(frame_indices[i]),
                "timestamp": float(timestamps[i]),
                "clip_embedding": clip_list[i],
                "siglip2_embedding": sig_list[i],
                "metadata": metadata_list[i] or {},
            }
            for i in range(n)
        ]

        try:
            res = self.client.insert(self.collection_name, data=data)
            self.client.flush(self.collection_name)  # seal segments so search sees the new rows
            ids = res.get("ids", []) if isinstance(res, dict) else []
            logger.info(f"Inserted {n} frames for video '{video_id}'")
            return list(ids)
        except Exception as e:
            logger.error(f"Failed to insert frames: {e}")
            raise

    def search_clip(self, query_embedding, top_k=10, expr=None):
        return self._search("clip_embedding", query_embedding, top_k, expr)

    def search_siglip2(self, query_embedding, top_k=10, expr=None):
        return self._search("siglip2_embedding", query_embedding, top_k, expr)

    def _search(self, anns_field, query_embedding, top_k, expr=None):
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[np.asarray(query_embedding, dtype=np.float32).tolist()],
                anns_field=anns_field,
                limit=top_k,
                filter=expr or "",
                output_fields=["video_id", "frame_idx", "timestamp", "metadata"],
                search_params={
                    "metric_type": settings.MILVUS_METRIC_TYPE,
                    "params": {"ef": max(64, top_k * 2)},
                },
            )
            if not results:
                return []
            formatted = []
            for hit in results[0]:
                ent = hit.get("entity", {}) or {}
                formatted.append({
                    "id": hit.get("id"),
                    "video_id": ent.get("video_id", ""),
                    "frame_idx": ent.get("frame_idx", 0),
                    "timestamp": ent.get("timestamp", 0.0),
                    "distance": hit.get("distance"),
                    "score": float(hit.get("distance", 0.0)),
                    "metadata": ent.get("metadata", {}),
                })
            return formatted
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def hybrid_search(self, clip_query, siglip2_query, top_k=10, expr=None):
        clip_results = self.search_clip(clip_query, top_k=top_k, expr=expr)
        siglip2_results = self.search_siglip2(siglip2_query, top_k=top_k, expr=expr)

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

    def is_empty(self) -> bool:
        """True if the collection has no entities yet."""
        return self.get_collection_stats().get("num_entities", 0) == 0

    def ingest_from_persistent_store(self) -> int:
        """Migrate embeddings from the old pickle-based PersistentVectorStore.

        Reads ``data/persistent_store/{metadata.pkl, embeddings.pkl}`` and
        inserts the already-computed embeddings into Milvus Lite (no
        re-embedding required). Returns the number of frames inserted.
        """
        import pickle

        ps_dir = settings.DATA_DIR / "persistent_store"
        meta_file = ps_dir / "metadata.pkl"
        emb_file = ps_dir / "embeddings.pkl"
        if not (meta_file.exists() and emb_file.exists()):
            return 0

        logger.info("Migrating PersistentVectorStore -> Milvus Lite ...")
        with open(meta_file, "rb") as f:
            meta_data = pickle.load(f)
        with open(emb_file, "rb") as f:
            emb_data = pickle.load(f)

        ids = meta_data.get("ids", [])
        metadata = meta_data.get("metadata", {})
        clip_embs = emb_data.get("clip", [])
        siglip2_embs = emb_data.get("siglip2", [])

        by_video = {}
        for i, fid in enumerate(ids):
            m = metadata.get(fid, {}) or {}
            vid = m.get("video_id", "unknown")
            fi = m.get("frame_idx", 0)
            ts = m.get("timestamp", 0.0)
            extra = {k: v for k, v in m.items()
                     if k not in ("video_id", "frame_idx", "timestamp")}
            by_video.setdefault(vid, []).append(
                (fi, ts, clip_embs[i], siglip2_embs[i], extra)
            )

        total = 0
        for vid, frames in by_video.items():
            frames.sort(key=lambda x: x[0])
            try:
                self.insert_frames(
                    video_id=vid,
                    frame_indices=[f[0] for f in frames],
                    timestamps=[f[1] for f in frames],
                    clip_embeddings=np.array([f[2] for f in frames], dtype=np.float32),
                    siglip2_embeddings=np.array([f[3] for f in frames], dtype=np.float32),
                    metadata_list=[f[4] or {} for f in frames],
                )
                total += len(frames)
            except Exception as e:
                logger.warning(f"Failed to migrate video '{vid}': {e}")

        logger.info(f"Migrated {total} frames from PersistentVectorStore")
        return total

    def get_frame_embeddings(self, video_id: str, frame_indices: Optional[List[int]] = None) -> List[dict]:
        """Return stored embeddings for specific frames of a video.

        Used by TRAKE alignment to score candidate frames against event text.
        Returns list of dicts with 'frame_idx', 'clip_emb', 'siglip2_emb'.
        """
        filter_expr = f'video_id == "{video_id}"'
        if frame_indices is not None:
            if not frame_indices:
                return []
            idxs = ", ".join(str(int(i)) for i in frame_indices)
            filter_expr += f" and frame_idx in [{idxs}]"

        try:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["frame_idx", "clip_embedding", "siglip2_embedding"],
            )
        except Exception as e:
            logger.error(f"get_frame_embeddings query failed: {e}")
            return []

        results = []
        for r in rows:
            results.append({
                "frame_idx": r.get("frame_idx", 0),
                "clip_emb": np.asarray(r.get("clip_embedding", []), dtype=np.float32),
                "siglip2_emb": np.asarray(r.get("siglip2_embedding", []), dtype=np.float32),
            })
        return results

    def get_collection_stats(self) -> dict:
        if self.client is None:
            return {"num_entities": 0}
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "name": self.collection_name,
                "num_entities": stats.get("row_count", 0),
            }
        except Exception as e:
            logger.warning(f"get_collection_stats failed: {e}")
            return {"name": self.collection_name, "num_entities": 0}

    def delete_by_video(self, video_id: str) -> bool:
        try:
            self.client.delete(self.collection_name, filter=f'video_id == "{video_id}"')
            logger.info(f"Deleted frames for video '{video_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete video frames: {e}")
            return False

    def close(self):
        if self.client:
            self.client.close()
        logger.info("Milvus Lite connection closed")

