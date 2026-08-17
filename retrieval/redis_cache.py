"""
Redis Cache Manager for high-speed caching of query embeddings, 
partial results, and user session states.
Provides ~30% I/O reduction on consecutive queries as described in the paper.
"""

import json
import hashlib
from typing import Optional, Any, List, Dict
import numpy as np
from loguru import logger
import redis.asyncio as aioredis

from config import settings

class RedisCacheManager:
    """
    Redis-based caching layer for:
    - Query embeddings (to avoid recomputation)
    - Top-N retrieval results
    - User session states
    - Temporary intermediate results
    """

    def __init__(self):
        self.host = settings.REDIS_HOST
        self.port = settings.REDIS_PORT
        self.db = settings.REDIS_DB
        self.password = settings.REDIS_PASSWORD or None
        self.ttl = settings.REDIS_CACHE_TTL
        self.client: Optional[aioredis.Redis] = None

        self._key_prefix = "aic2025"

    async def connect(self):
        """Establish async connection to Redis."""
        self.client = aioredis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=False,  # Keep bytes for numpy arrays
        )
        await self.client.ping()
        logger.info(f"Connected to Redis at {self.host}:{self.port}")

    async def disconnect(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")

    def _make_key(self, *parts: str) -> str:
        """Create a namespaced Redis key."""
        return f"{self._key_prefix}:{':'.join(parts)}"

    def _hash_query(self, query: str) -> str:
        """Create a deterministic hash for a text query."""
        return hashlib.md5(query.encode("utf-8")).hexdigest()[:16]

    # ---- Embedding Cache ----

    async def cache_embedding(
        self,
        query_type: str,  # e.g., "clip_text", "siglip2_text"
        query: str,
        embedding: np.ndarray,
        ttl: int = None,
    ):
        """Cache a query embedding."""
        query_hash = self._hash_query(query)
        key = self._make_key("emb", query_type, query_hash)

        # Store as bytes for efficiency
        embedding_bytes = embedding.astype(np.float32).tobytes()
        await self.client.set(key, embedding_bytes, ex=ttl or self.ttl)
        logger.debug(f"Cached embedding for '{query_type}:{query[:50]}...'")

    async def get_embedding(
        self,
        query_type: str,
        query: str,
    ) -> Optional[np.ndarray]:
        """Retrieve a cached query embedding."""
        query_hash = self._hash_query(query)
        key = self._make_key("emb", query_type, query_hash)

        cached = await self.client.get(key)
        if cached is None:
            return None

        embedding = np.frombuffer(cached, dtype=np.float32)
        logger.debug(f"Cache hit: embedding for '{query_type}:{query[:50]}...'")
        return embedding

    # ---- Result Cache ----

    async def cache_results(
        self,
        query: str,
        model: str,  # "clip", "siglip2", "fusion"
        top_k: int,
        results: List[dict],
        ttl: int = None,
    ):
        """Cache retrieval results."""
        query_hash = self._hash_query(query)
        key = self._make_key("results", model, str(top_k), query_hash)

        # Serialize results to JSON
        serialized = json.dumps(results, default=str)
        await self.client.set(key, serialized, ex=ttl or self.ttl)
        logger.debug(f"Cached {len(results)} results for '{model}:{query[:50]}...'")

    async def get_results(
        self,
        query: str,
        model: str,
        top_k: int,
    ) -> Optional[List[dict]]:
        """Retrieve cached results."""
        query_hash = self._hash_query(query)
        key = self._make_key("results", model, str(top_k), query_hash)

        cached = await self.client.get(key)
        if cached is None:
            return None

        results = json.loads(cached)
        logger.debug(f"Cache hit: {len(results)} results for '{model}:{query[:50]}...'")
        return results

    # ---- Session State ----

    async def set_session_state(
        self,
        user_id: str,
        state: dict,
        ttl: int = 7200,  # 2 hours default
    ):
        """Store user session state."""
        key = self._make_key("session", user_id)
        serialized = json.dumps(state, default=str)
        await self.client.set(key, serialized, ex=ttl)

    async def get_session_state(self, user_id: str) -> Optional[dict]:
        """Retrieve user session state."""
        key = self._make_key("session", user_id)
        cached = await self.client.get(key)
        if cached is None:
            return None
        return json.loads(cached)

    async def delete_session_state(self, user_id: str):
        """Remove user session state."""
        key = self._make_key("session", user_id)
        await self.client.delete(key)

    # ---- Utility ----

    async def clear_cache(self, pattern: str = "*"):
        """Clear all cache entries matching a pattern."""
        keys = await self.client.keys(self._make_key(pattern))
        if keys:
            await self.client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries matching '{pattern}'")

    async def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if not self.client:
            return {"status": "disconnected"}

        info = await self.client.info("stats")
        keys = await self.client.keys(self._make_key("*"))
        return {
            "total_keys": len(keys),
            "used_memory_human": info.get("used_memory_human", "N/A"),
            "connected_clients": info.get("connected_clients", 0),
            "uptime_in_seconds": info.get("uptime_in_seconds", 0),
        }
