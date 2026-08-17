"""
Retrieval package. Imports are best-effort: a missing optional dependency
(e.g. pymilvus if you're only using LocalVectorStore, or sentence-transformers
if you're not running the cross-encoder yet) should not break importing the
whole package -- only that specific symbol becomes unavailable (None).
"""

from .reranker import HybridReranker
from .local_store import LocalVectorStore
from .bm25_search import BM25Index

try:
    from .milvus_store import MilvusVectorStore
except ImportError:
    MilvusVectorStore = None  # pymilvus not installed / not needed for this run

try:
    from .redis_cache import RedisCacheManager
except ImportError:
    RedisCacheManager = None  # redis not installed / not needed for this run

try:
    from .cross_encoder_reranker import CrossEncoderReranker
except ImportError:
    CrossEncoderReranker = None  # sentence-transformers not installed yet

from .hybrid_fusion import HybridFusionPipeline

__all__ = [
    "HybridReranker",
    "LocalVectorStore",
    "MilvusVectorStore",
    "RedisCacheManager",
    "BM25Index",
    "CrossEncoderReranker",
    "HybridFusionPipeline",
]
