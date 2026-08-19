"""
Configuration settings for the Video Retrieval System.
Supports environment variable overrides via pydantic-settings.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VIDEO_DIR: Path = DATA_DIR / "videos"
    KEYFRAME_DIR: Path = DATA_DIR / "keyframes"
    CACHE_DIR: Path = DATA_DIR / "cache"

    # Milvus Configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "video_frames"
    MILVUS_INDEX_TYPE: str = "HNSW"
    MILVUS_METRIC_TYPE: str = "COSINE"
    MILVUS_EF_CONSTRUCTION: int = 200
    MILVUS_M: int = 16

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_CACHE_TTL: int = 3600  # 1 hour

    # Embedding Configuration
    CLIP_MODEL_NAME: str = "ViT-L-14"
    CLIP_PRETRAINED: str = "dfn2b"
    SIGLIP2_MODEL_NAME: str = "ViT-SO400M-14-SigLIP2"
    SIGLIP2_PRETRAINED: str = "webli"
    TULIP_MODEL_NAME: str = "ViT-L-14"
    TULIP_PRETRAINED: str = "openai"  # TULIP builds on OpenAI CLIP + RoPE architecture from the fork
    TULIP_CHECKPOINT: str = ""  # Optional: path to downloaded TULIP fine-tuned weights
    USE_TULIP: bool = True  # Set to True to use TULIP instead of CLIP
    # TULIP/CLIP ViT-L-14 produces 768-dim features in the open_clip model used here.
    # SigLIP2 is projected down to the same unified dimension to keep both fields compatible
    # with the Milvus collection schema and avoid dimension-mismatch insert errors.
    EMBEDDING_DIM_CLIP: int = 768
    EMBEDDING_DIM_SIGLIP2: int = 1152
    UNIFIED_DIM: int = 768
    USE_FP16: bool = True

    # Vietnamese -> English translation (improves retrieval for Vietnamese
    # queries on the English-only TULIP/SigLIP2 text encoders; English queries
    # are passed through unchanged).
    VI_EN_TRANSLATION_ENABLED: bool = True
    VI_EN_TRANSLATION_MODEL: str = "vinai/vinai-translate-vi2en-v2"
    VI_EN_TRANSLATION_DEVICE: str = "cpu"  # keep translation off the small 4GB GPU
    VI_EN_TRANSLATION_NUM_BEAMS: int = 4  # beam search improves translation quality

    # Keyframe Extraction
    KEYFRAME_THRESHOLD: float = 0.6
    KEYFRAME_MIN_DELTA: int = 5  # minimum frames between keyframes
    FRAME_SAMPLE_RATE: int = 5  # sample every N frames
    IMAGE_SIZE: int = 224

    # Reranking Configuration
    RRF_K: int = 60
    FUSION_ALPHA: float = 0.5  # weight for RRF vs CombMNZ

    # BM25 Sparse Search Configuration
    BM25_INDEX_PATH: str = ""  # "" -> defaults to CACHE_DIR/bm25_index.pkl at runtime
    BM25_USE_VN_SEGMENTATION: bool = False  # True -> use underthesea word segmentation

    # Cross-Encoder Re-ranker Configuration
    CROSS_ENCODER_ENABLED: bool = True
    CROSS_ENCODER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"  # multilingual, handles Vietnamese

    # Retrieval Funnel Top-K (Dense -> BM25 -> RRF Fusion -> Cross-Encoder -> Expansion)
    DENSE_TOP_K: int = 200
    SPARSE_TOP_K: int = 200
    FUSION_TOP_K: int = 50          # candidates surviving to cross-encoder stage
    CROSS_ENCODER_TOP_K: int = 50   # candidates surviving cross-encoder rerank
    FINAL_TOP_K: int = 100          # size of submission list (per AIC scoring: R@1..R@100)
    FRAME_EXPANSION_WINDOWS: list = [5]  # +-N frame neighbor expansion
    FRAME_EXPANSION_MAX_SOURCES: int = 3  # only expand from top-N real hits
    FRAME_EXPANSION_FOR_UI: bool = False  # allow expansion when final_top_k <= 20

    # Result deduplication (Stage 6)
    RESULT_DEDUP_ENABLED: bool = True
    RESULT_DEDUP_MIN_FRAME_GAP: int = 15  # suppress frames within N frames in same video
    RESULT_DEDUP_SCENE_WINDOW_SEC: float = 8.0  # max one hit per scene bucket per video

    # Retrieval Configuration
    TOP_K_DEFAULT: int = 10
    MAX_CONCURRENT_QUERIES: int = 8

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: list = ["*"]

    # OCR Configuration
    OCR_ENABLED: bool = True
    OCR_LANGUAGES: list = ["en", "vi"]

    # ASR Configuration
    ASR_ENABLED: bool = True
    ASR_MODEL_SIZE: str = "base"

    # LLM (Gemini) VQA Configuration
    GEMINI_API_KEY: str = ""            # Google AI Studio key; leave empty to use metadata-only VQA
    GEMINI_MODEL: str = "gemini-3.5-flash"
    GEMINI_ENABLED: bool = False        # set True in .env to use LLM answers
    GEMINI_TOP_K_FRAMES: int = 3        # evidence frames sent to the LLM
    GEMINI_TIMEOUT_SEC: float = 60.0    # per-request timeout

    # Self-hosted VLM Configuration (vLLM OpenAI-compatible)
    VLM_ENABLED: bool = False           # set True to use self-hosted VLM for Q&A
    VLM_URL: str = "http://localhost:8080"  # vLLM server base URL
    VLM_MODEL: str = "qwen2.5-vl-32b"  # model name passed to vLLM
    VLM_TIMEOUT_SEC: float = 120.0      # per-request timeout
    VLM_MAX_FRAMES: int = 30            # max frames sent to VLM per request

    # Contest Pipeline Settings
    TOPK_VIDEOS: int = 20               # candidates for Q&A (top-K videos to expand)
    TRAKE_VIDEOS: int = 50              # candidates for TRAKE (top-K videos for alignment)
    DELTA_RATIO: float = 0.05           # window = keyframe ± 5% of frame_count
    FRAME_STEP: int = 5                 # sample every N frames within delta window
    MAX_ROWS: int = 100                 # max CSV rows per query

    # GPU Configuration
    GPU_ENABLED: bool = True
    GPU_DEVICE_ID: int = 0

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
