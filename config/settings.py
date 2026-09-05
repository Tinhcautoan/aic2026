"""
Configuration settings for the Video Retrieval System.
Supports environment variable overrides via pydantic-settings.

CHANGES IN THIS REVISION (see PATCH_NOTES.md)
  * SigLIP2 no longer truncated to 768 -- SIGLIP2_PROJECTION_MODE defaults
    to "native" (1152 dims). Requires a Milvus re-index.
  * USE_TULIP defaults to False: TULIP_PRETRAINED was "openai", i.e. the
    "TULIP" branch was loading OpenAI CLIP ViT-L/14 from 2021. DFN2B is
    strictly better until real TULIP or PE-Core weights are wired up.
  * New confidence/calibration block -- this is what fixes the
    "confidence is always ~0.03" symptom.
  * FUSION_TOP_K raised 50 -> 100 so ranks 51-100 also get re-ranked.
  * BM25_USE_VN_SEGMENTATION enabled; translation beam search raised to 4.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---------------- Paths ----------------
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VIDEO_DIR: Path = DATA_DIR / "videos"
    KEYFRAME_DIR: Path = DATA_DIR / "keyframes"
    CACHE_DIR: Path = DATA_DIR / "cache"
    MODELS_DIR: Path = PROJECT_ROOT / "models"

    # ---------------- Milvus ----------------
    MILVUS_DB_PATH: Path = PROJECT_ROOT / "milvus.db"
    MILVUS_COLLECTION_NAME: str = "video_frames"
    # AUTOINDEX silently ignores MILVUS_M / MILVUS_EF_CONSTRUCTION below.
    # Set HNSW here once you move off Milvus Lite to standalone.
    MILVUS_INDEX_TYPE: str = "AUTOINDEX"
    MILVUS_METRIC_TYPE: str = "COSINE"
    MILVUS_EF_CONSTRUCTION: int = 200
    MILVUS_M: int = 16

    # ---------------- Redis ----------------
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_CACHE_TTL: int = 3600

    # ---------------- Embedding ----------------
    CLIP_MODEL_NAME: str = "ViT-L-14"
    CLIP_PRETRAINED: str = "dfn2b"
    SIGLIP2_MODEL_NAME: str = "ViT-SO400M-14-SigLIP2"
    SIGLIP2_PRETRAINED: str = "webli"

    # C2LIP replaces the SigLIP2 branch (the "siglip2" slot / siglip2_embedding
    # field) when USE_C2LIP is True -- same swap pattern as USE_PECORE above,
    # but for the second encoder instead of the first. Verified: c2lipweight.pt
    # is a fine-tuned open_clip ViT-B-16-SigLIP checkpoint (0 missing/unexpected
    # keys), 768-dim output. Different machines can run different encoders in
    # this slot by toggling this flag in each machine's .env.
    USE_C2LIP: bool = False
    C2LIP_MODEL_NAME: str = "ViT-B-16-SigLIP"
    C2LIP_CHECKPOINT: str = str(PROJECT_ROOT / "c2lipweight.pt")
    EMBEDDING_DIM_C2LIP: int = 768
    # Precomputed corpus-side embeddings for the frames already sitting in
    # data/persistent_store (see embedding/__init__.py's ingest path). Only
    # used once, to bootstrap Milvus without re-running inference on frames
    # already extracted. New frames ingested after USE_C2LIP is on are
    # encoded live via C2LIPEncoder instead.
    C2LIP_EMBEDDINGS_PKL: str = str(
        PROJECT_ROOT / "embedding finvit + c2lip" / "embeddings_c2lip.pkl"
    )

    TULIP_MODEL_NAME: str = "ViT-L-14"
    # NOTE: "openai" loads OpenAI CLIP ViT-L/14 weights, NOT TULIP. Leaving
    # USE_TULIP on with an empty TULIP_CHECKPOINT means running a 2021 model
    # while believing you run TULIP. Set TULIP_CHECKPOINT to real weights
    # before flipping USE_TULIP back to True.
    TULIP_PRETRAINED: str = "openai"
    TULIP_CHECKPOINT: str = ""
    USE_TULIP: bool = False

    # PE-Core (Meta) replaces CLIP/TULIP as the primary "clip_embedding"
    # branch. Loaded via open_clip's hf-hub path (validated by
    # export_embeddings_from_keyframes.py, which produced the 1280-dim
    # vectors already sitting in embeddings_pecore.pkl).
    USE_PECORE: bool = True
    PECORE_MODEL_NAME: str = "hf-hub:timm/PE-Core-bigG-14-448"
    EMBEDDING_DIM_PECORE: int = 1280

    EMBEDDING_DIM_CLIP: int = 768
    EMBEDDING_DIM_SIGLIP2: int = 1152
    # Dim of whichever model is active in the clip_encoder slot (PE-Core by
    # default now -- 1280). Falls back to 768 only if both USE_PECORE and
    # USE_TULIP are off and plain CLIP/TULIP (768-dim) is active instead --
    # change this back to 768 in that case.
    UNIFIED_DIM: int = 1280

    # "native" keeps all 1152 dims (correct default -- use this for a fresh
    #          setup with no existing embeddings)
    # "pca"    genuine reduction, needs SIGLIP2_PCA_PATH
    # "truncate" reproduces the old lossy behaviour -- only useful to match
    #          embeddings computed before this setting existed, at the cost
    #          of silently discarding 384 of 1152 dimensions. Override via
    #          .env, don't change the default here.
    SIGLIP2_PROJECTION_MODE: str = "native"
    SIGLIP2_PCA_PATH: str = ""
    # Dimensionality actually stored for the SigLIP2 field. Must match the
    # encoder's output_dim; the store asserts this at startup.
    EMBEDDING_DIM_SIGLIP2_STORED: int = 1152

    @property
    def SIGLIP2_SLOT_DIM(self) -> int:
        """Effective dim of whichever model backs the "siglip2" slot right
        now. Vector stores should read this, not EMBEDDING_DIM_SIGLIP2_STORED
        directly, so they auto-adjust when USE_C2LIP is toggled."""
        return self.EMBEDDING_DIM_C2LIP if self.USE_C2LIP else self.EMBEDDING_DIM_SIGLIP2_STORED

    USE_FP16: bool = True
    OPENCLIP_MODEL_DIR: str = ""

    # ---------------- Scoring & confidence ----------------
    # Raw cosine is not a confidence. Each open_clip model ships the
    # temperature it was trained with; applying it turns a "low-looking"
    # 0.28 cosine into a logit of ~28, which is a strong match.
    USE_MODEL_LOGIT_SCALE: bool = True
    DENSE_LOGIT_CLIP_WEIGHT: float = 0.5   # blend weight, CLIP vs SigLIP2

    # confidence = sigmoid(SEP*sep + MARGIN*margin - BIAS), scaled down the
    # ranking by relative score ** GAMMA. sep and margin are measured in
    # pool standard deviations, so a flat pool lands near sigmoid(-BIAS).
    CONFIDENCE_SEP_WEIGHT: float = 0.6
    CONFIDENCE_MARGIN_WEIGHT: float = 0.9
    CONFIDENCE_BIAS: float = 1.5
    CONFIDENCE_RANK_GAMMA: float = 1.0
    CONFIDENCE_SINGLETON: float = 0.5      # a pool of one proves nothing

    # ---------------- Vietnamese query handling ----------------
    VI_EN_TRANSLATION_ENABLED: bool = True
    VI_EN_TRANSLATION_MODEL: str = "vinai/vinai-translate-vi2en-v2"
    VI_EN_TRANSLATION_DEVICE: str = "cpu"
    VI_EN_TRANSLATION_NUM_BEAMS: int = 4   # was 1 (greedy); query-time cost is negligible
    VI_EN_TRANSLATION_MODEL_DIR: str = ""
    # "translate" | "native" | "both". SigLIP 2 uses a multilingual Gemma
    # tokenizer, so "native" may beat machine translation on proper nouns.
    VI_QUERY_MODE: str = "translate"

    # ---------------- Keyframe extraction ----------------
    # Features are L2-normalised before differencing, so ||f_prev|| == 1 and
    # this threshold is equivalent to cosine < 0.82 against the LAST ACCEPTED
    # keyframe (not the previous frame).
    KEYFRAME_THRESHOLD: float = 0.6
    KEYFRAME_MIN_DELTA: int = 5
    FRAME_SAMPLE_RATE: int = 5
    IMAGE_SIZE: int = 224

    # ---------------- Reranking ----------------
    RRF_K: int = 60
    FUSION_ALPHA: float = 0.5

    # ---------------- BM25 ----------------
    BM25_INDEX_PATH: str = ""
    BM25_USE_VN_SEGMENTATION: bool = True   # was False; Vietnamese needs word segmentation

    # ---------------- Media metadata ----------------
    MEDIA_INFO_DIR: Path = DATA_DIR / "media-info-aic25-b1" / "media-info"
    MEDIA_FRAME_COUNT_FPS: float = 25.0

    # ---------------- Cross-encoder ----------------
    CROSS_ENCODER_ENABLED: bool = True
    CROSS_ENCODER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    CROSS_ENCODER_MODEL_DIR: str = ""

    # ---------------- Funnel top-K ----------------
    DENSE_TOP_K: int = 200
    SPARSE_TOP_K: int = 200
    FUSION_TOP_K: int = 100         # was 50; FINAL_TOP_K is 100 so 51-100 were never re-ranked
    CROSS_ENCODER_TOP_K: int = 100
    FINAL_TOP_K: int = 100
    FRAME_EXPANSION_WINDOWS: list = [5]
    FRAME_EXPANSION_MAX_SOURCES: int = 3
    FRAME_EXPANSION_FOR_UI: bool = False

    # ---------------- Result dedup ----------------
    RESULT_DEDUP_ENABLED: bool = True
    RESULT_DEDUP_MIN_FRAME_GAP: int = 15
    RESULT_DEDUP_SCENE_WINDOW_SEC: float = 8.0
    # Synthetic neighbours exist to cover frame indices, not to add variety,
    # so the diversity rules must not delete them. Without this, Stage 6
    # removed everything Stage 5 created.
    RESULT_DEDUP_EXEMPT_NEIGHBORS: bool = True

    # ---------------- TRAKE ----------------
    # 0.0 = enforce temporal order only. 0.001-0.01 additionally prefers
    # events that sit close together in time.
    TRAKE_ORDER_PENALTY: float = 0.0

    # ---------------- Retrieval ----------------
    TOP_K_DEFAULT: int = 10
    MAX_CONCURRENT_QUERIES: int = 8

    # ---------------- API ----------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: list = ["*"]

    # ---------------- OCR / ASR ----------------
    OCR_ENABLED: bool = True
    OCR_LANGUAGES: list = ["en", "vi"]
    ASR_ENABLED: bool = True
    ASR_MODEL_SIZE: str = "base"

    # ---------------- LLM VQA ----------------
    # Verify this model id against the current Google AI Studio catalogue
    # before enabling; an unknown id fails at request time.
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash"
    GEMINI_ENABLED: bool = False
    GEMINI_TOP_K_FRAMES: int = 3
    GEMINI_TIMEOUT_SEC: float = 60.0

    # ---------------- Self-hosted VLM ----------------
    VLM_ENABLED: bool = False
    VLM_URL: str = "http://localhost:8080"
    VLM_MODEL: str = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
    VLM_TIMEOUT_SEC: float = 120.0
    VLM_MAX_FRAMES: int = 10

    # ---------------- Contest pipeline ----------------
    TOPK_VIDEOS: int = 20
    TRAKE_VIDEOS: int = 50
    DELTA_RATIO: float = 0.05
    FRAME_STEP: int = 5
    MAX_ROWS: int = 100
    AUTO_INGEST_LOCAL_DATA: bool = True

    # ---------------- GPU ----------------
    GPU_ENABLED: bool = True
    GPU_DEVICE_ID: int = 0

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
