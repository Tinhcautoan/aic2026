"""
Video Retrieval System — FastAPI Backend
Implements async endpoints for video search and ingestion.
Supports Milvus (Docker) or in-memory vector store.
"""

import os
import math
import re
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from loguru import logger
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, HTMLResponse

from config import settings
from api.models import (
    QueryRequest, QueryResponse, SearchResult,
    VideoProcessRequest, VideoProcessResponse,
    BatchProcessRequest, BatchProcessResponse,
    SystemStatus, ErrorResponse,
    VQARequest, VQABatchRequest, VQAResponse,
    QaSearchRequest, QaSearchResponse, QaSearchResultItem,
    TrakeRequest, TrakeResponse, TrakeResultItem,
    RankingSaveRequest,
)

# ---- Lazy imports ----
HybridEmbeddingEngine = None
HybridReranker = None
MilvusVectorStore = None
LocalVectorStore = None
PersistentVectorStore = None
RedisCacheManager = None
PreprocessingPipeline = None
BM25Index = None
CrossEncoderReranker = None
HybridFusionPipeline = None

def _lazy_import():
    global HybridEmbeddingEngine, HybridReranker, MilvusVectorStore, LocalVectorStore
    global RedisCacheManager, PreprocessingPipeline, PersistentVectorStore
    global BM25Index, CrossEncoderReranker, HybridFusionPipeline
    if HybridEmbeddingEngine is None:
        from embedding.hybrid_engine import HybridEmbeddingEngine as HEE
        from retrieval.reranker import HybridReranker as HR
        try:
            from retrieval.milvus_store import MilvusVectorStore as MVS
        except ImportError:
            MVS = None
        from retrieval.local_store import LocalVectorStore as LVS
        from retrieval.persistent_store import PersistentVectorStore as PVS
        from retrieval.redis_cache import RedisCacheManager as RCM
        from preprocessing.pipeline import PreprocessingPipeline as PP
        from retrieval.bm25_search import BM25Index as B25
        from retrieval.cross_encoder_reranker import CrossEncoderReranker as CER
        from retrieval.hybrid_fusion import HybridFusionPipeline as HFP
        HybridEmbeddingEngine = HEE; HybridReranker = HR
        MilvusVectorStore = MVS; LocalVectorStore = LVS; PersistentVectorStore = PVS
        RedisCacheManager = RCM; PreprocessingPipeline = PP
        BM25Index = B25; CrossEncoderReranker = CER; HybridFusionPipeline = HFP

# ---- Global State ----
embedding_engine = None
reranker = None
milvus_store = None
redis_cache = None
preprocessing_pipeline = None
bm25_index = None
cross_encoder = None
hybrid_pipeline = None
startup_time = 0.0
_known_videos = []  # Track processed videos for frame serving

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def resolve_video_path(video_id: str) -> Optional[str]:
    """Find the on-disk video file for a video_id / filename stem."""
    raw = (video_id or "").strip()
    if not raw:
        return None

    stem = Path(raw).stem
    candidates = {raw, stem, f"{stem}.mp4"}

    for v in _known_videos:
        vid = v.get("video_id", "")
        path = v.get("video_path")
        if path and (vid in candidates or Path(path).name in candidates or Path(path).stem == stem):
            return path

    search_roots = [
        str(settings.VIDEO_DIR),
        str(settings.DATA_DIR),
        str(settings.PROJECT_ROOT),
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VIDEO_EXTENSIONS:
                    continue
                if fname in candidates or Path(fname).stem == stem:
                    return os.path.join(dirpath, fname)
    return None


def get_video_info(video_path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 25.0, "frame_count": 0, "width": 0, "height": 0, "duration_sec": 0.0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def find_video_files(path: str) -> List[str]:
    """Find all video files from a file or folder path."""
    p = os.path.abspath(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path not found: {p}")
    if os.path.isfile(p):
        if os.path.splitext(p)[1].lower() in VIDEO_EXTENSIONS:
            return [p]
        raise ValueError(f"Not a video file: {p}")
    if os.path.isdir(p):
        files = []
        for root, dirs, filenames in os.walk(p):
            for f in filenames:
                if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                    files.append(os.path.join(root, f))
        return files
    raise ValueError(f"Unsupported path: {p}")


def resolve_query_image_path(image_path: str) -> str:
    """Resolve VKIS query image path from absolute or common project-relative locations."""
    raw = (image_path or "").strip()
    if not raw:
        raise FileNotFoundError("Empty image_path")

    candidates = []
    candidates.append(raw)
    candidates.append(os.path.join(str(settings.PROJECT_ROOT), raw))
    candidates.append(os.path.join(str(settings.DATA_DIR), raw))
    candidates.append(os.path.join(str(settings.DATA_DIR), "queries", raw))
    candidates.append(os.path.join(str(settings.DATA_DIR), "query_images", raw))

    seen = set()
    for p in candidates:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.isfile(ap):
            return ap

    raise FileNotFoundError(f"Query image not found: {raw}")


@asynccontextmanager
def _warn_torchaudio_mismatch():
    """Log a clear action message if torchaudio's CUDA build conflicts with torch.

    transformers v5 imports torchaudio even for text models (processing_utils ->
    audio_utils), and a mismatched torchaudio raises a RuntimeError at import
    time that surfaces as a confusing "Could not import module 'X'".
    """
    try:
        import importlib.util
        import torch
        if not torch.cuda.is_available() or importlib.util.find_spec("torchaudio") is None:
            return
        try:
            import torchaudio  # raises RuntimeError on CUDA mismatch
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.warning(
                    "torchaudio is compiled for a different CUDA version than torch, "
                    "which breaks transformers model loading. Fix: reinstall the matching set — "
                    "`pip uninstall -y torch torchvision torchaudio && pip install -r requirements.txt`"
                )
    except Exception:
        pass


def _display_confidence(r: dict) -> float:
    """Map a result's score fields to a 0-1 confidence for the UI.

    The cross-encoder (bge-reranker) outputs RAW LOGITS — unbounded, and
    often negative for weak matches — so the value must NOT be shown as a
    "confidence".  Its sigmoid probability is the honest 0-1 confidence.
    For dense-only results (cosine similarities) we clamp into [0, 1].
    Ordering is unchanged: sigmoid is strictly monotonic.
    """
    p = r.get("cross_encoder_prob")
    if p is not None:
        return round(min(max(float(p), 0.0), 1.0), 4)
    s = r.get("cross_encoder_score")
    if s is not None:
        return round(min(max(1.0 / (1.0 + math.exp(-float(s))), 0.0), 1.0), 4)
    score = r.get("final_rerank_score", r.get("final_score", r.get("clip_score", 0.0)))
    return round(min(max(float(score), 0.0), 1.0), 4)


async def lifespan(app: FastAPI):
    global embedding_engine, reranker, milvus_store, redis_cache, preprocessing_pipeline, startup_time
    global bm25_index, cross_encoder, hybrid_pipeline
    startup_time = time.time()
    logger.info("Starting Video Retrieval System...")
    _warn_torchaudio_mismatch()
    try:
        _lazy_import()
        embedding_engine = HybridEmbeddingEngine()
        logger.info("Embedding engine (TULIP+SigLIP2) initialized on GPU")
    except Exception as e:
        logger.error(f"Failed to initialize embedding engine: {e}")
        embedding_engine = None
    reranker = HybridReranker() if HybridReranker else None
    # Milvus Lite (file-based). Falls back to PersistentVectorStore if unavailable.
    if MilvusVectorStore:
        try:
            milvus_store = MilvusVectorStore()
            logger.info(f"Milvus Lite ready: {settings.MILVUS_DB_PATH}")
        except Exception as e:
            logger.opt(exception=True).error(
                f"Milvus Lite init failed: {e}. "
                f"Check read/write permissions for {settings.MILVUS_DB_PATH}"
            )
            milvus_store = None
    if milvus_store is None and PersistentVectorStore:
        try:
            store_path = str(settings.DATA_DIR / "persistent_store")
            milvus_store = PersistentVectorStore(store_path)
            logger.info(f"Fallback to PersistentVectorStore at {store_path}")
        except Exception as e:
            logger.error(f"PersistentVectorStore init failed: {e}")
            milvus_store = None
    if RedisCacheManager:
        redis_cache = RedisCacheManager()
        try: await redis_cache.connect()
        except Exception: redis_cache = None
    if PreprocessingPipeline:
        preprocessing_pipeline = PreprocessingPipeline(embedding_engine=embedding_engine, milvus_store=milvus_store)

    # ---- Preload all models at startup (except the VLM, which is an external API) ----
    if embedding_engine is not None:
        try:
            embedding_engine.warmup()
        except Exception as e:
            logger.warning(f"Embedding engine warmup failed: {e}")
    if preprocessing_pipeline is not None:
        try:
            preprocessing_pipeline.keyframe_extractor.warmup()
        except Exception as e:
            logger.warning(f"Keyframe model warmup failed: {e}")

    # ---- Ingest existing local data (persistent_store / keyframes / videos) ----
    if milvus_store is not None and settings.AUTO_INGEST_LOCAL_DATA:
        from retrieval.local_ingest import ingest_local_data_if_empty
        try:
            summary = ingest_local_data_if_empty(
                milvus_store,
                embedding_engine=embedding_engine,
                pipeline=preprocessing_pipeline,
            )
            logger.info(f"Local data ingest: {summary}")
        except Exception as e:
            logger.warning(f"Local data ingest failed: {e}")

    # ---- BM25 sparse index: try to restore from disk, else start empty ----
    # Then auto-build from stored frame metadata (incl. contest media-info
    # video text) when the persisted index is missing/stale/empty.
    if BM25Index:
        try:
            bm25_index = BM25Index()
            loaded = bm25_index.load()
            logger.info(f"BM25 index {'loaded from disk' if loaded else 'starting empty'} "
                        f"({bm25_index.size} documents)")
            if (not loaded or bm25_index.size == 0) and milvus_store is not None:
                from retrieval.bm25_search import build_bm25_index_from_store
                n = build_bm25_index_from_store(milvus_store, bm25_index)
                if n:
                    bm25_index.save()
                logger.info(f"BM25 index auto-built from Milvus store: {bm25_index.size} documents")
        except Exception as e:
            logger.warning(f"BM25 index unavailable: {e}")
            bm25_index = None

    # ---- Cross-Encoder: heavier load, don't crash startup if it fails ----
    if CrossEncoderReranker and settings.CROSS_ENCODER_ENABLED:
        try:
            cross_encoder = CrossEncoderReranker()
            logger.info(f"Cross-Encoder loaded: {settings.CROSS_ENCODER_MODEL_NAME}")
        except Exception as e:
            logger.opt(exception=True).warning(
                f"Cross-Encoder unavailable, Stage 4 will be skipped: {e}"
            )
            cross_encoder = None

    # ---- Full funnel orchestrator ----
    if HybridFusionPipeline and milvus_store and reranker:
        hybrid_pipeline = HybridFusionPipeline(
            vector_store=milvus_store,
            bm25_index=bm25_index,
            dense_reranker=reranker,
            cross_encoder=cross_encoder,
        )
        logger.info("HybridFusionPipeline ready (Dense -> BM25 -> RRF -> Cross-Encoder -> Expansion)")

    logger.info("System ready!")
    yield
    logger.info("Shutting down...")
    if milvus_store: milvus_store.close()
    if redis_cache and redis_cache.client: await redis_cache.disconnect()


PROCESS_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Process Video</title>
<style>
    :root {
        --bg: #0f0f14;
        --panel: #1a1a24;
        --border: #2a2a3a;
        --text: #e0e0e8;
        --text-secondary: #8888a0;
        --accent: #6c5ce7;
        --success: #00d68f;
        --error: #ff5252;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: 'Segoe UI', system-ui, sans-serif;
        background: var(--bg);
        color: var(--text);
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .container {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 32px;
        width: 520px;
        max-width: 95vw;
    }
    h1 { font-size: 1.3rem; margin-bottom: 6px; }
    p.sub { color: var(--text-secondary); font-size: 0.82rem; margin-bottom: 20px; }
    label {
        display: block;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    input[type="text"] {
        width: 100%;
        padding: 10px 12px;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
        font-size: 0.9rem;
        outline: none;
        font-family: inherit;
        transition: border-color 0.2s;
        margin-bottom: 16px;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    .mode-row {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
    }
    .mode-row label {
        font-size: 0.85rem;
        text-transform: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        color: var(--text);
    }
    input[type="radio"] { accent-color: var(--accent); }
    button {
        width: 100%;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        cursor: pointer;
        background: var(--accent);
        color: white;
        transition: opacity 0.2s;
    }
    button:hover { opacity: 0.9; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #status {
        margin-top: 16px;
        padding: 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        display: none;
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
    }
    #status.success { display: block; background: rgba(0,214,143,0.1); border: 1px solid var(--success); color: var(--success); }
    #status.error { display: block; background: rgba(255,82,82,0.1); border: 1px solid var(--error); color: var(--error); }
    #status.info { display: block; background: rgba(108,92,231,0.1); border: 1px solid var(--accent); color: var(--accent); }
    .nav { margin-top: 20px; text-align: center; }
    .nav a { color: var(--text-secondary); font-size: 0.82rem; text-decoration: none; }
    .nav a:hover { color: var(--accent); }
</style>
</head>
<body>
<div class="container">
    <h1>🎬 Process Video</h1>
    <p class="sub">Paste a video file path or a folder containing videos.</p>

    <label>Path</label>
    <input type="text" id="pathInput" placeholder="C:\Users\...\my_video.mp4  or  C:\Users\...\MyVideos\" autofocus>

    <div class="mode-row">
        <label><input type="radio" name="mode" value="file" checked> Single File</label>
        <label><input type="radio" name="mode" value="folder"> Entire Folder</label>
    </div>

    <button id="processBtn" onclick="doProcess()">Process</button>

    <div id="status"></div>

    <div class="nav">
        <a href="/ui">🔍 Search</a> &nbsp;·&nbsp;
        <a href="/api/status">📊 Status</a>
    </div>
</div>

<script>
async function doProcess() {
    const path = document.getElementById('pathInput').value.trim();
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const btn = document.getElementById('processBtn');
    const status = document.getElementById('status');

    if (!path) {
        status.className = 'error';
        status.textContent = 'Please enter a path.';
        return;
    }

    status.className = 'info';
    status.textContent = 'Processing...';
    btn.disabled = true;

    try {
        const r = await fetch('/api/process?mode=' + mode, {
            method: 'POST',
            headers: {'Content-Type': 'text/plain'},
            body: path,
        });
        const text = await r.text();
        status.className = r.ok ? 'success' : 'error';
        status.textContent = text;
    } catch(e) {
        status.className = 'error';
        status.textContent = 'Request failed: ' + e.message;
    } finally {
        btn.disabled = false;
    }
}

document.getElementById('pathInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') doProcess();
});
</script>
</body>
</html>"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="Video Retrieval System",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    static_dir = settings.PROJECT_ROOT / "frontend"
    if static_dir.exists(): app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def root():
        return {"name": "Video Retrieval System", "version": "2.0.0"}

    @app.get("/ui")
    async def get_ui():
        """Serve the main search UI."""
        ui_file = settings.PROJECT_ROOT / "frontend" / "index.html"
        if ui_file.exists():
            return FileResponse(str(ui_file), media_type="text/html")
        raise HTTPException(404, "UI not found")

    @app.get("/batch_answer.html")
    async def get_batch_answer():
        """Serve the batch answer submission UI."""
        batch_file = settings.PROJECT_ROOT / "frontend" / "batch_answer.html"
        if batch_file.exists():
            return FileResponse(str(batch_file), media_type="text/html")
        raise HTTPException(404, "Batch answer UI not found")

    @app.get("/api/status")
    async def get_status():
        ms = {}; rs = {}
        if milvus_store:
            try: ms = milvus_store.get_collection_stats()
            except Exception as e: ms = {"error": str(e)}
        if redis_cache:
            try: rs = await redis_cache.get_cache_stats()
            except Exception: pass
        ml = []
        model_label = f"TULIP ({settings.TULIP_MODEL_NAME})" if settings.USE_TULIP else f"CLIP ({settings.CLIP_MODEL_NAME})"
        if embedding_engine: ml = [model_label, f"SigLIP2 ({settings.SIGLIP2_MODEL_NAME})"]
        gpu = False
        try: import torch; gpu = torch.cuda.is_available()
        except Exception: pass
        return {"status": "healthy", "milvus": ms, "redis": rs, "gpu_available": gpu, "models_loaded": ml, "uptime_seconds": time.time() - startup_time}

    async def _search_by_embeddings(
        clip_emb: "np.ndarray",
        siglip2_emb: "np.ndarray",
        query_text: str,
        top_k: int,
        video_filter: Optional[str] = None,
        use_full_funnel: bool = True,
        use_bm25: bool = True,
        use_cross_encoder: bool = True,
        expand_neighbors: Optional[bool] = None,
        dedup: bool = True,
    ):
        if embedding_engine is None: raise HTTPException(503, "Embedding engine not initialized")
        if milvus_store is None: raise HTTPException(503, "Store not available. Process a video first.")
        qs = time.time()
        expr = f'video_id == "{video_filter}"' if video_filter else None

        if expand_neighbors is None:
            expand_neighbors = top_k > 20 or settings.FRAME_EXPANSION_FOR_UI

        if use_full_funnel and hybrid_pipeline is not None:
            fr = hybrid_pipeline.search(
                query=query_text,
                clip_query_emb=clip_emb,
                siglip2_query_emb=siglip2_emb,
                final_top_k=max(top_k, settings.FINAL_TOP_K),
                expr=expr,
                use_bm25=use_bm25,
                use_cross_encoder=use_cross_encoder,
                expand_neighbors=expand_neighbors,
                dedup=dedup,
            )
            fr = fr[:top_k] if top_k < len(fr) else fr
        else:
            from retrieval.result_dedup import deduplicate_results
            candidates = milvus_store.hybrid_search(clip_query=clip_emb, siglip2_query=siglip2_emb, top_k=top_k * 3, expr=expr)
            fr = reranker.rerank(candidates, top_k=top_k) if reranker else candidates[:top_k]
            if dedup:
                fr = deduplicate_results(fr, top_k=top_k)

        latency = (time.time() - qs) * 1000
        results = [{
            "id": r.get("id", 0), "video_id": r.get("video_id", ""), "frame_idx": r.get("frame_idx", 0),
            "timestamp": r.get("timestamp", 0.0),
            "clip_score": round(r.get("clip_score", 0.0), 4),
            "siglip2_score": round(r.get("siglip2_score", 0.0), 4),
            "bm25_score": round(r.get("bm25_score", 0.0), 4) if "bm25_score" in r else None,
            "cross_encoder_score": round(r.get("cross_encoder_score", 0.0), 4) if "cross_encoder_score" in r else None,
            "confidence": _display_confidence(r),
            "final_score": round(r.get("final_rerank_score", r.get("final_score", r.get("clip_score", 0.0))), 4),
            "is_expanded_neighbor": r.get("is_expanded_neighbor", False),
        } for r in fr]
        return {
            "query": query_text, "latency_ms": round(latency, 1), "total_results": len(results),
            "funnel_used": "full" if (use_full_funnel and hybrid_pipeline is not None) else "dense_only",
            "results": results,
        }

    @app.post("/api/search")
    async def search(
        request: QueryRequest,
        use_full_funnel: bool = Query(True, description="Dense->BM25->RRF->CrossEncoder->Expansion. False = dense-only (fast/debug)."),
        use_bm25: bool = Query(True),
        use_cross_encoder: bool = Query(True),
        expand_neighbors: bool = Query(None, description="Expand top hits to +-N frame neighbors. Default: off for top_k<=20."),
        dedup: bool = Query(True, description="Remove near-duplicate frames from results."),
    ):
        if embedding_engine is None:
            raise HTTPException(503, "Embedding engine not initialized")
        hybrid_embs = await embedding_engine.async_encode_text(request.query)
        clip_emb, siglip2_emb = hybrid_embs["clip"], hybrid_embs["siglip2"]
        return await _search_by_embeddings(
            clip_emb=clip_emb,
            siglip2_emb=siglip2_emb,
            query_text=request.query,
            top_k=request.top_k,
            video_filter=request.video_filter,
            use_full_funnel=use_full_funnel,
            use_bm25=use_bm25,
            use_cross_encoder=use_cross_encoder,
            expand_neighbors=expand_neighbors,
            dedup=dedup,
        )

    @app.post("/api/search-image")
    async def search_image(
        file: UploadFile = File(..., description="Upload image query for VKIS"),
        top_k: int = Form(10, ge=1, le=100),
        use_full_funnel: bool = Form(True),
        use_bm25: bool = Form(True),
        use_cross_encoder: bool = Form(True),
        expand_neighbors: bool = Form(False),
        dedup: bool = Form(True),
    ):
        if embedding_engine is None: raise HTTPException(503, "Embedding engine not initialized")
        if milvus_store is None: raise HTTPException(503, "Store not available. Process a video first.")

        try:
            from PIL import Image
            image = Image.open(file.file).convert("RGB")
            hybrid_embs = embedding_engine.encode_image(image)
            clip_emb, siglip2_emb = hybrid_embs["clip"], hybrid_embs["siglip2"]
            return await _search_by_embeddings(
                clip_emb=clip_emb,
                siglip2_emb=siglip2_emb,
                query_text=f"visual_query:{file.filename}",
                top_k=top_k,
                use_full_funnel=use_full_funnel,
                use_bm25=use_bm25,
                use_cross_encoder=use_cross_encoder,
                expand_neighbors=expand_neighbors,
                dedup=dedup,
            )
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            raise HTTPException(400, f"Failed to process uploaded image: {e}")

    @app.post("/api/search-image-path")
    async def search_image_path(
        image_path: str = Form(..., description="Server-side image path for VKIS"),
        top_k: int = Form(10, ge=1, le=100),
        use_full_funnel: bool = Form(True),
        use_bm25: bool = Form(True),
        use_cross_encoder: bool = Form(True),
        expand_neighbors: bool = Form(False),
        dedup: bool = Form(True),
    ):
        if embedding_engine is None:
            raise HTTPException(503, "Embedding engine not initialized")
        if milvus_store is None:
            raise HTTPException(503, "Store not available. Process a video first.")

        try:
            from PIL import Image

            resolved_path = resolve_query_image_path(image_path)
            image = Image.open(resolved_path).convert("RGB")
            hybrid_embs = embedding_engine.encode_image(image)
            clip_emb, siglip2_emb = hybrid_embs["clip"], hybrid_embs["siglip2"]
            return await _search_by_embeddings(
                clip_emb=clip_emb,
                siglip2_emb=siglip2_emb,
                query_text=f"visual_query:{os.path.basename(resolved_path)}",
                top_k=top_k,
                use_full_funnel=use_full_funnel,
                use_bm25=use_bm25,
                use_cross_encoder=use_cross_encoder,
                expand_neighbors=expand_neighbors,
                dedup=dedup,
            )
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            logger.error(f"Image path search failed: {e}")
            raise HTTPException(400, f"Failed to process image path: {e}")

    @app.post("/api/bm25/index")
    async def bm25_add_documents(records: List[dict]):
        """
        Add per-frame text records to the BM25 sparse index and persist it.
        Body: [{"video_id": "...", "frame_idx": 123, "metadata": {"ocr_text": "...",
                "asr_transcript": "...", "objects": ["car","person"]}}, ...]
        Call this once per video after /api/process, using the OCR/ASR/Objects
        JSON your team already has for that video (from BTC-provided data or
        your own extraction). Rebuilds the in-memory index and saves to disk.
        """
        global bm25_index, hybrid_pipeline
        if bm25_index is None:
            if BM25Index is None:
                raise HTTPException(503, "BM25Index module not available")
            bm25_index = BM25Index()
        if not records:
            raise HTTPException(400, "No records provided")
        t0 = time.time()
        bm25_index.add_documents(records)
        bm25_index.build()
        bm25_index.save()
        if hybrid_pipeline is not None:
            hybrid_pipeline.bm25_index = bm25_index
        return {
            "status": "ok", "added": len(records), "total_documents": bm25_index.size,
            "elapsed_s": round(time.time() - t0, 2),
        }

    @app.get("/api/bm25/status")
    async def bm25_status():
        if bm25_index is None:
            return {"available": False, "documents": 0}
        return {"available": True, "documents": bm25_index.size}

    @app.delete("/api/bm25/videos/{video_id}")
    async def bm25_delete_video(video_id: str):
        if bm25_index is None:
            raise HTTPException(503, "BM25 index not available")
        bm25_index.delete_by_video(video_id)
        bm25_index.save()
        return {"status": "deleted", "video_id": video_id, "remaining_documents": bm25_index.size}

    @app.post("/api/process")
    async def process_video(
        request: Request,
        mode: str = Query("file", pattern="^(file|folder)$"),
    ):
        """Process a video file or all videos in a folder.
        Accepts raw text body with the file/folder path.
        Use ?mode=file (default) for a single file, ?mode=folder for a directory.
        """
        global _known_videos
        pipeline = preprocessing_pipeline
        if pipeline is None:
            raise HTTPException(503, "Pipeline not available")

        body = (await request.body()).decode("utf-8").strip()
        if not body:
            raise HTTPException(400, "Request body must contain a file or folder path")

        # Find video files
        try:
            video_paths = find_video_files(body)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

        if not video_paths:
            raise HTTPException(400, f"No video files found at: {body}")

        results = []
        total_start = time.time()
        for vp in video_paths:
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda vp=vp: pipeline.process_video(
                        video_path=vp, save_keyframes=False))
                vid = result.get("video_id", "")
                if result.get("status") == "success" and vid:
                    _known_videos.append({"video_id": vid, "video_path": vp})
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {vp}: {e}")
                results.append({"video_path": vp, "status": "error", "error": str(e)})

        total_elapsed = time.time() - total_start
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") != "success"]

        lines = []
        lines.append(f"Done! Processed {len(video_paths)} video(s) in {total_elapsed:.1f}s")
        lines.append(f"  Success: {len(successful)}")
        lines.append(f"  Failed:  {len(failed)}")
        for r in successful:
            lines.append(f"  [OK] {r.get('video_id', '?')} — {r.get('keyframes', 0)} frames in {r.get('processing_time', 0):.1f}s")
        for r in failed:
            lines.append(f"  [FAIL] {r.get('video_path', '?')} — {r.get('error', 'unknown')}")

        return Response("\n".join(lines), media_type="text/plain")

    @app.get("/api/frame/{video_id}/{frame_idx}")
    async def get_frame(video_id: str, frame_idx: int):
        """Serve a frame from a processed video as JPEG image."""
        import cv2
        video_path = resolve_video_path(video_id)
        if not video_path:
            raise HTTPException(404, f"Video not found: {video_id}")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(404, f"Frame {frame_idx} not found")
        _, jpeg = cv2.imencode(".jpg", frame)
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")

    @app.get("/api/video/{video_id}")
    async def get_video(video_id: str):
        """Stream the source video file for in-browser playback."""
        video_path = resolve_video_path(video_id)
        if not video_path:
            raise HTTPException(404, f"Video not found: {video_id}")
        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/video/{video_id}/info")
    async def get_video_info_endpoint(video_id: str):
        video_path = resolve_video_path(video_id)
        if not video_path:
            raise HTTPException(404, f"Video not found: {video_id}")
        info = get_video_info(video_path)
        return {"video_id": Path(video_id).stem, "video_path": video_path, **info}

    async def _answer_vqa_item(item: VQARequest) -> VQAResponse:
        from retrieval.vqa import normalize_video_id, answer_from_retrieval_results

        hybrid_embs = await embedding_engine.async_encode_text(item.question)
        clip_emb, siglip2_emb = hybrid_embs["clip"], hybrid_embs["siglip2"]

        # Global search (no video filter) when video is None; within-video otherwise
        expr = f'video_id == "{normalize_video_id(item.video)}"' if item.video else None

        if hybrid_pipeline is not None:
            raw_results = hybrid_pipeline.search(
                query=item.question,
                clip_query_emb=clip_emb,
                siglip2_query_emb=siglip2_emb,
                final_top_k=20,
                expr=expr,
                use_bm25=True,
                use_cross_encoder=True,
                expand_neighbors=False,
                dedup=True,
            )
        else:
            candidates = milvus_store.hybrid_search(
                clip_query=clip_emb, siglip2_query=siglip2_emb, top_k=20, expr=expr,
            )
            raw_results = reranker.rerank(candidates, top_k=20) if reranker else candidates[:20]

        out = answer_from_retrieval_results(
            question=item.question,
            results=raw_results,
            cross_encoder=cross_encoder,
        )

        if settings.GEMINI_ENABLED and settings.GEMINI_API_KEY:
            try:
                from retrieval.vqa_llm import answer_from_llm

                llm_out = await answer_from_llm(
                    question=item.question,
                    results=raw_results,
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.GEMINI_MODEL,
                    top_k=settings.GEMINI_TOP_K_FRAMES,
                    timeout=settings.GEMINI_TIMEOUT_SEC,
                )
                if llm_out.get("answer"):
                    out = llm_out
            except Exception as e:
                logger.warning(f"LLM VQA fallback to metadata: {e}")

        return VQAResponse(
            id=item.id,
            video=item.video,
            question=item.question,
            answer=out.get("answer", ""),
            evidence_video_id=out.get("evidence_video_id") or item.video,
            evidence_frame_idx=out.get("evidence_frame_idx"),
            evidence_timestamp=out.get("evidence_timestamp"),
            confidence=out.get("confidence", 0.0),
            source=out.get("source", "metadata"),
        )

    @app.post("/api/vqa/answer", response_model=VQAResponse)
    async def vqa_answer(item: VQARequest):
        if embedding_engine is None:
            raise HTTPException(503, "Embedding engine not initialized")
        if milvus_store is None:
            raise HTTPException(503, "Store not available. Process videos first.")
        return await _answer_vqa_item(item)

    @app.post("/api/vqa/batch")
    async def vqa_batch(request: VQABatchRequest):
        if embedding_engine is None:
            raise HTTPException(503, "Embedding engine not initialized")
        if milvus_store is None:
            raise HTTPException(503, "Store not available. Process videos first.")
        results = []
        for item in request.items:
            try:
                results.append((await _answer_vqa_item(item)).model_dump())
            except Exception as e:
                results.append({
                    "id": item.id,
                    "video": item.video,
                    "question": item.question,
                    "answer": "",
                    "error": str(e),
                })
        return {"total": len(results), "results": results}

    @app.post("/api/qa/search", response_model=QaSearchResponse)
    async def qa_search(request: QaSearchRequest):
        if embedding_engine is None:
            raise HTTPException(503, "Embedding engine not initialized")
        if milvus_store is None:
            raise HTTPException(503, "Store not available. Process videos first.")

        from retrieval.vqa_llm import answer_from_vlm
        from retrieval.frame_sampler import sample_video_windows
        import time as _time

        t0 = _time.time()

        hybrid_embs = await embedding_engine.async_encode_text(request.description)
        clip_emb, siglip2_emb = hybrid_embs["clip"], hybrid_embs["siglip2"]

        if hybrid_pipeline is not None:
            search_results = hybrid_pipeline.search(
                query=request.description,
                clip_query_emb=clip_emb,
                siglip2_query_emb=siglip2_emb,
                final_top_k=request.top_k,
                use_bm25=True,
                use_cross_encoder=True,
                expand_neighbors=False,
                dedup=True,
            )
        else:
            candidates = milvus_store.hybrid_search(
                clip_query=clip_emb, siglip2_query_emb=siglip2_emb, top_k=request.top_k,
            )
            search_results = reranker.rerank(candidates, top_k=request.top_k) if reranker else candidates[:request.top_k]

        out_results: list[QaSearchResultItem] = []
        for r in search_results[:request.top_k]:
            vid = r["video_id"]
            kf_idx = r.get("frame_idx", 0)
            score = r.get("final_rerank_score", r.get("rrf_fusion_score", 0.0))
            frame_count = r.get("frame_count", 2000)
            try:
                sampled = sample_video_windows(
                    video_id=vid,
                    keyframes=[{"frame_idx": kf_idx, "score": score}],
                    frame_count=frame_count,
                    delta_ratio=settings.DELTA_RATIO,
                    frame_step=settings.FRAME_STEP,
                )
                vlm_frames = [{"video_id": vid, "frame_idx": sf.frame_idx, "score": sf.score} for sf in sampled]
                vlm_out = await answer_from_vlm(request.question, vlm_frames, mode="qa")
                answer = vlm_out.get("answer", "") if vlm_out else ""
            except Exception as e:
                logger.warning(f"VLM call failed for {vid} frame {kf_idx}: {e}")
                answer = ""

            _QA_VAGUE_RE = re.compile(
                r"^\s*$|không\s+xác\s+định|unknown|undefined|n/?a|không\s+biết|"
                r"không\s+thể\s+xác\s+định|không\s+trả\s+lời|không\s+có\s+đáp\s+án",
                re.IGNORECASE,
            )
            is_vague = bool(_QA_VAGUE_RE.match(answer))

            out_results.append(QaSearchResultItem(
                video_id=vid,
                frame_idx=kf_idx,
                answer=answer,
                is_vague=is_vague,
                final_score=score,
                confidence=_display_confidence(r),
                timestamp=r.get("timestamp", 0.0),
                clip_score=r.get("clip_score", 0.0),
                siglip2_score=r.get("siglip2_score", 0.0),
            ))

        return QaSearchResponse(
            query=request.description,
            question=request.question,
            total_results=len(out_results),
            latency_ms=round((_time.time() - t0) * 1000, 1),
            results=out_results,
        )

    def _make_event_search_fn_for_video(video_id: str):
        """Create an event_search_fn scoped to a single video's frames.

        Uses ALL stored frame embeddings for the video (not the sampled
        candidate indices, which may not exist in the store).
        """
        import numpy as np

        _cache: dict = {}

        def event_search_fn(event_text: str, candidates: list[dict]) -> list[tuple[int, float]]:
            if "all_embs" not in _cache:
                _cache["all_embs"] = milvus_store.get_frame_embeddings(
                    video_id=video_id, frame_indices=None,
                )

            cand_indices = {c["frame_idx"] for c in candidates}

            embs = embedding_engine.encode_text(event_text)
            clip_emb = embs["clip"]
            siglip2_emb = embs["siglip2"]

            frame_embs = _cache["all_embs"]
            results = []
            for fe in frame_embs:
                fi = fe["frame_idx"]
                if cand_indices and fi not in cand_indices:
                    continue
                clip_sim = float(np.dot(clip_emb, fe["clip_emb"]))
                siglip2_sim = float(np.dot(siglip2_emb, fe["siglip2_emb"]))
                avg_sim = (clip_sim + siglip2_sim) / 2.0
                results.append((fi, avg_sim))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

        return event_search_fn

    @app.post("/api/trake", response_model=TrakeResponse)
    async def trake_endpoint(request: TrakeRequest):
        if embedding_engine is None:
            raise HTTPException(503, "Embedding engine not initialized")
        if milvus_store is None:
            raise HTTPException(503, "Store not available. Process videos first.")

        from retrieval.trake_aligner import align_video_events, rank_videos
        from retrieval.frame_sampler import sample_video_windows
        from retrieval.query_parser import ParsedTrake
        import time as _time

        t0 = _time.time()
        parsed = ParsedTrake(main=request.description, events=request.events, event_count=len(request.events))

        hybrid_embs = await embedding_engine.async_encode_text(parsed.main)
        clip_emb, siglip2_emb = hybrid_embs["clip"], hybrid_embs["siglip2"]

        if hybrid_pipeline is not None:
            search_results = hybrid_pipeline.search(
                query=parsed.main,
                clip_query_emb=clip_emb,
                siglip2_query_emb=siglip2_emb,
                final_top_k=request.top_k,
                use_bm25=True,
                use_cross_encoder=True,
                expand_neighbors=False,
                dedup=True,
            )
        else:
            candidates = milvus_store.hybrid_search(
                clip_query=clip_emb, siglip2_query_emb=siglip2_emb, top_k=request.top_k,
            )
            search_results = reranker.rerank(candidates, top_k=request.top_k) if reranker else candidates[:request.top_k]

        # Per-video real FPS (not all videos are 25fps -- e.g. some are 30/29.97),
        # cached per request so repeated hits on the same video_id don't reopen it.
        _fps_cache: dict = {}

        def _video_fps(vid: str) -> float:
            if vid in _fps_cache:
                return _fps_cache[vid]
            fps = 25.0
            path = resolve_video_path(vid)
            if path:
                try:
                    import cv2
                    cap = cv2.VideoCapture(path)
                    f = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if f and f > 0:
                        fps = f
                except Exception:
                    pass
            _fps_cache[vid] = fps
            return fps

        alignments = []
        for r in search_results[:request.top_k]:
            vid = r["video_id"]
            kf_idx = r.get("frame_idx", 0)
            score = r.get("final_rerank_score", r.get("rrf_fusion_score", 0.0))
            frame_count = r.get("frame_count", 2000)
            try:
                sampled = sample_video_windows(
                    video_id=vid,
                    keyframes=[{"frame_idx": kf_idx, "score": score}],
                    frame_count=frame_count,
                    delta_ratio=settings.DELTA_RATIO,
                    frame_step=settings.FRAME_STEP,
                )
                candidate_dicts = [{"video_id": vid, "frame_idx": sf.frame_idx, "score": sf.score} for sf in sampled]
                event_search_fn = _make_event_search_fn_for_video(vid)
                alignment = align_video_events(
                    video_id=vid,
                    candidate_frames=candidate_dicts,
                    event_texts=parsed.events,
                    query_score=score,
                    event_search_fn=event_search_fn,
                )
                alignments.append(alignment)
            except Exception as e:
                logger.warning(f"TRAKE alignment failed for {vid}: {e}")

        ranked = rank_videos(alignments, max_rows=request.top_k)
        out_results: list[TrakeResultItem] = []
        for a in ranked:
            frames = [ef.frame_idx for ef in a.event_frames if ef.frame_idx >= 0]
            if len(frames) == len(parsed.events):
                out_results.append(TrakeResultItem(
                    video_id=a.video_id,
                    frame_idx=frames[0] if frames else 0,
                    frames=frames,
                    score=a.total_score,
                    confidence=round(min(max(1.0 / (1.0 + math.exp(-float(a.total_score))), 0.0), 1.0), 4),
                    timestamp=float(frames[0]) / _video_fps(a.video_id) if frames else 0.0,
                    event_frames=[{"event": ef.event_text, "frame_idx": ef.frame_idx, "score": ef.score} for ef in a.event_frames],
                ))

        return TrakeResponse(
            query=parsed.main,
            total_results=len(out_results),
            latency_ms=round((_time.time() - t0) * 1000, 1),
            results=out_results,
        )

    @app.get("/api/videos")
    async def list_videos():
        return _known_videos

    @app.delete("/api/videos/{video_id}")
    async def delete_video(video_id: str):
        global _known_videos
        if milvus_store is None: raise HTTPException(503, "Store not available")
        milvus_store.delete_by_video(video_id)
        _known_videos = [v for v in _known_videos if v.get("video_id") != video_id]
        return {"status": "deleted", "video_id": video_id}

    # ---- Manual ranking store (display-layer submission text files) ----
    # These endpoints only read/write plain-text files under submission/.
    # They never call the embedding engine or the vector store, so editing
    # a ranking here cannot change any model output.

    @app.get("/api/ranking/list")
    async def ranking_list():
        from retrieval import ranking_store
        return {
            "files": ranking_store.list_rankings(),
            "dir": str(ranking_store.ranking_dir()),
        }

    @app.get("/api/ranking/load")
    async def ranking_load(name: str = Query(..., description="File name under submission/")):
        from retrieval import ranking_store
        try:
            return ranking_store.load_ranking(name)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.post("/api/ranking/save")
    async def ranking_save(req: RankingSaveRequest):
        from retrieval import ranking_store
        rows = [r.model_dump(exclude_none=True) for r in req.rows]
        try:
            res = ranking_store.save_ranking(req.name, rows, mode=req.mode)
        except Exception as e:
            logger.error(f"Ranking save failed: {e}")
            raise HTTPException(400, f"Failed to save ranking: {e}")
        return {"status": "ok", **res}

    @app.delete("/api/ranking/delete")
    async def ranking_delete(name: str = Query(..., description="File name under submission/")):
        from retrieval import ranking_store
        if not ranking_store.delete_ranking(name):
            raise HTTPException(404, f"Ranking file not found: {name}")
        return {"status": "deleted", "name": name}

    @app.get("/api/health")
    async def health_check():
        return {"status": "ok"}

    @app.get("/process", response_class=HTMLResponse)
    async def serve_process_page():
        return PROCESS_PAGE_HTML

    return app

app = create_app()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host=settings.API_HOST, port=settings.API_PORT, reload=True, log_level="info")