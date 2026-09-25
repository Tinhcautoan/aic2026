"""
AIC 2026 final round -- lean PE-Core-only retrieval API + DRES submission.

    text  --translate(vi->en, GPU)-->  PE-Core text tower  -->  one matrix-vector product
    over every frame in the GPU index  -->  temporal NMS  -->  JSON

No fusion, BM25, cross-encoder, second encoder or vector database. TRAKE follows the
AIC-HQT method (sub-event ranking + whole-video candidates + monotone DP alignment).
The browser talks only to this server; DRES credentials/sessions live here, and the
DRES endpoints answer this machine only (see DRES_ALLOW_REMOTE).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from retrieval.dres_client import DresClient, DresError, build_kis, build_qa, build_trake

# --------------------------------------------------------------------------- state


class Engine(SimpleNamespace):
    """Everything heavy. Filled by load_engine() (in a thread) or injected by tests."""

    index = None          # retrieval.frame_index.FrameIndex
    text = None           # PE-Core text tower (.encode(list[str]) -> [n, D])
    translator = None     # embedding.translate_fast.FastTranslator or None
    thumbs = None         # retrieval.thumbs.ThumbService
    ready = False
    error = None
    load_s = None


def _pick_translate_device() -> str:
    if settings.TRANSLATE_DEVICE != "auto":
        return settings.TRANSLATE_DEVICE
    try:
        import torch
        return f"cuda:{settings.GPU_DEVICE_ID}" if settings.GPU_ENABLED and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_engine(eng: Engine) -> None:
    t0 = time.perf_counter()
    try:
        from retrieval.frame_index import FrameIndex
        from retrieval.thumbs import ThumbService

        eng.index = FrameIndex.load(settings.INDEX_DIR)
        logger.info(f"Index: {eng.index.n:,} frames / {len(eng.index.video_ids):,} videos on {eng.index.device}")

        from embedding.pecore_text import PECoreText
        eng.text = PECoreText()
        if eng.text.dim != eng.index.dim:
            raise RuntimeError(f"Số chiều không khớp: text tower {eng.text.dim} vs index {eng.index.dim}")

        if settings.VI_EN_TRANSLATION_ENABLED:
            from embedding.translate_fast import make_translator
            eng.translator = make_translator(
                device=_pick_translate_device(), num_beams=settings.TRANSLATE_NUM_BEAMS,
                max_tokens=settings.TRANSLATE_MAX_TOKENS, model_name=settings.VI_EN_TRANSLATION_MODEL,
                model_dir=settings.VI_EN_TRANSLATION_MODEL_DIR)
            if not eng.translator.preload():
                logger.warning("Bộ dịch không nạp được: truy vấn tiếng Việt sẽ được gửi nguyên văn")
                eng.translator = None

        dirs = [d for d in str(settings.KEYFRAME_DIRS).split(";") if d.strip()]
        eng.thumbs = ThumbService(dirs, settings.VIDEO_DIR, settings.THUMB_WIDTH, settings.THUMB_CACHE_ITEMS, index=eng.index)

        eng.text.warmup()
        if eng.translator is not None:
            eng.translator.translate_text("một người đàn ông đang đi bộ trên đường")
        eng.ready = True
        eng.load_s = round(time.perf_counter() - t0, 1)
        logger.info(f"Engine ready in {eng.load_s}s")
    except Exception as e:                                   # keep the server up so the UI can show why
        logger.opt(exception=True).error(f"Engine failed to load: {e}")
        eng.error = str(e)


# ------------------------------------------------------------------------ requests


class SearchReq(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=60, ge=1, le=300)
    en_override: Optional[str] = Field(default=None, max_length=4000)
    video_filter: Optional[List[str]] = None


class TrakeReq(BaseModel):
    events: List[str] = Field(min_length=1, max_length=12)
    top_k: int = Field(default=20, ge=1, le=100)


class DresLoginReq(BaseModel):
    username: str = ""
    password: str = ""
    base_url: Optional[str] = None


class DresSessionReq(BaseModel):
    session_id: str = ""
    base_url: Optional[str] = None


class DresSelectReq(BaseModel):
    evaluation_id: str


class DresCollectionReq(BaseModel):
    collection: str = Field(default="", max_length=100)


class DresSubmitReq(BaseModel):
    kind: str                                   # "kis" | "qa" | "trake"
    video_id: str
    start_ms: Optional[float] = None
    end_ms: Optional[float] = None
    answer: Optional[str] = None
    time_ms: Optional[float] = None
    frames: Optional[List[float]] = None
    force: bool = False


# --------------------------------------------------------------------------- app


def create_app(engine: Optional[Engine] = None, dres: Optional[DresClient] = None, autoload: bool = True) -> FastAPI:
    eng = engine or Engine()
    real_dres = dres is None
    client = dres or DresClient(settings.DRES_BASE_URL, timeout=settings.DRES_TIMEOUT_SEC, log_path=settings.DRES_LOG_PATH,
                                state_path=settings.DRES_STATE_PATH)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if autoload and engine is None:
            # load in the background so the port opens at once and the UI can say "loading"
            threading.Thread(target=load_engine, args=(eng,), name="engine-load", daemon=True).start()
        boot = None
        if real_dres and autoload:
            async def dres_boot():
                await client.startup(settings.DRES_USERNAME, settings.DRES_PASSWORD)     # resume the saved session
                await client.keepalive(settings.DRES_KEEPALIVE_SEC)
            boot = asyncio.create_task(dres_boot())
        yield
        if boot is not None:
            boot.cancel()
        await client.close()
        if eng.thumbs is not None:
            eng.thumbs.close()

    app = FastAPI(title="AIC 2026 retrieval", version="3.0", lifespan=lifespan, docs_url="/docs", redoc_url=None)
    app.state.engine = eng
    app.state.dres = client

    def need_ready() -> Engine:
        if eng.error:
            raise HTTPException(503, f"Engine lỗi khi nạp: {eng.error}")
        if not eng.ready:
            raise HTTPException(503, "Server đang nạp mô hình / index, đợi vài chục giây")
        return eng

    # ---- security for the endpoints that can act as the team on DRES ---------------------
    async def dres_guard(request: Request) -> None:
        host = request.client.host if request.client else ""
        if not settings.DRES_ALLOW_REMOTE and host not in ("127.0.0.1", "::1", "localhost", "testclient"):
            raise HTTPException(403, "Endpoint DRES chỉ dùng được từ chính máy này")
        if request.method == "POST" and not request.headers.get("content-type", "").startswith("application/json"):
            raise HTTPException(415, "Cần Content-Type: application/json")
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host", ""):
            raise HTTPException(403, "Origin không hợp lệ")

    def dres_error(e: DresError) -> JSONResponse:
        code = e.status if e.status and 400 <= e.status < 600 else 400
        return JSONResponse({"detail": str(e)}, status_code=code)

    # ---- UI --------------------------------------------------------------------------------
    ui_file = settings.PROJECT_ROOT / "frontend" / "index.html"

    @app.get("/")
    @app.get("/ui")
    def ui():
        if not ui_file.exists():
            raise HTTPException(404, "frontend/index.html không tồn tại")
        return FileResponse(ui_file, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/status")
    def status():
        idx = eng.index
        return {
            "ready": bool(eng.ready), "error": eng.error, "load_s": eng.load_s,
            "frames": idx.n if idx is not None else 0,
            "videos": len(idx.video_ids) if idx is not None else 0,
            "dim": idx.dim if idx is not None else None,
            "device": str(idx.device) if idx is not None else None,
            "translator": eng.translator is not None,
            "dres_base_url": client.base_url,
            "trake_frame_id_mode": settings.TRAKE_FRAME_ID_MODE,
        }

    # ---- text -> english -> embedding ----------------------------------------------------------
    def to_english(texts: List[str]) -> List[str]:
        if eng.translator is None:
            return list(texts)
        return eng.translator.translate_many(texts)

    def ms_since(t: float) -> float:
        return round((time.perf_counter() - t) * 1000, 2)

    # ---- KIS / Q&A ---------------------------------------------------------------------------
    @app.post("/api/search")
    def search(req: SearchReq):
        e = need_ready()
        t0 = time.perf_counter()
        en = req.en_override.strip() if req.en_override and req.en_override.strip() else to_english([req.query])[0]
        t1 = time.perf_counter()
        q = e.text.encode([en])[0]
        t2 = time.perf_counter()
        hits = e.index.top_frames(q, req.top_k, nms_sec=settings.RESULT_NMS_SEC, video_ids=req.video_filter)
        t3 = time.perf_counter()
        results = [{"video_id": h.video_id, "frame_idx": h.frame_idx, "ms": h.ms, "fps": h.fps, "score": round(h.score, 4)}
                   for h in hits]
        if e.thumbs is not None:
            e.thumbs.prefetch((h.video_id, h.frame_idx) for h in hits)
        return {
            "query": req.query, "query_en": en, "total": len(results), "results": results,
            "latency_ms": ms_since(t0),
            "timings": {"translate_ms": round((t1 - t0) * 1000, 2), "encode_ms": round((t2 - t1) * 1000, 2),
                        "search_ms": round((t3 - t2) * 1000, 2)},
        }

    # ---- TRAKE -------------------------------------------------------------------------------
    @app.post("/api/trake")
    def trake(req: TrakeReq):
        from retrieval.trake_engine import rank_trake
        e = need_ready()
        events = [x.strip() for x in req.events if x and x.strip()]
        if not events:
            raise HTTPException(422, "Chưa có sự kiện nào")
        t0 = time.perf_counter()
        en = to_english(events)
        t1 = time.perf_counter()
        Q = e.text.encode(en)
        t2 = time.perf_counter()
        vids = rank_trake(e.index, Q, events, top_k=req.top_k, shortlist=settings.TRAKE_SHORTLIST,
                          alt_n=settings.TRAKE_ALT_FRAMES, alt_min_sec=settings.TRAKE_ALT_MIN_SEC)
        t3 = time.perf_counter()
        results = []
        for v in vids:
            frames = [{"event": events[i], **f, "score": round(f["score"], 4),
                       "alts": [{**a, "score": round(a["score"], 4)} for a in f["alts"]]}
                      for i, f in enumerate(v.frames)]
            results.append({"video_id": v.video_id, "fps": v.fps, "score": round(v.score, 4),
                            "stage_a": round(v.stage_a, 4), "frames": frames})
        if e.thumbs is not None:
            e.thumbs.prefetch((r["video_id"], f["frame_idx"]) for r in results[:5] for f in r["frames"])
        return {
            "events": events, "events_en": en, "results": results, "latency_ms": ms_since(t0),
            "timings": {"translate_ms": round((t1 - t0) * 1000, 2), "encode_ms": round((t2 - t1) * 1000, 2),
                        "search_ms": round((t3 - t2) * 1000, 2)},
        }

    # ---- frames / video -----------------------------------------------------------------------
    @app.get("/api/thumb/{video_id}/{frame_idx}")
    def thumb(video_id: str, frame_idx: int, w: int = 0):
        if eng.thumbs is None:
            raise HTTPException(503, "Chưa sẵn sàng")
        try:
            data = eng.thumbs.get(video_id, frame_idx, width=(min(max(int(w), 64), 1920) if w else None))
        except Exception as e:                # never a 500 for a missing/unreadable image: the UI shows a placeholder
            logger.warning(f"thumb {video_id}@{frame_idx}: {type(e).__name__}: {e}")
            data = None
        if data is None:
            raise HTTPException(404, "Không có ảnh")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/neighbors/{video_id}/{frame_idx}")
    def neighbors(video_id: str, frame_idx: int, n: int = 8):
        e = need_ready()
        if not e.index.has_video(video_id):
            raise HTTPException(404, f"Video không có trong index: {video_id}")
        n = max(1, min(int(n), 60))
        v = e.index.video_index(video_id)
        return {"fps": float(e.index.fps[v]), "items": e.index.neighbors(video_id, frame_idx, n)}

    @app.get("/api/video/{video_id}")
    def video(video_id: str):
        if eng.thumbs is None:
            raise HTTPException(503, "Chưa sẵn sàng")
        p = eng.thumbs.video_path(video_id)
        if p is None:
            raise HTTPException(404, f"Không thấy file video: {video_id}")
        return FileResponse(p, media_type="video/mp4", headers={"Cache-Control": "no-store"})

    # ---- DRES ---------------------------------------------------------------------------------
    @app.get("/api/dres/state", dependencies=[Depends(dres_guard)])
    def dres_state():
        return client.state()

    @app.post("/api/dres/login", dependencies=[Depends(dres_guard)])
    async def dres_login(req: DresLoginReq):
        try:
            if req.base_url:
                client.set_base_url(req.base_url)
            return await client.login(req.username, req.password)
        except DresError as e:
            return dres_error(e)

    @app.post("/api/dres/session", dependencies=[Depends(dres_guard)])
    async def dres_session(req: DresSessionReq):
        try:
            if req.base_url:
                client.set_base_url(req.base_url)
            return await client.use_session(req.session_id)
        except DresError as e:
            return dres_error(e)

    @app.post("/api/dres/refresh", dependencies=[Depends(dres_guard)])
    async def dres_refresh():
        try:
            await client.refresh_evaluations()
            return client.state()
        except DresError as e:
            return dres_error(e)

    @app.post("/api/dres/select", dependencies=[Depends(dres_guard)])
    async def dres_select(req: DresSelectReq):
        try:
            client.select_evaluation(req.evaluation_id)
            return client.state()
        except DresError as e:
            return dres_error(e)

    @app.post("/api/dres/collection", dependencies=[Depends(dres_guard)])
    async def dres_collection(req: DresCollectionReq):
        client.collection = req.collection.strip()
        return client.state()

    @app.post("/api/dres/logout", dependencies=[Depends(dres_guard)])
    async def dres_logout():
        client.logout()
        return client.state()

    @app.post("/api/dres/new_query", dependencies=[Depends(dres_guard)])
    async def dres_new_query():
        return {"query_epoch": client.new_query()}

    @app.post("/api/dres/submit", dependencies=[Depends(dres_guard)])
    async def dres_submit(req: DresSubmitReq):
        try:
            if req.kind == "kis":
                if req.start_ms is None:
                    raise DresError("Thiếu start_ms")
                payload = build_kis(req.video_id, req.start_ms, req.end_ms, collection=client.collection or None)
            elif req.kind == "qa":
                if req.time_ms is None:
                    raise DresError("Thiếu time_ms")
                payload = build_qa(req.answer or "", req.video_id, req.time_ms)
            elif req.kind == "trake":
                payload = build_trake(req.video_id, req.frames or [])
            else:
                raise DresError(f"kind không hợp lệ: {req.kind!r}")
            return await client.submit(payload, force=req.force)
        except DresError as e:
            return dres_error(e)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host=settings.API_HOST, port=settings.API_PORT, log_level="info")
