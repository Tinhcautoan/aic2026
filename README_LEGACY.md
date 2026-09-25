**Team ** — 1st Place, High-School Division

Video retrieval for the AIC challenge (KIS / Q&A / TRAKE) using PE-Core + SigLIP2 dual-encoder dense search, BM25 sparse search, cross-encoder reranking, and a calibrated confidence layer.

---

## What This System Does

1. **Ingests a video** → extracts keyframes via feature-difference scene detection
2. **Generates embeddings** for each keyframe using two dense encoders:
   - **PE-Core** (Meta, `hf-hub:timm/PE-Core-bigG-14-448`, 1280-dim) — primary visual-semantic encoder
   - **SigLIP2** (ViT-SO400M-14, 1152-dim native) — second, independently-trained encoder for fusion diversity
3. **Stores embeddings** in Milvus Lite (file-based, no Docker/server needed) — falls back automatically to an in-process `PersistentVectorStore` if Milvus Lite can't initialize
4. **Searches** in stages: dense fusion (RRF + CombMNZ over PE-Core/SigLIP2) → BM25 over OCR/ASR/object text → cross-encoder rerank (BAAI/bge-reranker-v2-m3) → frame-neighbor expansion → dedup → confidence scoring
5. **Serves results** through a web UI (`/ui`) with thumbnails, timestamps, and a calibrated confidence score (separate from the ranking score — see `retrieval/scoring.py` for why that split exists)

Supports three query modes: **KIS** (find the matching frame from a text/image description), **Q&A** (KIS + VLM-generated answer), **TRAKE** (align a sequence of events in temporal order via DP, `retrieval/trake_aligner.py`).

---

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** — PE-Core-bigG alone needs ~4.9GB VRAM at fp16; add SigLIP2 (~1GB), the keyframe-extraction CLIP model (~0.6GB), and the cross-encoder (~1.5GB) and you're comfortably fine on **8GB+ VRAM**. Runs fine at that budget on an RTX 4060 laptop (8GB) — verified this session. CPU-only works but is slow.
- **~20GB free disk** for model weights on first run (PE-Core ~9.7GB, SigLIP2 ~4-5GB, translator + cross-encoder ~3GB). See [HF_HOME](#disk-space--model-cache) below if your system drive is tight.
- Video files (`.mp4` etc.) for the videos you want to index.

---

## Setup

### 1. Install dependencies

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` already pins the CUDA 12.6 build of torch via `--extra-index-url`. If you're on CPU-only, edit out the `+cu126` version suffixes for `torch`/`torchvision`/`torchaudio` first.

### 2. Configure

```cmd
copy .env.example .env
```

Defaults work out of the box (PE-Core + SigLIP2 native, Milvus Lite storage). See [`.env.example`](.env.example) for what's tunable — model cache directories, VLM (optional, for Q&A answers), contest metadata paths.

### 3. Disk space / model cache

Model weights auto-download from HuggingFace Hub on first run and default into `./hf-cache` (project-relative, see `run.py`) so they can't silently fill a small system drive. Point `HF_HOME` at a different drive in `.env` if you'd rather reuse an existing cache or have more room elsewhere:

```
HF_HOME=D:\hf-cache
```

### 4. Add your videos and generate embeddings

Drop your `.mp4` files anywhere and run the batch setup script — it copies them into `data/videos/`, extracts keyframes, and generates PE-Core + SigLIP2 embeddings for all of them:

```cmd
python scripts/setup_local.py --videos-dir C:\path\to\your\aic\videos
```

This can take a while for a full contest dataset (hundreds of videos) — keyframe extraction and embedding are both GPU-bound. Progress prints per-video. Re-running is safe: it skips videos/keyframes/embeddings that already exist.

If you're only testing with one or two videos, you can instead use the CLI directly:
```cmd
python main.py process path\to\video.mp4 --video-id my_video --save-keyframes
```

### 5. Start the server

```cmd
python run.py
```

First boot downloads and loads all models (~5-10 min depending on connection). You'll see each stage log as it completes, ending with:
```
System ready!
Uvicorn running on http://0.0.0.0:8000
```

Subsequent restarts are fast (~1 min) since models are cached.

### 6. Search

Open **http://localhost:8000/ui** — pick a mode (TKIS / Q&A / TRAKE), type a query, hit Search.

---

## Using the System

### Web UI
| URL | Purpose |
|-----|---------|
| http://localhost:8000/ui | Search interface — TKIS/Q&A/TRAKE, thumbnails, confidence scores |
| http://localhost:8000/process | Upload/process a video from the browser |
| http://localhost:8000/docs | Swagger API docs |
| http://localhost:8000/api/status | System health & frame count |

### Key API Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/search` | Text/fusion search (KIS) |
| `POST` | `/api/search-image` | Search by uploaded image |
| `POST` | `/api/qa/search` | KIS + VLM answer (Q&A mode) |
| `POST` | `/api/trake` | Multi-event temporal alignment |
| `GET` | `/api/frame/{video_id}/{frame_idx}` | Serve a frame as JPEG (seeks the source video live) |
| `GET` | `/api/video/{video_id}` | Stream the source video for playback |

---

## Project Structure

```
aicv5/
├── run.py                          # Entry point (Milvus Lite / PersistentVectorStore, no Docker needed)
├── main.py                         # CLI: serve / process / search / evaluate / status
├── requirements.txt
│
├── config/settings.py              # All tunables
│
├── embedding/                      # Dual dense-encoder engine
│   ├── pecore_encoder.py           # PE-Core-bigG-14-448 (primary, 1280d)
│   ├── siglip2_encoder.py          # SigLIP2 (1152d native / pca / truncate modes)
│   ├── clip_encoder.py, tulip_encoder.py   # Alternate "clip slot" encoders (USE_PECORE=False to switch)
│   └── hybrid_engine.py            # Wires the active encoder pair together
│
├── retrieval/                      # Storage, fusion, reranking, confidence
│   ├── milvus_store.py             # Milvus Lite vector store (default)
│   ├── persistent_store.py         # File-based numpy fallback, no server needed
│   ├── local_store.py              # In-memory numpy store (ephemeral, tests)
│   ├── reranker.py                 # Stage 1: RRF + CombMNZ dense fusion
│   ├── bm25_search.py              # Stage 2: sparse text search (OCR/ASR/object/media-info)
│   ├── cross_encoder_reranker.py   # Stage 4: BGE-reranker text rerank
│   ├── trake_aligner.py            # DP-based multi-event temporal alignment
│   ├── result_dedup.py             # Stage 6: NMS + scene-bucket dedup
│   ├── scoring.py                  # Stage 7: score banding + calibrated confidence
│   └── hybrid_fusion.py            # Orchestrates the full funnel
│
├── preprocessing/                  # Video ingestion
│   ├── keyframe_extractor.py       # Feature-diff scene detection
│   └── pipeline.py                 # Keyframes → OCR/ASR → embed → store
│
├── api/app.py                      # FastAPI backend, all endpoints + lifespan startup
├── frontend/index.html             # Web UI
├── scripts/setup_local.py          # Bulk video ingestion for a fresh checkout
│
└── data/                           # Not committed (see .gitignore) — created by setup
    ├── videos/                     # Your .mp4 files
    ├── keyframes/                  # Extracted frames (auto)
    └── persistent_store/           # Fallback embeddings store (auto, if Milvus Lite unavailable)
```

---

## Troubleshooting

### CUDA out of memory / crash on startup with no traceback
On tight VRAM budgets (8GB and under), loading a large fp32 checkpoint straight onto the GPU before converting to fp16 can exceed free memory and crash the process natively (no Python exception). The encoders already load CPU→fp16→GPU to avoid this (`embedding/pecore_encoder.py`, `preprocessing/keyframe_extractor.py`) — if you hit this again after modifying those files, check you didn't reintroduce a `device=cuda` load-time argument to `create_model_and_transforms`.

If you're still tight on VRAM, set `CROSS_ENCODER_ENABLED=False` in `.env` — Stage 4 becomes a no-op and search still works off dense fusion + BM25.

### Windows: process crashes under memory pressure, no error logged
Usually a near-full system drive — Windows needs pagefile headroom during large model loads. Check free space on your `HF_HOME` drive and the drive Windows itself is installed on.

### Search results have no thumbnails
The `/api/frame/{video_id}/{frame_idx}` endpoint seeks the *source video file* live (not a static keyframe image) — it needs your `.mp4` to be discoverable under `data/videos/` (or wherever `VIDEO_DIR` points). If videos aren't there yet, thumbnails 404 silently; ranking/search still works.

### BM25 / cross-encoder contribute nothing
Both stages need per-frame text (OCR, ASR transcript, detected objects, or contest media-info JSON) to have anything to match against. A purely visual frame with no text has nothing for these stages to score — dense search (PE-Core + SigLIP2) still ranks it normally. Check `pipeline.diagnostics()['cross_encoder']['text_hit_rate']` — below ~0.3 means the stage is mostly dead weight for your data.

### "Failed to connect to Milvus" / Milvus Lite init failed
Falls back automatically to `PersistentVectorStore` (file-based, `data/persistent_store/`). No action needed unless you specifically want Milvus's ANN indexing at larger scale.

### Port 8000 already in use
```cmd
taskkill /F /IM python.exe
```

---

## Key Formulas

- **Keyframe differencing**: `feature_diff = ||f_current - f_prev|| / ||f_prev||` (threshold = 0.6)
- **Dense fusion**: `Final(d) = α·RRF(d) + (1-α)·CombMNZ(d)`, α=0.5, RRF k=60 (`retrieval/reranker.py`)
- **Confidence** (separate from ranking, `retrieval/scoring.py`): `sigmoid(0.6·sep + 0.9·margin − 1.5) · rel_i^γ`, where `sep`/`margin` are the top candidate's separation from the pool in standard deviations — a flat, undiscriminating pool reports low confidence even for a perfect rank-1 match
- **TRAKE alignment**: `dp[t][j] = s_t(F[j]) + max{dp[t-1][k] − λ(F[j]−F[k]) : F[k] < F[j]}`, O(N·M) via prefix-max

---

## Credits

Vo Thanh Nhan, Doan Ngoc Binh Minh, Pi-Dieu Sam, Hai Dang Nguyen, Khoi-Nguyen Nguyen
VNU-HCM High School for the Gifted — Team
