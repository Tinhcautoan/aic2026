**Team ** — 1st Place, High-School Division, 

Real-time video retrieval using CLIP+SigLIP2 fusion with RRF+CombMNZ reranking.

---

## What This System Does

1. **Ingests a video** → extracts keyframes using CLIP-based scene-change detection
2. **Generates embeddings** for each keyframe using two models:
   - CLIP (ViT-L-14, 1024-dim)
   - SigLIP2 (ViT-SO400M-14, 1152-dim → projected to 1024)
3. **Stores embeddings** in a vector database (Milvus or in-memory LocalVectorStore)
4. **Searches** your text query against stored embeddings using RRF + CombMNZ reranking
5. **Shows results** as ranked thumbnails with similarity scores

| Paper Metric | Target |
|-------------|--------|
| Recall@10 | 0.872 |
| MRR | 0.811 |
| Latency | <200ms (GPU) |

---

## Prerequisites

### Required:
- Python 3.10+
- NVIDIA GPU (or CPU-only — slower but works)
- Video files (`.mp4`, `.avi`, etc.)

### Optional (for persistence):
- Docker Desktop with WSL 2 — allows indexed data to survive restarts

---

## Option A: Quick Start (No Docker, In-Memory)

> **Good for**: testing, quick demos. Data is lost when you stop the server.

### Step 1: Create Virtual Environment & Install

```cmd
cd path\to\AIC

python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install PyTorch with CUDA support (for GPU)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

If you don't have a GPU, skip the last line — the `torch` from `requirements.txt` (CPU version) is fine.

### Step 2: Start the Server

```cmd
venv\Scripts\activate
python run.py
```

Wait ~60 seconds for models to load on GPU. You'll see:
```
CLIP encoder loaded: ViT-L-14 (dim=1024, fp16=True)
SigLIP2 encoder loaded: ViT-SO400M-14-SigLIP2
System ready!
Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Open the Web Interface

Open your browser to: **http://localhost:8000/ui**

### Step 4: Process a Video

1. Put your `.mp4` file in the `data/videos/` folder
2. Go to **http://localhost:8000/docs** (Swagger API docs)
3. Find `POST /api/process` → click **"Try it out"**
4. Replace the body with:
```json
{
  "video_path": "data/videos/your_video.mp4",
  "video_id": "my_video"
}
```
5. Click **Execute** — wait a few minutes (processing time depends on video length)
   - A ~90 MB, 30 FPS video takes ~3–5 minutes on RTX 3050
   - Keyframe extraction: ~1 min
   - Embedding generation: ~2 min

### Step 5: Search

Go to **http://localhost:8000/ui** and type your query (e.g., "pink car", "person dancing").

### Step 6: Stop the Server

Press `Ctrl+C` in the terminal.

---

## Option B: Full Setup (Docker + Persistent Storage)

> **Good for**: long-running deployments. Data persists even after restart.

### Prerequisites
- Docker Desktop installed and running (whale icon in system tray must be solid)
- WSL 2 installed (check with `wsl --version` in PowerShell)

### Step 1: Start Docker Services

```cmd
cd path\to\AIC
docker-compose up -d
```

This starts: Milvus (vector DB), Redis (cache), and the FastAPI server.

### Step 2: Check Status

```cmd
docker-compose ps
```

Wait until all services show `healthy` (~2 minutes first time).

### Step 3: Process & Search

Same as **Option A Steps 3-5** — the API auto-detects Milvus and uses it.

### Step 4: Stop

```cmd
docker-compose down
```

---

## Using the System

### Web UI
| URL | Purpose |
|-----|---------|
| http://localhost:8000/ui | Search interface with result grid |
| http://localhost:8000/docs | Swagger API docs (Try it out) |
| http://localhost:8000/api/status | System health & stats |

### API Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/process` | Index a video |
| `POST` | `/api/search` | Search with text query |
| `GET` | `/api/status` | Health check |
| `GET` | `/api/frame/{video_id}/{frame_idx}` | Serve frame image |
| `DELETE` | `/api/videos/{video_id}` | Remove video from index |

### Search Models
- `fusion` — CLIP + SigLIP2 with RRF+CombMNZ (default, best accuracy)
- `clip` — CLIP only
- `siglip2` — SigLIP2 only

---

## Project Structure

```
AIC/
├── run.py                          # Main entry point
├── requirements.txt                # Python packages
├── docker-compose.yml              # Docker services
├── Dockerfile                      # Docker image for API
├── .env                            # Configuration
│
├── config/settings.py              # All parameters
│
├── embedding/                      # Multimodal Embedding Engine
│   ├── clip_encoder.py             # CLIP ViT-L-14 (1024-dim)
│   ├── siglip2_encoder.py          # SigLIP2 native 1152-dim (no projection)
│   └── hybrid_engine.py            # Async dual-encoder pipeline
│
├── retrieval/                      # Vector storage & reranking
│   ├── milvus_store.py             # Milvus vector DB
│   ├── local_store.py              # In-memory numpy store
│   ├── redis_cache.py              # Redis caching layer
│   └── reranker.py                 # RRF + CombMNZ fusion
│
├── preprocessing/                  # Video ingestion
│   ├── keyframe_extractor.py       # CLIP feature-difference extraction
│   └── pipeline.py                 # Full pipeline: KF→Embed→Index
│
├── api/                            # FastAPI backend
│   ├── app.py                      # All endpoints
│   └── models.py                   # Pydantic models
│
├── evaluation/                     # Evaluation suite
│   └── metrics.py                  # Recall@k, MRR, 5-repeat protocol
│
├── frontend/index.html             # Web UI
│
└── data/
    ├── videos/                     # Put your MP4 files here
    ├── keyframes/                  # Extracted frames (auto)
    └── cache/                      # Temp files
```

---

## Troubleshooting

### "Embedding engine not initialized"
- Torch or open_clip not installed. Re-run:
  ```cmd
  venv\Scripts\python.exe -m pip install open_clip_torch transformers torch torchvision --index-url https://download.pytorch.org/whl/cu126
  ```

### "Store not available. Process a video first."
- No videos indexed yet. Go to http://localhost:8000/docs → `POST /api/process`

### Search returns 0 results
- Option A: each restart wipes data. Process video again.
- Option B: video needs to be processed at least once via `/api/process`

### "Failed to connect to Milvus"
- Expected on Option A. The system falls back to LocalVectorStore automatically.
- On Option B: wait for `docker-compose ps` to show all services healthy.

### GPU out of memory
- `USE_FP16=true` in `.env` reduces memory (already enabled)
- Close other GPU apps (Chrome, Discord, etc.)

### Port 8000 already in use
```cmd
taskkill /F /IM python.exe
```

---

## Key Formulas (from the paper)

- **Keyframe differencing**: `feature_diff = ||f_current - f_prev|| / ||f_prev||` (threshold = 0.6)
- **Cosine similarity**: `S_m = (V_m · T_m) / (||V_m || ||T_m||)`
- **RRF**: `Σ 1/(k + rank_m)` where k=60
- **CombMNZ**: `(Σ ŝ_m(d)) × count(ŝ_m(d) > 0)` (z-score normalized)
- **Final fusion**: `Final(d) = α × RRF(d) + (1-α) × CombMNZ(d)` where α=0.5

---

## Credits

Vo Thanh Nhan, Doan Ngoc Binh Minh, Pi-Dieu Sam, Hai Dang Nguyen, Khoi-Nguyen Nguyen  
VNU-HCM High School for the Gifted — Team
