# Chạy trên Google Colab (A100 40GB)

## Yêu cầu
- **Colab Pro** (A100 40GB GPU)
- **Google Drive** — lưu video + embeddings

## Chuẩn bị (trước khi mở Colab)

### 1. Upload video thi lên Drive
```
Google Drive/
└── aicv5/
    ├── videos/          ← upload toàn bộ video thi vào đây
    └── queries/         ← upload các file query .txt (kis/qa/trake)
```

### 2. Clone repo (hoặc upload zip)
```bash
git clone https://github.com/YOUR_USERNAME/aicv5.git
# hoặc upload zip → Drive/aicv5/
```

## Cách chạy (6 cells)

| Cell | Nội dung | Thời gian |
|------|----------|-----------|
| **1** | Install dependencies | ~2 phút |
| **2** | Mount Drive, symlink data | 30s |
| **3** | Keyframe extraction + embeddings | ~30 phút (lần đầu) |
| **4** | Start VLM server (Qwen2.5-VL-32B-AWQ) | ~5 phút (lần đầu download model) |
| **5** | Start API server + ngrok | 30s |
| **6** | Run contest → xuất `submission/*.csv` | ~30s/question |

**Lần chạy lại:** Chỉ cần chạy Cell 1-2, skip Cell 3 (data đã có trên Drive), Cell 4 load model nhanh từ checkpoint.

## GPU Budget (A100 40GB)

| Model | VRAM | Notes |
|---|---|---|
| TULIP (ViT-L-14) | ~1.5GB | Retrieval embeddings |
| SigLIP2 (ViT-SO400M) | ~1.5GB | Retrieval embeddings |
| Cross-Encoder (bge-reranker-v2-m3) | ~1.5GB | Re-ranking |
| Qwen2.5-VL-32B-AWQ | ~20GB | VLM (4-bit quantized) |
| **Tổng** | ~24.5GB | Còn ~15GB cho overhead |

## Lưu ý quan trọng

### Model download lần đầu
- Qwen2.5-VL-32B-AWQ: ~16GB download, ~30 phút
- TULIP + SigLIP2: ~2GB download, ~5 phút
- **Lưu checkpoints vào Drive** để lần sau load nhanh

### Colab timeout
- Colab Pro timeout ~24h
- Nếu batch chạy lâu → script tự động save intermediate results vào Drive
- Nếu session disconnect → chạy lại từ Cell 4 (model cached trên Drive)

### Troubleshooting
| Vấn đề | Giải pháp |
|---|---|
| `CUDA out of memory` | Giảm `gpu-memory-utilization` trong Cell 4 |
| `Model not found` | Kiểm tra Drive có đủ space, chạy Cell 3 lại |
| ngrok limit | Dùng `--addr 8000` trực tiếp nếu không cần public URL |
| Query file không đọc được | Kiểm tra encoding UTF-8, mở bằng Notepad++ |

## Format file query
```
queries/
├── query-1-kis.txt     ← TKIS: mô tả sự kiện
├── query-2-kis.txt     ← TKIS: mô tả sự kiện
├── query-3-qa.txt      ← Q&A: mô tả + câu hỏi
└── query-4-trake.txt   ← TRAKE: mô tả + (1) event1, (2) event2, ...
```

## Output
```
submission/
├── query-1-kis.csv     ← video,frame (top-100)
├── query-2-kis.csv
├── query-3-qa.csv      ← video,frame,"answer"
└── query-4-trake.csv   ← video,f1,f2,f3,...
```

Zip thư mục `submission/` → nộp lên Codabench.
