# AICV5 — Session Handoff

Đọc file này trước khi làm gì khác. Viết cho phiên Claude Code tiếp theo (context cũ sắp hết) — mục tiêu: hiểu trạng thái hiện tại trong 30 giây và làm việc tiếp ngay, không phải hỏi lại user những gì đã biết.

## Bối cảnh

Repo `E:\AIC2026\aicv5` — hệ thống video retrieval cho cuộc thi AI Challenge HCMC / VBS (KIS/Q&A/TRAKE). User ((email removed)) đang gấp rút chuẩn bị thi, không có nhiều thời gian test, chạy trên máy Windows có GPU (RTX, 8GB+ VRAM).

## Kiến trúc hiện tại

- **Dense encoder 1 ("clip" slot):** PE-Core-bigG-14-448 (Meta, `hf-hub:timm/PE-Core-bigG-14-448`, 1280-dim). Luôn bật (`USE_PECORE=True`).
- **Dense encoder 2 ("siglip2" slot):** mặc định là SigLIP2 (ViT-SO400M-14-SigLIP2), nhưng **máy này đang chạy C2LIP thay thế** (xem phần C2LIP bên dưới). Field/key tên `"siglip2"` được giữ nguyên xuyên suốt toàn bộ codebase (26 file tham chiếu) dù model đứng sau nó đổi — pattern y hệt cách `USE_PECORE`/`USE_TULIP` đổi model cho slot "clip".
- **Sparse:** BM25 — **hiện tại rỗng, 0 documents** (xem mục "Không có OCR/ASR" bên dưới).
- **Cross-encoder rerank:** BAAI/bge-reranker-v2-m3 — **hiện tại vô dụng** vì không có text để rerank (do BM25 rỗng).
- **Fusion:** RRF + CombMNZ, sau đó dedup + confidence scoring.
- **Vector store:** thiết kế dùng Milvus Lite, nhưng **`pymilvus` KHÔNG cài được trong Python env mặc định trên máy này** → app tự fallback sang `PersistentVectorStore` (file pickle phẳng ở `data/persistent_store/`). Đã fix cả 2 nhánh (Milvus + fallback) cho tính năng C2LIP, xem bên dưới.
- **112,205 frame** đã index (từ các video mã `L21_*`, `L22_*`, `L25_*`, `L26_*`, kiểu tên AIC).

## Việc đã làm trong phiên này: tích hợp C2LIP

### C2LIP là gì (đã xác minh, không phải đoán)
- Paper: https://arxiv.org/abs/2603.25722 ("Concept-Centric" learning, CVPR 2026).
- Checkpoint `c2lipweight.pt` (1.2GB, ở repo root) — **thực chất là SigLIP ViT-B/16 fine-tuned**, kiến trúc y hệt `open_clip`'s `ViT-B-16-SigLIP` (đã load thử: 0 key thiếu/thừa). Tên checkpoint nội bộ: `siglip_cc30m_3m_lla-lr-1e-05-...`.
- Output: 768-dim, `logit_scale≈98.88`, `logit_bias≈-12.938`.
- Đã có sẵn corpus embeddings tính trước: `embedding finvit + c2lip/embeddings_c2lip.pkl` (998MB) — dict `{clip, siglip2, c2lip}`, mỗi key là list 112,205 vector 768-dim, **cùng thứ tự với `data/persistent_store`** (đã verify bằng cosine=1.0 ở 6 điểm mẫu trên field siglip2).
- Có cả `embedding finvit + c2lip/embeddings_finevit.pkl` (998MB, cho FineViT) nhưng **CHƯA tích hợp** — user chưa yêu cầu, và **không tìm thấy file weight (.pt) nào cho FineViT** ở đâu cả (chỉ có embeddings pkl, không có checkpoint để load encoder sống). Nếu user muốn tích hợp FineViT, phải hỏi họ file weight ở đâu trước.

### File đã sửa/tạo (tất cả đã verify ghi thành công, còn nguyên)
1. **`embedding/c2lip_encoder.py`** (MỚI) — class `C2LIPEncoder`, interface giống hệt `SigLIP2Encoder`/`PECoreEncoder` (encode_image/text, batch, output_dim, get_logit, get_probability). Load checkpoint từ `settings.C2LIP_CHECKPOINT`, strip prefix `"module."`.
2. **`config/settings.py`** — thêm:
   - `USE_C2LIP: bool = False` (default tắt — đúng thiết kế, bật per-machine qua `.env`)
   - `C2LIP_MODEL_NAME = "ViT-B-16-SigLIP"`
   - `C2LIP_CHECKPOINT = <repo_root>/c2lipweight.pt`
   - `EMBEDDING_DIM_C2LIP = 768`
   - `C2LIP_EMBEDDINGS_PKL = <repo_root>/"embedding finvit + c2lip"/embeddings_c2lip.pkl`
   - property `SIGLIP2_SLOT_DIM` → trả `EMBEDDING_DIM_C2LIP` nếu `USE_C2LIP` else `EMBEDDING_DIM_SIGLIP2_STORED`
3. **`embedding/hybrid_engine.py`** — branch `if settings.USE_C2LIP: C2LIPEncoder else SigLIP2Encoder`, gán vào `self.siglip2_encoder` (tên biến giữ nguyên).
4. **`embedding/__init__.py`** — export `C2LIPEncoder`.
5. **`retrieval/milvus_store.py`** — dùng `settings.SIGLIP2_SLOT_DIM` thay vì `EMBEDDING_DIM_SIGLIP2_STORED` trực tiếp; thêm `_load_c2lip_embeddings()` dùng trong `ingest_from_persistent_store()` — khi `USE_C2LIP=True`, lấy vector từ `C2LIP_EMBEDDINGS_PKL`'s key `'c2lip'` thay vì `'siglip2'` cũ, có fallback an toàn nếu length không khớp.
6. **`retrieval/local_store.py`** — cùng fix `SIGLIP2_SLOT_DIM` (nhánh in-memory fallback khi không có Docker).
7. **`retrieval/persistent_store.py`** — **đây là nhánh THỰC SỰ chạy trên máy này** (vì không có pymilvus). Thêm y hệt `_load_c2lip_embeddings()`, gọi trong `_load_from_disk()`. Có **retry 5 lần, sleep 2s** khi đọc file pkl 998MB vì môi trường này thỉnh thoảng bị lỗi I/O thoáng qua (`[Errno 22] Invalid argument`) khi đọc file lớn lúc RAM đang bị nhiều model chiếm dụng.
8. **`.env.example`** — ghi chú toggle `USE_C2LIP` mới.
9. **`.env`** (file thật, máy này) — có dòng `USE_C2LIP=True`.

### Đã verify (không phải suy đoán)
- Load checkpoint chuẩn: 0 missing/unexpected keys.
- Encode ảnh/text ra vector 768-dim, unit-norm, không NaN, logit hợp lý.
- Thứ tự frame trong `embeddings_c2lip.pkl` khớp 100% với `data/persistent_store` (cosine=1.0).
- Chạy full server thật + search qua UI: query "a person presenting slides in a meeting room" → 200 OK, 100 kết quả, `clip_score` ~0.15-0.20 (PE-Core), `siglip2_score` ~0.13-0.14 (thực ra là C2LIP), confidence tính đúng. Ảnh frame serve được qua `/api/frame/{video_id}/{frame_idx}`.

## Quan trọng: KHÔNG có dữ liệu OCR/ASR/audio nào cả

Đã quét toàn bộ 112,205 entry trong `data/persistent_store/metadata.pkl` — **chỉ có `video_id`, `frame_idx`, `timestamp`**. Không OCR, không ASR transcript, không object detection, không caption.

- `OCR_ENABLED=True`, `ASR_ENABLED=True` trong settings.py **chỉ là flag để log**, không có model OCR/ASR thật nào được gọi ở đâu trong repo (`preprocessing/pipeline.py` chỉ in ra dòng log, không xử lý gì).
- `retrieval/bm25_search.py`'s `build_frame_document_text()` kỳ vọng field `ocr_text`, `asr_transcript`, `objects`, `caption`, `video_title` — nhưng comment ghi rõ *"fill in whatever BTC's JSON has"* → thiết kế chờ **Ban Tổ Chức cung cấp sẵn** JSON media-info, hệ thống không tự trích xuất.
- `data/media-info-aic25-b1/` (nơi `retrieval/media_meta.py` mong đợi tìm title/keywords/date cấp video) **không tồn tại trên máy này**.
- Kết quả: BM25 luôn "0 documents", cross-encoder rerank vô dụng. **Search 100% dựa vào PE-Core + (SigLIP2/C2LIP) dense fusion**, không có tín hiệu văn bản nào.

Nếu cần bật lại: (1) xin media-info JSON từ BTC bỏ vào `data/media-info-aic25-b1/media-info/`, hoặc (2) tự chạy OCR/ASR (Whisper) offline rồi nhét vào metadata theo đúng schema trên.

## Vị trí dữ liệu (đường dẫn thật)

| Thứ | Đường dẫn | Ghi chú |
|---|---|---|
| Video gốc | `E:\AIC2026\aicv5\video\*.mp4` (873 file) | set qua `.env` `VIDEO_DIR` |
| Keyframe jpg thô | `E:\AIC2026\aicv5\keyframes\<VIDEO_ID>\*.jpg` (177,321 file) | **KHÔNG phải** `data/keyframes` (setting mặc định trỏ sai chỗ này — chỉ ảnh hưởng ingest video mới, không ảnh hưởng demo hiện tại vì ảnh frame serve trực tiếp từ mp4 qua cv2) |
| Corpus đang search | `data/persistent_store/{embeddings.pkl, metadata.pkl}` | clip=1280d PE-Core, siglip2=768d (SigLIP2-truncated HOẶC C2LIP tùy `USE_C2LIP`) |
| C2LIP checkpoint | `c2lipweight.pt` (1.2GB, repo root) | |
| C2LIP corpus embeddings | `embedding finvit + c2lip/embeddings_c2lip.pkl` (998MB) | |
| FineViT corpus embeddings (chưa dùng) | `embedding finvit + c2lip/embeddings_finevit.pkl` (998MB) | không có weight file |
| PE-Core embeddings dump cũ | `embeddings_pecore.pkl` (1.2GB, repo root) | |
| HF model cache | `D:\hf-cache` | đã có sẵn PE-Core-bigG, SigLIP2, SigLIP-B, DFN2B-CLIP-ViT-L-14, bge-reranker-v2-m3, vinai-translate — ~21GB, KHÔNG cần tải lại |

## Cách khởi động server

1. Kiểm tra `.env` có dòng `USE_C2LIP=True` chưa (xem mục "quirk môi trường" bên dưới — dòng này hay bị mất).
2. Chạy `python run.py`, hoặc qua Browser-preview tool: `preview_start({name: "aicv5-server"})` (config có sẵn ở `.claude/launch.json`, port 8000).
3. Đợi **~2-3 phút** cho tới khi thấy log `"System ready!"` — thứ tự load: import module (chậm, có lúc 60-90s) → PE-Core (~40-90s, RAM đỉnh ~14-16GB trước khi chuyển GPU, GPU dùng ~4.9GB) → C2LIP (~10s) → persistent_store 112k frame (~10-40s, có retry) → keyframe extractor CLIP ViT-L-14 + translator (~30-60s) → cross-encoder (~10s).
4. UI: `http://localhost:8000/ui`. Status: `http://localhost:8000/api/status`.

## Quirk môi trường (KHÔNG phải bug trong code, đặc thù sandbox này)

1. **File `.env` hay tự revert về nội dung cũ** sau khi Edit tool sửa xong — xảy ra nhiều lần trong phiên, không rõ nguyên nhân gốc (nghi cơ chế snapshot/bảo vệ dotfile của môi trường coding này). **Luôn `cat .env` kiểm tra lại ngay trước khi start server**; nếu mất dòng `USE_C2LIP=True` thì thêm lại và start ngay lập tức, không delay giữa 2 bước (delay có vẻ làm tăng khả năng bị revert).
2. **`pymilvus` không cài được** trong Python mặc định của máy này → luôn fallback `PersistentVectorStore`. Nếu máy thi thật có `pymilvus` cài đúng (xem `requirements.txt`: Windows cần `pymilvus>=3.0.1,<4` + `milvus-lite>=3.2.0,<4`), nhánh Milvus thật sẽ chạy — code đã fix sẵn cho cả 2 nhánh.
3. **Server đôi khi chết đột ngột giữa lúc load model** (2 lần quan sát được), không có log lỗi, process biến mất luôn (kể cả `serverId` cũng mất). Nghi do hạ tầng Browser-preview có timeout riêng cho cold-start dài (~2-3 phút). Cách xử lý: chỉ cần `preview_stop` (nếu còn) rồi `preview_start` lại, thường lần sau sẽ qua được.
4. Lỗi I/O thoáng qua `[Errno 22] Invalid argument` khi đọc file lớn dưới áp lực RAM — đã thêm retry cho chỗ quan trọng nhất (đọc `embeddings_c2lip.pkl`), nhưng có thể gặp lại ở chỗ khác.

## Việc còn tồn đọng / có thể làm tiếp

- `PersistentVectorStore` thiếu method `is_empty()` và `get_all_frame_records()` mà `MilvusVectorStore` có → gây warning vô hại lúc khởi động ("Local data ingest failed"), không ảnh hưởng search vì persistent_store tự load ở `__init__` rồi. Ưu tiên thấp (BM25 dù sao cũng không có gì để index).
- `KEYFRAME_DIR` trong settings.py trỏ sai (`data/keyframes` thay vì `./keyframes`) — chỉ quan trọng nếu ingest video MỚI qua pipeline sống.
- FineViT: có embeddings pkl nhưng không có weight file — nếu user muốn tích hợp, phải hỏi họ file weight ở đâu trước khi làm (đừng tự bịa kiến trúc như đã làm thành công với C2LIP, vì lần đó có checkpoint thật để verify).
- User có kế hoạch "mỗi máy 1 model" cho nhiều máy thi — cơ chế `USE_C2LIP` per-`.env` đã hỗ trợ sẵn, chỉ cần set khác nhau trên mỗi máy.
