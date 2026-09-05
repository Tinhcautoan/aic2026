# Hướng Dẫn Sử Dụng — AICV5

Dành cho người nhận code này lần đầu (đã có sẵn video + embedding riêng). Đọc README.md để biết chi tiết kỹ thuật/kiến trúc; file này chỉ tập trung vào "làm sao để chạy được với dữ liệu của bạn" và "cách dùng giao diện".

---

## 1. Hệ thống này làm gì

Video retrieval cho AI Challenge (KIS / Q&A / TRAKE): nhập mô tả (hoặc ảnh), tìm ra (các) frame khớp nhất trong kho video, xuất ra file nộp bài đúng định dạng BTC.

Pipeline: 2 dense encoder (PE-Core + 1 slot thứ 2 linh hoạt: SigLIP2 hoặc C2LIP tùy `.env`) → BM25 (nếu có OCR/ASR/media-info) → cross-encoder rerank → dedup → confidence.

---

## 2. Cài đặt lần đầu

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

Mở `.env` vừa tạo và sửa theo đúng máy của bạn — **đừng dùng chung `.env` với ai khác**, mỗi máy có đường dẫn riêng:

| Biến | Ý nghĩa |
|---|---|
| `VIDEO_DIR` | Trỏ tới thư mục chứa file `.mp4` của bạn |
| `HF_HOME` | Nơi cache model HuggingFace (nên để ổ đĩa rộng, không phải ổ C nếu ổ C ít chỗ) |
| `USE_C2LIP` | `True` nếu bạn có sẵn `c2lipweight.pt` + `embeddings_c2lip.pkl` riêng và muốn dùng C2LIP thay SigLIP2 (mặc định `False` = SigLIP2 chuẩn). **2 file này không có trong git** (quá nặng, ~1-2GB) — ai đưa code cho bạn thì xin trực tiếp file này qua Zalo/Drive nếu cần |

Xem đầy đủ các biến khác trong `.env.example` (đã chú thích sẵn).

## 3. Bạn đã có sẵn "không gian embedding" — dùng lại thế nào?

Corpus embeddings mà server load lúc khởi động nằm ở `data/persistent_store/{embeddings.pkl, metadata.pkl}`. Nếu bạn đã tính embeddings cho video của mình:
- Nếu **đúng định dạng này** (dict theo `video_id`/`frame_idx`/`clip_emb`/`siglip2_emb`) → copy thẳng 2 file vào `data/persistent_store/`, server tự nhận lúc start, không cần embed lại.
- Nếu khác định dạng, hoặc bạn có video mới chưa từng embed → chạy:
  ```cmd
  python scripts/setup_local.py --videos-dir C:\đường\dẫn\video\của\bạn
  ```
  Script tự copy video, tách keyframe, tính embedding PE-Core + SigLIP2/C2LIP, ghi vào `data/persistent_store/`. Chạy lại an toàn — bỏ qua video/frame đã có sẵn.

## 4. Khởi động server

```cmd
python run.py
```

Lần đầu load model mất **~2-5 phút** (tải model từ HuggingFace nếu chưa có trong `HF_HOME`, rồi load PE-Core + encoder slot 2 + cross-encoder). Chờ tới khi thấy dòng:
```
System ready!
Uvicorn running on http://0.0.0.0:8000
```
Các lần sau nhanh hơn (~1-2 phút) vì model đã cache.

Mở **http://localhost:8000/ui**.

---

## 5. Dùng giao diện tìm kiếm (`/ui`)

3 tab: **TKIS** (mô tả cảnh hoặc ảnh), **Q&A** (mô tả + câu hỏi), **TRAKE** (chuỗi sự kiện theo thứ tự thời gian). Gõ mô tả → Enter hoặc bấm Search.

### Panel "Submission ranking" (dưới cùng màn hình)

Đây là công cụ chỉnh sửa kết quả trước khi nộp bài — **chỉ sửa file kết quả hiển thị, không đụng gì tới model/điểm số**:

- **Đổi tên file tải xuống**: gõ tên vào ô đầu tiên (`filename.txt`) — dùng cho cả nút Download và Save to file.
- **Sửa tay thứ hạng**: mỗi dòng có ô `video_id` / `frame_id` sửa trực tiếp, nút `▲▼` đổi vị trí, `✕` xoá. Hoặc dùng form **"Set rank"**: nhập `video_id` + `frame_id` + số thứ hạng muốn gán (`→ Top #`) → bấm Apply, dòng đó sẽ nhảy đúng vị trí.
- **Gán hàng loạt theo bước nhảy**: form **"Batch step"** — nhập `video_id` gốc, `frame_id` gốc, bước nhảy `N`, số lượng `count` → bấm Generate: Top 1 = (video_id, frame_id), Top 2 = (video_id, frame_id+N), Top 3 = (video_id, frame_id+2N)... Dùng khi bạn chắc chắn 1 video đúng và muốn "rải" các frame lân cận làm phương án dự phòng.
- **💾 Save to file**: lưu vĩnh viễn xuống `submission/<tên file>.txt` trên đĩa (không mất khi tắt trình duyệt). **Load saved…** để mở lại 1 file đã lưu trước đó.
- **Download**: tải file về máy ngay, dùng tên đã đặt ở ô filename.

### Trang Batch (`/batch_answer.html`)

Nạp file `questions.csv` (100 câu hỏi), tự động chạy search cho từng câu (Auto current / Auto all), sửa tay từng ô nếu cần, rồi **Export CSV** — cũng có ô đổi tên file xuất.

---

## 6. Nộp bài

File nộp nằm ở `submission/*.txt` — đúng định dạng BTC (không header, dòng 1 = Top 1):
- KIS: `video_id, frame_idx`
- Q&A: `video_id, frame_idx, "câu trả lời"`
- TRAKE: `video_id, f1, f2, ..., fN`

Không cần chỉnh sửa gì thêm trước khi nộp.

---

## 7. Sự cố thường gặp

- **`.env` load xong mà `USE_C2LIP` không có hiệu lực** → kiểm tra lại `.env` (không phải `.env.example`) có đúng dòng `USE_C2LIP=True`, và bạn có đủ 2 file `c2lipweight.pt` + `embeddings_c2lip.pkl` ở đúng đường dẫn `C2LIP_CHECKPOINT` / `C2LIP_EMBEDDINGS_PKL`.
- **Không thấy thumbnail trong kết quả search** → `/api/frame/...` đọc trực tiếp từ file `.mp4` gốc, cần `VIDEO_DIR` trỏ đúng chỗ chứa video của bạn.
- **BM25 / cross-encoder không có tác dụng** → cần OCR/ASR/media-info JSON (từ BTC hoặc tự trích xuất); nếu không có, search vẫn chạy tốt chỉ dựa vào 2 dense encoder.
- Các lỗi CUDA/VRAM/Milvus khác → xem mục **Troubleshooting** trong `README.md`.

---

## 8. Tài liệu liên quan

- `README.md` — kiến trúc, công thức, cấu trúc thư mục đầy đủ.
- `.env.example` — toàn bộ biến cấu hình có thể chỉnh, đã chú thích.
- `SESSION_HANDOFF.md` — nhật ký kỹ thuật chi tiết của máy gốc (C2LIP, quirk môi trường...) — tham khảo nếu cần đào sâu, không bắt buộc đọc để chạy được hệ thống.
