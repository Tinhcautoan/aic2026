# Hướng dẫn sử dụng — AIC 2026 chung kết (tìm kiếm PE-Core + nộp DRES)

Tìm kiếm chỉ dùng **PE-Core-bigG-14-448**; mỗi lần nộp **một** đáp án lên DRES.

```
câu hỏi (Việt/Anh) → dịch vi→en (GPU) → tháp văn bản PE-Core → so với TẤT CẢ khung hình (1 phép nhân ma trận, GPU)
   → lọc trùng theo thời gian + tối đa 6 khung/video → giao diện → chọn khung / mở video chỉnh → NỘP lên DRES
```

**Dữ liệu hiện có:** 1.487 video (614 mới M/N/S + 873 cũ L21–L30), 318.809 khung trong chỉ mục
(keyframe theo cảnh của mọi video + khung lấy mẫu mỗi 2 giây của 287 video N camera giao thông).

---

## 1. Chạy trên máy thi (Windows, máy này)

1. Chạy `D:\aic_work\aicv5\START_CONTEST.bat` (một cửa sổ đen hiện ra — **không đóng nó**).
2. Đợi ~1 phút đến khi thấy `Engine ready`, mở **http://localhost:8000**.
3. Máy khởi động lại hoặc lỡ đóng cửa sổ → server tắt → chạy lại bước 1.

| Thứ | Ở đâu |
|---|---|
| Code | `D:\aic_work\aicv5` |
| Chỉ mục (embedding) | `D:\aic_index_all` (bản cũ để quay lại: `D:\aic_index_all_nodense`, `D:\aic_index_all_prev`) |
| Keyframe (ảnh) | `D:\aic_keyframes_all` (234.083 ảnh) + `D:\aic_N_dense_thumbs` (ảnh khung dày của N) |
| Video (mới + cũ) | `E:\AIC2026\aicv5\data\videos` (1.487 file) |
| Mô hình | `D:\hf-cache` (chạy offline) |
| Nhật ký mọi lần nộp | `D:\aic_work\aicv5\submission\dres_log.jsonl` |

Sửa giao diện không cần khởi động lại (chỉ Ctrl+F5 trang); sửa code Python thì phải tắt cửa sổ đen và chạy lại `.bat`.

---

## 2. Dùng giao diện

**Thanh trên cùng:** tab loại câu (Textual KIS / Video KIS / Q&A / TRAKE) · ô **Mở video** (gõ `L26_V156` → mở ngay) ·
đồng hồ + điểm ước tính · chip **task DRES đang chạy** · chip **DRES** (đăng nhập / evaluation) · 🧾 nhật ký · ⚙ cài đặt.

**Tìm:** gõ mô tả (Việt hoặc Anh) rồi **Enter** (hoặc Ctrl+Enter / nút Tìm). Shift+Enter = xuống dòng. Không tự tìm khi đang gõ
(bật lại trong ⚙ nếu muốn). Bản dịch tiếng Anh hiện dưới ô — sửa rồi Enter để tìm lại bằng bản đã sửa.

### KIS (Textual / Video) và Q&A
1. Bấm một khung trong lưới → đáp án hiện ở thanh dưới: **mã video**, **ms ⇄ frame** (sửa ô nào ô kia tự đổi).
2. Muốn chính xác hơn: **▶ Xem / chỉnh** (hoặc nút ▶ trên khung) → video mở đúng thời điểm → tua, dùng −1 s / −0,1 s / +0,1 s / +1 s,
   hoặc gõ ms / frame → **Chọn thời điểm này**.
3. Q&A: gõ câu trả lời vào ô ở thanh dưới (Enter = nộp).
4. **NỘP ĐÁP ÁN** (hoặc Ctrl+Shift+Enter).

Định dạng: KIS `mediaItemName` + `start = end` (ms); Q&A `QA-<ĐÁP ÁN>-<VIDEO>-<ms>`.

### TRAKE
1. Dán các sự kiện: mỗi dòng một sự kiện, hoặc cả đoạn `(1) … (2) … (3) …` → Ctrl+Enter (Enter ở đây là xuống dòng).
2. Mỗi hàng kết quả là một video với khung hệ thống căn chỉnh cho từng sự kiện.
3. **Chọn video này · xem & chỉnh khung ▶** → cửa sổ video + **danh sách khung sẽ nộp**:
   - tua tới khoảnh khắc → **＋ Thêm khung này** (bao nhiêu khung tùy bạn, tự sắp theo thời gian);
   - sửa ms hoặc frame từng khung, ✕ để bỏ; bấm khung **đề xuất** của hệ thống để thêm/bỏ; **↺ Về khung hệ thống**;
   - **NỘP ĐÁP ÁN** ngay trong cửa sổ.
4. **FRAME_ID của TRAKE = chỉ số frame gốc** (đã xác nhận với DRES: nộp bằng ms bị chấm SAI, cùng đáp án bằng frame được
   ĐÚNG MỘT PHẦN). Ô "FRAME_ID là" trong cửa sổ để đổi nếu BTC đổi quy định.
5. "Đúng một phần" = 50% ≤ số khung đúng < 100%; điểm tính ở lần đúng một phần **đầu tiên**, chỉ lần **đúng hoàn toàn** mới
   thay đổi điểm. Khung đúng cần là khoảnh khắc đặc trưng của sự kiện; đừng lệch vài giây.

### Cài đặt (⚙)
Kích thước ảnh · số kết quả (30/60/100/150) · **tối đa khung của một video** (mặc định 6 — để camera giao thông N không
chiếm hết lưới; "không giới hạn" khi đã biết đúng video) · khoảng ± giây cho KIS · đơn vị FRAME_ID TRAKE · tự tìm khi ngừng gõ ·
**theo task DRES** (tự đổi tab, bắt đầu câu mới và lấy thời lượng khi DRES sang task khác) · hỏi xác nhận trước khi nộp.

Màn hình **không bao giờ tự cuộn**; dải keyframe trong cửa sổ video chỉ nạp lại khi bấm **⟳ Keyframe quanh đây**.

---

## 3. DRES

- Bấm chip **DRES** → đăng nhập username/mật khẩu **hoặc** dán `sessionId` (trang `https://eventretrieval.one/user`).
  Phiên được lưu, khởi động lại server không phải đăng nhập lại.
- Mỗi loại câu có **evaluation riêng** (Textual KIS / Video KIS / QA / TRAKE): giao diện **tự chọn theo tab**, dòng "Nộp vào: …"
  ở thanh dưới cho biết đang nộp vào đâu; chữ vàng = không khớp → kiểm tra trước khi nộp.
- Lỗi thường gặp:

| Thông báo | Nghĩa |
|---|---|
| `412 … does include non-temporal information` | nộp nhầm evaluation (vd. TRAKE vào evaluation KIS) |
| `Could not find media item … collection name` | điền **Tên collection** trong hộp DRES nếu BTC yêu cầu |
| `401` | phiên hết hạn → đăng nhập lại |
| `412` khác | nộp trùng hoặc hết giờ |

- Có chặn nộp trùng; đáp án đã nộp sai bị làm mờ; mọi lần nộp ghi vào `submission\dres_log.jsonl`.
- Chỉ **máy chạy server** (hoặc qua đường hầm SSH) mới nộp được. Cả đội dùng **một** tài khoản → thống nhất **một người bấm nộp**.

---

## 4. Chạy trên máy/server khác với dữ liệu từ Hugging Face (Windows hoặc Linux)

### 4.1 Dữ liệu trên Hugging Face
`https://huggingface.co/datasets/5clnl/idk` — server chỉ cần **3 thư mục** (các thư mục khác trong repo là bản cũ/trung gian):

| Thư mục trên HF | Nội dung | Dung lượng | Bắt buộc? |
|---|---|---|---|
| `index_all/` | chỉ mục PE-Core của 1.487 video, 318.809 khung (`emb.npy`, `frames.npy`, `vid.npy`, `ms.npy`, `videos.json`) | ~0,8 GB | **có** — thiếu là không tìm được |
| `keyframes_all/` | 7 file zip = 234.083 ảnh keyframe (+ `manifest.json`: video nào nằm trong zip nào) | ~67 GB | nên có — để hiện ảnh kết quả |
| `videos/` | 1.487 video (614 mới + 873 cũ) trong một thư mục, từng file | ~350 GB | tùy — để xem/tua video |
| `videos_all/` | **cùng 1.487 video**, gói thành 9 zip ≤ 45 GB (chọn `videos/` HOẶC `videos_all/`) | ~350 GB | tùy |

### 4.2 Cấu trúc thư mục dữ liệu mà server cần

```
<DATA>/
├── index_all/        emb.npy  frames.npy  vid.npy  ms.npy  videos.json
├── keyframes/        M01_V001_frame_000000.jpg  L21_V001_frame_000035.jpg  ...  (234.083 ảnh, PHẲNG, không thư mục con)
├── videos/           M01_V001.mp4  N001-V001.mov  L21_V001.mp4  ...          (1.487 file)
└── hf-cache/         mô hình PE-Core + bộ dịch (tự tải ở lần chạy đầu, ~12 GB)
```

Server tìm dữ liệu qua **biến môi trường** (script chạy đặt sẵn, chỉ cần truyền `<DATA>`):

| Biến | Trỏ tới | Ghi chú |
|---|---|---|
| `INDEX_DIR` | `<DATA>/index_all` | bắt buộc |
| `KEYFRAME_DIRS` | `<DATA>/keyframes` | nhiều thư mục thì ngăn cách bằng `;` |
| `VIDEO_DIR` | `<DATA>/videos` | |
| `EXTRA_VIDEO_DIRS` | (trống) | thêm thư mục video khác, ngăn cách `;` |
| `HF_HOME` | `<DATA>/hf-cache` | nơi lưu mô hình |

Ảnh của một khung = `<video>_frame_<chỉ số frame 6 chữ số>.jpg` trong `KEYFRAME_DIRS`. Khung nào không có ảnh
(khung lấy mẫu dày của video N) thì server tự cắt từ video trong `VIDEO_DIR`.

### 4.3 Các bước

Cần: GPU NVIDIA ≥ 8 GB (server dùng ~4 GB), Python 3.10+, ổ trống ~417 GB với `--delete-zips` (+ tối đa 45 GB tạm khi giải nén;
`--no-videos --delete-zips`: ~67 GB nhưng không xem được video). Repo GitHub riêng tư thì người tải cần được cấp quyền.

```bash
# 1) code — nhánh final-round-pecore-dres
git clone -b final-round-pecore-dres https://github.com/Tinhcautoan/aic2026.git aicv5
cd aicv5

# 2) thư viện: PyTorch có CUDA trước, rồi phần còn lại (gọn, chỉ những gì server dùng)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-server.txt

# 3) dữ liệu từ Hugging Face -> <DATA> (tải + giải nén + kiểm tra; chạy lại được, tự nối tiếp khi đứt mạng)
python scripts/setup_from_hf.py /data/aic                  # Linux
python scripts\setup_from_hf.py D:\aic_data                # Windows
#   thêm --videos-zip (video dạng 9 zip)  /  --no-videos (không tải video)  /  --delete-zips (xóa từng zip sau khi giải nén)

# 4) chạy (lần đầu tự tải mô hình ~12 GB), rồi mở http://localhost:8000
bash start_server.sh /data/aic                             # Linux
start_server_hf.bat D:\aic_data                            # Windows
```

Cuối bước 3 script in kết quả kiểm tra, ví dụ `chỉ mục: 1487 video, 318,809 khung … khung thiếu ảnh: 0 … OK`.
Nếu báo thiếu, chạy lại đúng lệnh đó.

Chỉ muốn tải bằng tay một phần (không dùng script):

```bash
hf download 5clnl/idk --repo-type dataset --include "index_all/*" --local-dir <DATA>
hf download 5clnl/idk --repo-type dataset --include "keyframes_all/*" --local-dir <DATA>   # rồi giải nén TẤT CẢ zip vào <DATA>/keyframes
hf download 5clnl/idk --repo-type dataset --include "videos/L26_V156.mp4" --local-dir <DATA> # một video lẻ
```

### 4.4 Cho người khác dùng chung server
Server chỉ nghe trên chính nó (127.0.0.1). Người khác vào bằng đường hầm SSH từ laptop của mình:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<địa chỉ server>
# rồi mở http://localhost:8000 trên laptop
```

Ai vào qua đường hầm cũng **nộp được bằng phiên DRES của server** → thống nhất **một người bấm nộp**.

---

## 5. Cập nhật dữ liệu (khi cần)

| Việc | Lệnh / notebook |
|---|---|
| Dựng lại chỉ mục (+ khung dày N, + video cũ bổ sung) | `python scripts\ingest_dense.py` → `D:\aic_index_all_new`, tắt server, đổi tên thành `D:\aic_index_all`, chạy lại |
| Đưa đủ video cũ vào thư mục video (lấy từng video trong `videos.zip` trên HF) | `python scripts\fetch_old_videos.py` |
| Gộp keyframe vào một thư mục | `python scripts\merge_keyframes.py` |
| Dựng chỉ mục từ đầu | `python scripts\build_index.py --old-dir D:\aic_oldcorpus_backup --dense-npz-dir … --out …` |

Notebook Colab (3 notebook mới trong `notebooks/`, notebook tách keyframe/embedding bộ mới ở gốc repo; trên máy thi: `D:\aic_colab_notebooks`):

| File | Việc |
|---|---|
| `colab_gom_du_lieu_hf.ipynb` | gom keyframe mới + cũ vào `keyframes_all/` (7 zip) và đưa video cũ vào `videos/` trên HF |
| `colab_embed_N_dense.ipynb` | lấy mẫu mỗi 2 giây video N + embedding (287/298 đã xong; chạy lại cell 5 để làm nốt 11 video) |
| `colab_embed_old_missing11.ipynb` | tách keyframe + embedding 11 video cũ từng bị thiếu (đã xong) |
| `colab_embed_pecore.ipynb`, `colab_keyframe_*.ipynb` | tách keyframe và embedding bộ mới (đã xong) |

Kiểm thử: `python -m pytest tests/test_lean_engine.py tests/test_dres_client.py tests/test_dres_task.py tests/test_video_fetch.py tests/test_build_index.py`
(chỉ mục, TRAKE, thumbnail, API, toàn bộ luồng DRES bằng DRES giả — không chạm server thật).

---

## 6. Sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| Trang không mở / "Không kết nối được server" | server tắt → chạy lại `START_CONTEST.bat` (máy khác: `start_server_hf.bat` / `start_server.sh`) |
| `Không thấy chỉ mục ở …` khi chạy | chưa tải dữ liệu hoặc sai `<DATA>` → `python scripts/setup_from_hf.py <DATA>` |
| Lỗi CUDA / hết bộ nhớ GPU | cài đúng bản PyTorch có CUDA; GPU cần ≥ 8 GB, đóng chương trình khác đang dùng GPU |
| "Server đang nạp mô hình…" | đợi ~1 phút |
| Video không phát, hiện keyframe lớn | file video thiếu / ổ E: rớt → vẫn chọn được khung; kiểm tra ổ E: |
| Ảnh "không có ảnh" | thiếu keyframe → server thử cắt từ video; kiểm tra `KEYFRAME_DIRS` |
| TRAKE mãi "đúng một phần" | có khung chưa đúng khoảnh khắc: giữ các khung chắc chắn, thay từng khung để tìm khung sai |
| Chip DRES đỏ | chưa đăng nhập / phiên hết hạn |
