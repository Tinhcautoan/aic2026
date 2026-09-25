# AIC 2026 — Tìm kiếm video bằng PE-Core + nộp bài DRES

Hệ thống dùng ở vòng chung kết AIC 2026 cho 3 loại câu hỏi **KIS** (Textual / Video), **Q&A** và **TRAKE**.
Nhập mô tả (tiếng Việt hoặc tiếng Anh) → hệ thống tìm khung hình khớp nhất trong **1.487 video** → chọn / chỉnh thời điểm trên video → **nộp thẳng lên DRES** từ giao diện.

```
câu hỏi (Việt/Anh) ──dịch vi→en (GPU)──► PE-Core text encoder ──► 1 phép nhân ma trận với TẤT CẢ 318.809 khung (GPU, ~5 ms)
      ──► lọc trùng theo thời gian + tối đa 6 khung/video ──► giao diện web ──► chọn khung / xem video ──► NỘP lên DRES
TRAKE: chấm từng sự kiện → xếp hạng video (trung bình điểm cao nhất mỗi sự kiện) → quy hoạch động chọn 1 khung/sự kiện, tăng dần theo thời gian
```

- **Một mô hình duy nhất:** [PE-Core-bigG-14-448](https://huggingface.co/timm/PE-Core-bigG-14-448) (vector 1280 chiều). Vector ảnh đã tính sẵn; server chỉ chạy phần mã hoá văn bản.
- **Không cần cơ sở dữ liệu vector:** toàn bộ chỉ mục (fp16, 0,8 GB) nằm trên GPU, tìm chính xác trên mọi khung.
- **Dịch Việt → Anh** bằng [vinai-translate-vi2en-v2](https://huggingface.co/vinai/vinai-translate-vi2en-v2) trên GPU, có nhớ đệm từng câu.
- **Dữ liệu:** 614 video mới (M, N, S) + 873 video cũ (L21–L30) = 1.487 video, 318.809 khung trong chỉ mục (keyframe của mọi video + khung lấy mẫu mỗi 2 giây của 287 video camera giao thông N).

---

## Mục lục
1. [Yêu cầu máy](#1-yêu-cầu-máy)
2. [Cài đặt nhanh (4 bước)](#2-cài-đặt-nhanh-4-bước)
3. [Dữ liệu trên Hugging Face: tải gì, bỏ gì](#3-dữ-liệu-trên-hugging-face-tải-gì-bỏ-gì)
4. [Cấu trúc dữ liệu và biến môi trường](#4-cấu-trúc-dữ-liệu-và-biến-môi-trường)
5. [Dùng giao diện](#5-dùng-giao-diện)
6. [Nộp bài DRES](#6-nộp-bài-dres)
7. [Nhiều người dùng chung một server](#7-nhiều-người-dùng-chung-một-server)
8. [Cấu trúc code](#8-cấu-trúc-code)
9. [Sự cố thường gặp](#9-sự-cố-thường-gặp)

---

## 1. Yêu cầu máy

| Thứ | Tối thiểu | Ghi chú |
|---|---|---|
| GPU | NVIDIA **≥ 8 GB VRAM** | server dùng ~4 GB (chỉ mục + mô hình văn bản + bộ dịch). Đã chạy ổn trên RTX 4060 laptop 8 GB |
| Python | 3.10+ | |
| Ổ đĩa | **~67 GB** (không video) · **~417 GB** (đủ video) | + ~12 GB mô hình + tối đa 45 GB tạm khi giải nén. Xem [mục 3](#3-dữ-liệu-trên-hugging-face-tải-gì-bỏ-gì) |
| Mạng | lần đầu | tải dữ liệu + mô hình; sau đó chạy offline được |

---

## 2. Cài đặt nhanh (4 bước)

**Bước 1 — Lấy code** (nhánh `final-round-pecore-dres`; repo riêng tư thì cần được cấp quyền):

```bash
git clone -b final-round-pecore-dres https://github.com/Tinhcautoan/aic2026.git aicv5
cd aicv5
```

**Bước 2 — Cài thư viện.** Cài PyTorch có CUDA **trước** (chọn đúng bản CUDA của máy), rồi các thư viện gọn của server:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-server.txt
```

> `requirements.txt` là bộ đầy đủ của hệ thống cũ (Milvus, BM25, OCR, ASR…) — **không cần** cho server này.

**Bước 3 — Tải dữ liệu từ Hugging Face** (tải + giải nén + kiểm tra, tự nối tiếp khi đứt mạng, chạy lại bao nhiêu lần cũng được):

```bash
# Windows
python scripts\setup_from_hf.py D:\aic_data --delete-zips
# Linux
python3 scripts/setup_from_hf.py /data/aic --delete-zips
```

| Tuỳ chọn | Tác dụng |
|---|---|
| *(không có)* | tải `index_all/` + `keyframes_all/` (7 zip, tự giải nén) + `videos/` (1.487 file) |
| `--videos-zip` | tải video dạng **9 zip** (`videos_all/`) thay vì 1.487 file, tự giải nén |
| `--no-videos` | bỏ video (tiết kiệm 350 GB) — vẫn tìm và xem ảnh được, **không** xem / tua video được |
| `--delete-zips` | xoá **từng** zip ngay sau khi giải nén xong (không cần chỗ trống gấp đôi) |

Chọn gì, cần bao nhiêu ổ, tải bằng tay: [mục 3](#3-dữ-liệu-trên-hugging-face-tải-gì-bỏ-gì).

Kết thúc, script in ra kiểm tra, ví dụ:
```
chỉ mục: 1487 video, 318,809 khung
keyframe: 234,083 ảnh; khung thiếu ảnh: 0 (khung dày của N, cắt từ video khi cần: 84,726)
video: có 1487/1487
OK
```

**Bước 4 — Chạy server** rồi mở **http://localhost:8000** khi cửa sổ hiện `Engine ready` (~1 phút; lần đầu lâu hơn vì tự tải ~12 GB mô hình vào `<DATA>/hf-cache`):

```bash
# Windows
start_server_hf.bat D:\aic_data
# Linux
bash start_server.sh /data/aic          # thêm cổng khác: bash start_server.sh /data/aic 8001
```

Kiểm tra nhanh: http://localhost:8000/api/status phải trả `"ready": true, "frames": 318809, "videos": 1487`.

---

## 3. Dữ liệu trên Hugging Face: tải gì, bỏ gì

Toàn bộ dữ liệu ở dataset **[`5clnl/idk`](https://huggingface.co/datasets/5clnl/idk)**. Server cần **chỉ mục + keyframe + video**;
video có **hai dạng như nhau — chọn một**:

| Thư mục | Nội dung | Dung lượng | Cần? |
|---|---|---|---|
| **`index_all/`** | **embedding PE-Core** của mọi khung (`emb.npy`, 318.809 × 1280 fp16) + bảng tra `frames.npy`, `vid.npy`, `ms.npy`, `videos.json` — xem [Embedding PE-Core ở đâu](#embedding-pe-core-ở-đâu) | 0,8 GB | 🔴 **bắt buộc** — thiếu là không tìm được |
| **`keyframes_all/`** | 7 zip = **234.083** ảnh keyframe + `manifest.json` | 66 GB | 🟠 **rất nên** — ảnh trong lưới kết quả |
| **`videos/`** | **1.487** video, **từng file** (`.mp4`, video N là `.mov`) | 350 GB | 🟠 **nên** — xem / tua video khi chọn thời điểm nộp. **Chọn `videos/` HOẶC `videos_all/`** |
| **`videos_all/`** | **cùng 1.487 video**, gói thành **9 zip** (không nén, mỗi file ≤ 45 GB) + `manifest.json` | 350 GB | |

**`videos/` hay `videos_all/`?**
- **`videos/`** (mặc định): tải song song nhiều file, lấy được **một video lẻ**, không phải giải nén, không cần chỗ trống thêm.
- **`videos_all/`** (`--videos-zip`): chỉ **9 file** — hợp khi công cụ tải / mạng xử lý nhiều file kém. Phải giải nén;
  với `--delete-zips` script giải nén **từng zip rồi xoá ngay**, nên chỉ cần thêm tối đa ~45 GB tạm.

**Dung lượng ổ cần** (chưa tính ~12 GB mô hình tự tải):

| Cách tải | Ổ cần |
|---|---|
| đủ, video dạng `videos/`, `--delete-zips` | **~417 GB** (+ ≤ 21 GB tạm khi giải nén zip keyframe lớn nhất) |
| đủ, video dạng `videos_all/`, `--delete-zips` | **~417 GB** (+ ≤ 45 GB tạm) |
| không video (`--no-videos --delete-zips`) | **~67 GB** (+ ≤ 21 GB tạm) — tìm và xem ảnh được, **không** xem / tua video được |
| không `--delete-zips` | cộng thêm 66 GB (zip keyframe), và 350 GB nếu tải `videos_all/` |

### Cách tải

**Dùng script (khuyên dùng)** — tải + giải nén + kiểm tra, tự nối tiếp khi đứt mạng, chạy lại bao nhiêu lần cũng được
(zip đã giải nén xong thì không tải lại):

```bash
# Windows
python scripts\setup_from_hf.py D:\aic_data --delete-zips                  # video từng file (videos/)
python scripts\setup_from_hf.py D:\aic_data --videos-zip --delete-zips     # video dạng 9 zip (videos_all/)
# Linux
python3 scripts/setup_from_hf.py /data/aic --delete-zips
python3 scripts/setup_from_hf.py /data/aic --no-videos --delete-zips      # không video
```

**Tải bằng tay** (lệnh `hf` có trong `pip install -U huggingface_hub`):

```bash
hf download 5clnl/idk --repo-type dataset --include "index_all/*"      --local-dir <DATA>
hf download 5clnl/idk --repo-type dataset --include "keyframes_all/*"  --local-dir <DATA>
hf download 5clnl/idk --repo-type dataset --include "videos/*"         --local-dir <DATA>     # video từng file
#   hoặc: --include "videos_all/*"                                                          # video dạng 9 zip
hf download 5clnl/idk --repo-type dataset --include "videos/L26_V156.mp4" --local-dir <DATA>  # một video lẻ
```

Rồi giải nén:

```bash
# Linux: keyframe PHẲNG vào <DATA>/keyframes; zip video đã chứa sẵn thư mục videos/ nên giải nén vào <DATA>
cd <DATA>
for f in keyframes_all/*.zip; do unzip -q -o -j "$f" -d keyframes; done
for f in videos_all/*.zip;    do unzip -q -o "$f" -d .; done
```

```powershell
# Windows (PowerShell, tar có sẵn từ Windows 10)
cd <DATA>; mkdir keyframes -Force
Get-ChildItem keyframes_all\*.zip | ForEach-Object { tar -xf $_.FullName -C keyframes }
Get-ChildItem videos_all\*.zip    | ForEach-Object { tar -xf $_.FullName -C . }
```

### Chi tiết từng file zip

`keyframes_all/` — giải nén **tất cả** vào **một** thư mục phẳng:

| File | Ảnh | Dung lượng |
|---|---|---|
| `keyframes_M_1of4.zip` … `keyframes_M_4of4.zip` | 99.877 (video M) | 36,8 GB |
| `keyframes_N_S_1of1.zip` | 20.384 (video N, S) | 8,1 GB |
| `keyframes_old_L21-L30.zip` | 112.205 (video cũ L21–L30) | 20,7 GB |
| `keyframes_old_extra_1of1.zip` | 1.617 (11 video cũ bổ sung) | 0,3 GB |

`videos_all/` — bên trong mỗi zip video nằm ở `videos/<tên file>`; `manifest.json` cho biết video nào ở zip nào:

| File | Video | Dung lượng |
|---|---|---|
| `videos_M_1of3.zip`, `videos_M_2of3.zip`, `videos_M_3of3.zip` | 304 (M) | 110,1 GB |
| `videos_N_S_1of4.zip` … `videos_N_S_4of4.zip` | 310 (N, S) | 156,7 GB |
| `videos_L_1of2.zip`, `videos_L_2of2.zip` | 873 (cũ L21–L30) | 82,9 GB |

**Không cần tải:** `embeddings_pecore/`, `embeddings_pecore_N_dense/`, `embeddings_pecore_old_missing/` — embedding gốc từng video,
đã gộp vào `index_all/emb.npy`; chỉ cần khi dựng lại chỉ mục. Dataset [`Liamislazy/idk`](https://huggingface.co/datasets/Liamislazy/idk)
là nguồn gốc của dữ liệu **cũ** (L21–L30) — nội dung đã nằm trong `5clnl/idk`.

### Embedding PE-Core ở đâu

**Embedding PE-Core đã có sẵn trên HF — không cần tự tính lại.** Có 2 dạng:

**a) Bản đã gộp — server dùng bản này:** `index_all/emb.npy` (0,82 GB) = embedding PE-Core-bigG-14-448 của **cả 318.809 khung / 1.487 video**:
float16, 1280 chiều, đã chuẩn hoá L2. Dòng thứ *i* của `emb.npy` ứng với dòng thứ *i* của:

| File | Cho biết |
|---|---|
| `vid.npy` | video nào (vị trí trong `videos.json` → `videos[vid[i]]["id"]`) |
| `frames.npy` | chỉ số frame gốc trong video |
| `ms.npy` | thời điểm (ms) |
| `videos.json` | danh sách video: `id`, `fps`, `start`/`end` (khúc dòng của video đó), `source` (`batch2` = mới, `batch1-old` = cũ), `dense` (video N có khung lấy mẫu dày) |

```python
import json, numpy as np
d = "<DATA>/index_all"
emb = np.load(f"{d}/emb.npy", mmap_mode="r")          # [318809, 1280] float16
frames, vid, ms = np.load(f"{d}/frames.npy"), np.load(f"{d}/vid.npy"), np.load(f"{d}/ms.npy")
videos = json.load(open(f"{d}/videos.json", encoding="utf-8"))["videos"]
i = 0
print(videos[vid[i]]["id"], frames[i], ms[i], emb[i][:5])
```

**b) Bản gốc chưa gộp** — chỉ cần khi **dựng lại chỉ mục** (`scripts/build_index.py`, `ingest_dense.py`, `ingest_old_missing.py`):

| Nơi | Nội dung | Dung lượng |
|---|---|---|
| `5clnl/idk/embeddings_pecore/` | 614 `.npz`, mỗi file một video mới (M, N, S) — keyframe | 0,62 GB |
| `5clnl/idk/embeddings_pecore_N_dense/` | 287 `.npz` — khung lấy mẫu mỗi 2 giây của video N | 0,44 GB |
| `5clnl/idk/embeddings_pecore_old_missing/` | 11 `.npz` — 11 video cũ từng bị thiếu | 0,01 GB |
| `Liamislazy/idk/embeddings_pecore.pkl` | video cũ L21–L30 (862 video, 112.205 khung) | 1,28 GB |

**Mô hình** (không nằm trong dataset — server tự tải ở lần chạy đầu): `timm/PE-Core-bigG-14-448` (~9 GB) và `vinai/vinai-translate-vi2en-v2` (~3 GB).

**Ảnh khung dày của video N** (84.726 khung, mỗi 2 giây) **không có trên HF** — khi cần, server tự cắt ảnh từ video (hơi chậm hơn).
Dùng `--no-videos` thì các khung này hiện "không có ảnh" nhưng vẫn chọn và nộp được.

---

## 4. Cấu trúc dữ liệu và biến môi trường

```
<DATA>/
├── index_all/     emb.npy  frames.npy  vid.npy  ms.npy  videos.json
├── keyframes/     M01_V001_frame_000000.jpg  L21_V001_frame_000035.jpg ...   (234.083 ảnh, phẳng, không thư mục con)
├── videos/        M01_V001.mp4  N001-V001.mov  L21_V001.mp4 ...              (1.487 file)
└── hf-cache/      mô hình (tự tải)
```

`start_server_hf.bat` / `start_server.sh` đặt sẵn các biến này từ `<DATA>`; chạy kiểu khác thì tự đặt:

| Biến | Giá trị | Ghi chú |
|---|---|---|
| `INDEX_DIR` | `<DATA>/index_all` | bắt buộc |
| `KEYFRAME_DIRS` | `<DATA>/keyframes` | nhiều thư mục: ngăn cách bằng `;`, tìm theo thứ tự |
| `VIDEO_DIR` | `<DATA>/videos` | |
| `EXTRA_VIDEO_DIRS` | *(trống)* | thư mục video phụ, ngăn cách `;`. Video cũ thiếu có thể được tải lẻ từ HF vào thư mục phụ đầu tiên |
| `HF_HOME` | `<DATA>/hf-cache` | nơi lưu mô hình |
| `HF_HUB_OFFLINE=1` | *(tuỳ chọn)* | sau khi đã có mô hình: chạy không cần mạng |
| `RESULT_MAX_PER_VIDEO` | `6` | tối đa khung của một video trong lưới KIS/Q&A (0 = không giới hạn) |
| `DRES_BASE_URL` | `https://eventretrieval.one` | địa chỉ DRES |

Mọi thông số khác: [`config/settings.py`](config/settings.py) (mục *Lean PE-Core-only engine* và *DRES*).
Ảnh của một khung là `<video>_frame_<chỉ số frame 6 chữ số>.jpg`; không có ảnh thì server lấy keyframe gần nhất, rồi mới cắt từ video.

---

## 5. Dùng giao diện

**Thanh trên cùng:** tab loại câu (Textual KIS / Video KIS / Q&A / TRAKE) · ô **Mở video** (gõ `L26_V156`) · đồng hồ · task DRES đang chạy · chip **DRES** · 🧾 nhật ký nộp · ⚙ cài đặt.

**Tìm:** gõ mô tả rồi **Enter** (Shift+Enter = xuống dòng). Bản dịch tiếng Anh hiện dưới ô — sửa bản dịch rồi Enter để tìm lại bằng bản đã sửa. Gõ thẳng tiếng Anh thì không qua bộ dịch.

**KIS / Q&A**
1. Bấm một khung trong lưới → thanh dưới hiện **mã video**, **ms ⇄ frame** (sửa ô này ô kia tự đổi).
2. **▶ Xem / chỉnh** → video mở đúng thời điểm → tua (−1 s / −0,1 s / +0,1 s / +1 s hoặc gõ ms / frame) → **Chọn thời điểm này**.
3. Q&A: gõ câu trả lời vào ô ở thanh dưới.
4. **NỘP ĐÁP ÁN** (Ctrl+Shift+Enter).

**TRAKE**
1. Dán các sự kiện, mỗi dòng một sự kiện (hoặc đoạn `(1) … (2) … (3) …`) → **Ctrl+Enter**.
2. Mỗi hàng kết quả = một video + khung hệ thống căn cho từng sự kiện (kèm 8 khung thay thế mỗi sự kiện).
3. **Chọn video này · xem & chỉnh khung ▶** → tua tới khoảnh khắc → **＋ Thêm khung này**, sửa / bỏ từng khung → **NỘP ĐÁP ÁN**.

**⚙ Cài đặt:** cỡ ảnh · số kết quả · tối đa khung của một video · khoảng ± giây cho KIS · đơn vị FRAME_ID của TRAKE · tự tìm khi ngừng gõ · theo task DRES · hỏi xác nhận trước khi nộp.
Sửa `frontend/index.html` không cần khởi động lại server (Ctrl+F5); sửa code Python thì phải chạy lại.

Hướng dẫn chi tiết từng nút: [HUONG_DAN_CHUNG_KET.md](HUONG_DAN_CHUNG_KET.md).

---

## 6. Nộp bài DRES

1. Bấm chip **DRES** → đăng nhập bằng username/mật khẩu của đội **hoặc** dán `sessionId`. Phiên được lưu ở `submission/dres_session.json` (khởi động lại không phải đăng nhập lại — **đừng** đưa file này lên git).
2. Mỗi loại câu có **evaluation riêng**; giao diện tự chọn theo tab. Dòng "Nộp vào: …" ở thanh dưới cho biết đang nộp vào đâu.
3. Mọi lần nộp được ghi vào `submission/dres_log.jsonl`; hệ thống chặn nộp trùng.

| Loại | Nội dung gửi | Đơn vị thời gian |
|---|---|---|
| KIS | `mediaItemName = <VIDEO_ID>`, `start = end` | **ms** |
| Q&A | `QA-<ĐÁP ÁN>-<VIDEO_ID>-<ms>` | **ms** |
| TRAKE | `TR-<VIDEO_ID>-<f1>,<f2>,…` (tăng dần) | **chỉ số frame** (đã xác nhận trên DRES: nộp ms bị chấm sai, nộp frame được chấm đúng) |

| Lỗi | Nghĩa |
|---|---|
| `412 … does include non-temporal information` | nộp nhầm evaluation (vd. TRAKE vào evaluation KIS) |
| `Could not find media item … collection name` | điền **Tên collection** trong hộp DRES |
| `401` | phiên hết hạn → đăng nhập lại |

Các endpoint DRES chỉ nhận yêu cầu từ **chính máy chạy server** (`DRES_ALLOW_REMOTE=false`). Cả đội dùng một tài khoản → thống nhất **một người bấm nộp**.

---

## 7. Nhiều người dùng chung một server

Server chỉ nghe trên `127.0.0.1`. Người khác vào bằng đường hầm SSH (không mở cổng ra internet):

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<địa chỉ server>
```

rồi mở http://localhost:8000 trên máy mình. Ai vào qua đường hầm cũng nộp được bằng phiên DRES của server.

---

## 8. Cấu trúc code

| Đường dẫn | Vai trò |
|---|---|
| [`api/app.py`](api/app.py) | server FastAPI: `/api/search`, `/api/trake`, `/api/thumb`, `/api/video`, `/api/dres/*`, `/api/status` |
| [`retrieval/frame_index.py`](retrieval/frame_index.py) | chỉ mục trên GPU, tìm top-k + lọc trùng theo thời gian + giới hạn khung/video |
| [`retrieval/trake_engine.py`](retrieval/trake_engine.py) | TRAKE: xếp hạng video theo sự kiện + quy hoạch động |
| [`retrieval/thumbs.py`](retrieval/thumbs.py) | ảnh thu nhỏ (JPEG → keyframe gần nhất → cắt từ video) + tìm file video |
| [`retrieval/video_fetch.py`](retrieval/video_fetch.py) | tải lẻ một video cũ từ `videos.zip` trên HF khi máy không có |
| [`retrieval/dres_client.py`](retrieval/dres_client.py) | đăng nhập / phiên / evaluation / task / nộp bài DRES |
| [`embedding/pecore_text.py`](embedding/pecore_text.py) | phần mã hoá văn bản của PE-Core |
| [`embedding/translate_fast.py`](embedding/translate_fast.py) | dịch Việt → Anh theo câu, có nhớ đệm |
| [`frontend/index.html`](frontend/index.html) | toàn bộ giao diện (một file) |
| [`config/settings.py`](config/settings.py) | cấu hình, đè được bằng biến môi trường |
| [`scripts/setup_from_hf.py`](scripts/setup_from_hf.py) | tải dữ liệu từ HF (tải + giải nén từng zip + kiểm tra) |
| [`scripts/zip_videos_to_hf.py`](scripts/zip_videos_to_hf.py) | gói toàn bộ video thành zip và đưa lên HF (`videos_all/`) |
| `scripts/build_index.py`, `ingest_dense.py`, `ingest_old_missing.py` | dựng lại chỉ mục |
| `scripts/dense_sample.py`, `make_dense_thumbs.py` | lấy mẫu khung dày (N) |
| `notebooks/`, `colab_*.ipynb` | Colab: tách keyframe, tính embedding, gom dữ liệu lên HF |
| `START_CONTEST.bat` | trình chạy của máy thi (đường dẫn D:/E: riêng của máy đó) |
| `api/app_legacy.py`, `README_LEGACY.md` | hệ thống cũ (PE-Core + SigLIP2 + BM25 + Milvus), giữ để tham khảo |

Kiểm thử (dùng DRES giả, không chạm server thật):

```bash
python -m pytest tests/test_lean_engine.py tests/test_dres_client.py tests/test_dres_task.py tests/test_video_fetch.py tests/test_build_index.py
```

---

## 9. Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Trang không mở | server chưa chạy / đã tắt → chạy lại bước 4, đợi `Engine ready` |
| `Không thấy chỉ mục ở …` | sai `<DATA>` hoặc chưa tải → `python scripts/setup_from_hf.py <DATA>` |
| "Server đang nạp mô hình…" | đợi ~1 phút (lần đầu lâu hơn vì tải mô hình) |
| Lỗi CUDA / hết VRAM | cài đúng bản PyTorch có CUDA; đóng chương trình khác đang dùng GPU |
| Kết quả tiếng Việt rất tệ | bộ dịch không nạp được (xem log) → gõ tiếng Anh, hoặc kiểm tra mô hình `vinai` trong `hf-cache` |
| Ảnh "không có ảnh" | thiếu keyframe → kiểm tra `KEYFRAME_DIRS`; khung dày N cần có video |
| Video không phát | thiếu file trong `VIDEO_DIR` → tải lẻ: `hf download 5clnl/idk --repo-type dataset --include "videos/<ID>.*" --local-dir <DATA>` |
| TRAKE mãi "đúng một phần" | có khung chưa đúng khoảnh khắc: giữ khung chắc chắn, thay từng khung còn lại |
| Chip DRES đỏ | chưa đăng nhập / phiên hết hạn |
