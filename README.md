# AIC 2026 — tìm kiếm video bằng PE-Core và nộp bài DRES

> **English summary.** The interactive video-retrieval system our team used in the final round of the
> AI Challenge HCMC 2026. A text query (Vietnamese or English) is matched against **318,809 frames of 1,487 videos**
> with one GPU matrix product over [PE-Core-bigG-14-448](https://huggingface.co/timm/PE-Core-bigG-14-448)
> embeddings (~5 ms). A single-page web UI lets the operator browse results, scrub the video to the exact moment
> and **submit straight to DRES** for the three task types: Known-Item Search (KIS), Q&A and TRAKE
> (ordered multi-event grounding). All embeddings, keyframes and videos are on Hugging Face
> ([`5clnl/idk`](https://huggingface.co/datasets/5clnl/idk)); `scripts/setup_from_hf.py` downloads them and
> `start_server.sh` / `start_server_hf.bat` runs everything. The repo also keeps our earlier multi-model system
> ([README_LEGACY.md](README_LEGACY.md)). Docs below are in Vietnamese.

Hệ thống đội dùng ở **vòng chung kết AI Challenge HCMC 2026**: gõ mô tả (tiếng Việt hoặc tiếng Anh) → tìm khung hình khớp nhất
trong **1.487 video** → xem / tua video để chọn đúng thời điểm → **nộp thẳng lên DRES** từ giao diện web.

- **Một mô hình duy nhất:** PE-Core-bigG-14-448 (vector 1280 chiều). Vector ảnh đã tính sẵn; server chỉ chạy phần mã hoá văn bản.
- **Không cần cơ sở dữ liệu vector:** toàn bộ chỉ mục (fp16, 0,8 GB) nằm trên GPU, tìm **chính xác** trên mọi khung trong ~5 ms.
- **Tiếng Việt:** dịch sang tiếng Anh bằng [vinai-translate-vi2en-v2](https://huggingface.co/vinai/vinai-translate-vi2en-v2) trên GPU, có nhớ đệm từng câu.
- **TRAKE:** xếp hạng video theo **từng sự kiện**, xét **toàn bộ video**, căn **một khung cho mỗi sự kiện, tăng dần theo thời gian** bằng quy hoạch động.
- **Nộp DRES:** đăng nhập một lần, giữ phiên qua các lần khởi động lại, chặn nộp trùng, ghi nhật ký mọi lần nộp.

---

## Mục lục
1. [Bắt đầu nhanh](#1-bắt-đầu-nhanh)
2. [Dữ liệu](#2-dữ-liệu)
3. [Cách hệ thống hoạt động](#3-cách-hệ-thống-hoạt-động)
4. [Dùng giao diện và nộp bài](#4-dùng-giao-diện-và-nộp-bài)
5. [Chạy trên server thuê (vast.ai…)](#5-chạy-trên-server-thuê-vastai)
6. [Cấu hình](#6-cấu-hình)
7. [Dựng lại dữ liệu từ đầu](#7-dựng-lại-dữ-liệu-từ-đầu)
8. [Cấu trúc repo và kiểm thử](#8-cấu-trúc-repo-và-kiểm-thử)
9. [Sự cố thường gặp](#9-sự-cố-thường-gặp)

---

## 1. Bắt đầu nhanh

**Cần:** GPU NVIDIA **≥ 8 GB VRAM** (server dùng ~4 GB; đã chạy trên RTX 4060 laptop 8 GB và RTX 4090), Python 3.10+,
ổ trống **~420 GB** (đủ video) hoặc **~70 GB** (không video), cộng ~12 GB mô hình.

```bash
# 1) code
git clone https://github.com/Tinhcautoan/aic2026.git && cd aic2026

# 2) thư viện: PyTorch có CUDA trước (chọn đúng bản CUDA của máy), rồi phần còn lại (chỉ những gì server thi cần)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-server.txt

# 3) dữ liệu từ Hugging Face -> <DATA>  (tải + giải nén + kiểm tra; chạy lại được, tự nối tiếp khi đứt mạng)
python scripts/setup_from_hf.py <DATA> --delete-zips                 # đủ video
python scripts/setup_from_hf.py <DATA> --no-videos --delete-zips     # không video

# 4) chạy server, rồi mở http://localhost:8000 khi thấy "Engine ready"
bash start_server.sh <DATA>          # Linux
start_server_hf.bat <DATA>           # Windows
```

Lần chạy đầu server tự tải mô hình (~12 GB) vào `<DATA>/hf-cache`, mất vài phút; các lần sau ~1 phút.
Kiểm tra nhanh: http://localhost:8000/api/status phải trả `"ready": true, "frames": 318809, "videos": 1487, "translator": true`.

> `requirements.txt` (không có `-server`) là của hệ thống cũ nhiều mô hình — server thi **không** cần.

---

## 2. Dữ liệu

Toàn bộ dữ liệu ở dataset [`5clnl/idk`](https://huggingface.co/datasets/5clnl/idk):

| Thư mục | Nội dung | Dung lượng | Cần? |
|---|---|---|---|
| **`index_all/`** | chỉ mục: embedding PE-Core của mọi khung (`emb.npy`, 318.809 × 1280, fp16) + `frames.npy`, `vid.npy`, `ms.npy`, `videos.json` | 0,8 GB | **bắt buộc** |
| **`keyframes_all/`** | 7 zip = 234.083 ảnh keyframe `<video>_frame_<frame 6 chữ số>.jpg` + `manifest.json` | 66 GB | rất nên (ảnh trong lưới kết quả) |
| **`videos/`** | 1.487 video, từng file (`.mp4`; video N là `.mov`) | 350 GB | nên (xem / tua video) — chọn **một trong hai** |
| **`videos_all/`** | cùng 1.487 video, gói thành 9 zip (≤ 45 GB) | 350 GB | |

- 1.487 video = 614 video bộ mới (M: tin tức, N: camera giao thông cố định, S) + 873 video bộ cũ (L21–L30).
- 318.809 khung = keyframe theo đổi cảnh của mọi video (234.083) + khung lấy mẫu **mỗi 2 giây** của 287 video N (84.726).
  Video N gần như không đổi cảnh nên chỉ có 1–6 keyframe mỗi video; khung lấy mẫu dày giúp tìm đúng thời điểm.
  Ảnh của các khung dày **không** có trên Hugging Face: server tự cắt ảnh từ video khi cần.
- `embeddings_pecore*/` là embedding gốc từng video (chỉ cần khi dựng lại chỉ mục).

**Dung lượng ổ** (chưa tính ~12 GB mô hình): ~417 GB đủ video với `--delete-zips` (+ tối đa 45 GB tạm khi giải nén) ·
~67 GB không video. Không dùng `--delete-zips` thì cộng thêm 66 GB zip keyframe (và 350 GB nếu tải `videos_all/`).

**Tuỳ chọn của `setup_from_hf.py`:** `--videos-zip` (tải video dạng 9 zip thay vì 1.487 file) · `--no-videos` ·
`--delete-zips` (xoá **từng** zip ngay sau khi giải nén). Script in kết quả kiểm tra cuối cùng, ví dụ:

```
chỉ mục: 1487 video, 318,809 khung
keyframe: 234,083 ảnh; khung thiếu ảnh: 0 (khung dày của N, cắt từ video khi cần: 84,726)
video: có 1487/1487
OK
```

**Tải bằng tay** (lệnh `hf` có trong `pip install -U huggingface_hub`):

```bash
hf download 5clnl/idk --repo-type dataset --include "index_all/*"     --local-dir <DATA>
hf download 5clnl/idk --repo-type dataset --include "keyframes_all/*" --local-dir <DATA>
hf download 5clnl/idk --repo-type dataset --include "videos/*"        --local-dir <DATA>   # hoặc "videos_all/*"
cd <DATA> && for f in keyframes_all/*.zip; do unzip -q -o -j "$f" -d keyframes; done     # ảnh PHẲNG vào <DATA>/keyframes
for f in videos_all/*.zip; do unzip -q -o "$f" -d .; done                                 # chỉ khi tải videos_all
```

Cấu trúc server cần:

```
<DATA>/
├── index_all/     emb.npy  frames.npy  vid.npy  ms.npy  videos.json
├── keyframes/     234.083 ảnh .jpg (phẳng, không thư mục con)
├── videos/        1.487 video
└── hf-cache/      mô hình (tự tải)
```

---

## 3. Cách hệ thống hoạt động

```
câu hỏi (Việt/Anh) ─► dịch vi→en (GPU, nhớ đệm từng câu) ─► PE-Core text encoder ─► vector 1280 chiều
      ─► 1 phép nhân ma trận fp16 với TẤT CẢ 318.809 khung trên GPU (~5 ms) ─► chấm lại top bằng fp32
      ─► mỗi video tối đa 1 khung / 6 giây và tối đa 6 khung ─► lưới kết quả ─► chọn / chỉnh ─► NỘP lên DRES
```

**KIS / Q&A:** tìm theo độ tương đồng cosine giữa vector câu hỏi và vector từng khung. Không có cơ sở dữ liệu vector hay tìm
gần đúng: mọi khung đều được chấm điểm ([`retrieval/frame_index.py`](retrieval/frame_index.py)).

**TRAKE** ([`retrieval/trake_engine.py`](retrieval/trake_engine.py)) — câu hỏi là danh sách sự kiện theo thứ tự thời gian:
1. **Xếp hạng video theo từng sự kiện:** chấm riêng từng sự kiện, lấy khung khớp nhất của mỗi video cho mỗi sự kiện, rồi lấy
   trung bình qua các sự kiện (thay vì ghép cả đoạn mô tả thành một câu).
2. **Xét toàn bộ video:** bước căn chỉnh được dùng mọi khung của video, không chỉ một cửa sổ hẹp quanh khung khớp nhất.
3. **Căn chỉnh tăng dần:** quy hoạch động chọn đúng một khung cho mỗi sự kiện sao cho tổng điểm cao nhất và thời gian tăng
   dần, O(số sự kiện × số khung). Mỗi sự kiện kèm 8 khung thay thế để người dùng sửa tay.

---

## 4. Dùng giao diện và nộp bài

Tóm tắt (chi tiết từng nút: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)):

- **Tìm:** gõ mô tả rồi **Enter** (Shift+Enter xuống dòng). Bản dịch tiếng Anh hiện dưới ô, sửa được. Gõ thẳng tiếng Anh thì bỏ qua bộ dịch.
- **KIS / Q&A:** bấm một khung → thanh dưới hiện mã video và **ms ⇄ frame** (sửa ô này ô kia tự đổi) → **▶ Xem / chỉnh** để tua video
  → **NỘP ĐÁP ÁN** (Ctrl+Shift+Enter).
- **TRAKE:** dán các sự kiện (mỗi dòng một sự kiện) → Ctrl+Enter → **Chọn video này · xem & chỉnh khung** → thêm / bỏ / sửa từng khung → **NỘP**.
- **DRES:** bấm chip **DRES** → đăng nhập bằng tài khoản đội hoặc dán `sessionId`. Mỗi loại câu có evaluation riêng; giao diện tự chọn theo tab.

| Loại | Nội dung gửi DRES | Đơn vị |
|---|---|---|
| KIS | `mediaItemName = <VIDEO_ID>`, `start = end` | ms |
| Q&A | `QA-<ĐÁP ÁN>-<VIDEO_ID>-<ms>` | ms |
| TRAKE | `TR-<VIDEO_ID>-<f1>,<f2>,…` (tăng dần) | **chỉ số frame** (nộp bằng ms bị DRES chấm sai) |

Phiên DRES lưu ở `submission/dres_session.json` và nhật ký nộp ở `submission/dres_log.jsonl` (cả hai đã nằm trong `.gitignore`,
**đừng** đưa lên git). Các endpoint DRES chỉ nhận yêu cầu từ chính máy chạy server.

---

## 5. Chạy trên server thuê (vast.ai…)

`start_server.sh` / `start_server_hf.bat` chỉ cho server nghe trên `127.0.0.1`. Vào từ laptop bằng **đường hầm SSH**
(không mở cổng ra internet):

```bash
ssh -N -L 8000:127.0.0.1:8000 -o Compression=no -p <PORT> root@<IP>     # rồi mở http://localhost:8000 trên laptop
```

- Ai vào qua đường hầm cũng nộp được bằng phiên DRES của server → thống nhất **một người bấm nộp**.
- **Video giật / tải lâu:** video được gửi nguyên chất lượng (1,8–4,5 Mbps, tức 0,2–0,6 MB/s; một số video N tới 12 Mbps).
  Tìm kiếm và ảnh thì nhẹ, nhưng xem video cần đường truyền ổn định. Đo tốc độ thật trên laptop:
  `curl -o /dev/null -w "%{speed_download}\n" -H "Range: bytes=0-9999999" http://localhost:8000/api/video/M01_V001`.
  Dưới ~0,5 MB/s thì nên dùng **SSH trực tiếp** (vast.ai: "Direct ssh connect", không qua máy trung chuyển) và thuê máy ở châu Á.
  Trong lúc chờ, giao diện hiện keyframe tạm và tự chuyển sang video khi tải xong (nút **▶ Thử lại video**).

---

## 6. Cấu hình

Mọi thông số nằm trong [`config/settings.py`](config/settings.py), đè được bằng biến môi trường cùng tên (hoặc file `.env`).
`start_server.sh` / `start_server_hf.bat` đặt sẵn đường dẫn dữ liệu từ `<DATA>`:

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| `INDEX_DIR` | `<DATA>/index_all` | chỉ mục |
| `KEYFRAME_DIRS` | `<DATA>/keyframes` | thư mục ảnh keyframe, nhiều thư mục ngăn cách bằng `;` |
| `VIDEO_DIR`, `EXTRA_VIDEO_DIRS` | `<DATA>/videos`, *(trống)* | thư mục video (thư mục phụ ngăn cách `;`) |
| `HF_HOME` | `<DATA>/hf-cache` | nơi lưu mô hình; `HF_HUB_OFFLINE=1` để chạy không cần mạng sau khi đã có mô hình |
| `VI_EN_TRANSLATION_ENABLED` | `true` | tắt bộ dịch (tiết kiệm VRAM) nếu luôn gõ tiếng Anh |
| `RESULT_MAX_PER_VIDEO` | `6` | tối đa khung của một video trong lưới KIS/Q&A (0 = không giới hạn) |
| `TRAKE_FRAME_ID_MODE` | `frame` | đơn vị FRAME_ID của TRAKE |
| `DRES_BASE_URL` | `https://eventretrieval.one` | địa chỉ DRES |
| `DRES_USERNAME`, `DRES_PASSWORD` | *(trống)* | tuỳ chọn tự đăng nhập — chỉ đặt bằng biến môi trường / `.env`, **không** ghi vào code |

---

## 7. Dựng lại dữ liệu từ đầu

Dữ liệu trên Hugging Face đã đủ để chạy; phần này chỉ để làm lại hoặc mở rộng. Notebook chạy trên Google Colab, ghi lại đúng
cách đội đã làm (đọc / ghi dataset `5clnl/idk`; muốn dùng cho dữ liệu của bạn thì đổi tên dataset và đăng nhập token có quyền ghi):

| Bước | Notebook / script | Việc |
|---|---|---|
| 1 | `colab_keyframe_s01.ipynb` | tách keyframe theo đổi cảnh (CLIP ViT-L-14 dfn2b, xét mỗi 5 frame, giữ khung khác keyframe trước > 0,6 theo khoảng cách tương đối) |
| 2 | `colab_embed_pecore.ipynb` | embedding PE-Core cho keyframe, mỗi video một `.npz` |
| 3 | `notebooks/colab_embed_N_dense.ipynb` + `scripts/dense_sample.py` | lấy mẫu mỗi 2 giây video N + embedding |
| 4 | `scripts/make_dense_thumbs.py` | cắt ảnh nhỏ cho khung dày (từ video trên máy) |
| 5 | `scripts/build_index.py`, `scripts/ingest_dense.py` | gộp các `.npz` thành chỉ mục `index_all/` |
| 6 | `notebooks/colab_gom_du_lieu_hf.ipynb`, `colab_bundle_keyframes.ipynb`, `scripts/zip_videos_to_hf.py` | gom keyframe / video thành zip và đưa lên Hugging Face |

Các script trên có tham số mặc định là đường dẫn trên máy của đội (ổ `D:` / `E:`): hãy truyền đường dẫn của bạn
(`python scripts/<tên>.py --help`).

---

## 8. Cấu trúc repo và kiểm thử

**Hệ thống thi** (những gì server ở trên dùng):

| Đường dẫn | Vai trò |
|---|---|
| [`api/app.py`](api/app.py) | server FastAPI: `/api/search`, `/api/trake`, `/api/thumb`, `/api/video`, `/api/dres/*`, `/api/status`, giao diện |
| [`retrieval/frame_index.py`](retrieval/frame_index.py) | chỉ mục trên GPU: top-k chính xác, lọc trùng theo thời gian, giới hạn khung/video |
| [`retrieval/trake_engine.py`](retrieval/trake_engine.py) | TRAKE: xếp hạng theo sự kiện + quy hoạch động |
| [`retrieval/thumbs.py`](retrieval/thumbs.py) | ảnh thu nhỏ (JPEG → keyframe gần nhất → cắt từ video) và tìm file video |
| [`retrieval/dres_client.py`](retrieval/dres_client.py) | DRES: đăng nhập, phiên, evaluation, task đang chạy, nộp bài |
| [`embedding/pecore_text.py`](embedding/pecore_text.py) | phần mã hoá văn bản của PE-Core |
| [`embedding/translate_fast.py`](embedding/translate_fast.py), [`query_translator.py`](embedding/query_translator.py) | dịch Việt → Anh |
| [`frontend/index.html`](frontend/index.html) | toàn bộ giao diện (một file, không cần build) |
| [`config/settings.py`](config/settings.py) | cấu hình |
| `scripts/setup_from_hf.py`, `start_server.sh`, `start_server_hf.bat` | tải dữ liệu, chạy server |

**Phần khác trong repo** (không cần để chạy hệ thống thi):

| Đường dẫn | Là gì |
|---|---|
| [`README_LEGACY.md`](README_LEGACY.md), `api/app_legacy.py`, `main.py`, `run.py`, `embedding/hybrid_engine.py`, các module còn lại trong `retrieval/`, `milvus/`, `Dockerfile` | hệ thống trước đó: PE-Core + SigLIP2, BM25 trên OCR/ASR, cross-encoder rerank, Milvus Lite |
| `tulip/` | mã TULIP / OpenCLIP của bên thứ ba, giữ giấy phép gốc (Apache 2.0, xem `tulip/LICENSE`) |
| `HUONG_DAN_SU_DUNG.md`, `README_colab.md`, `colab_setup.ipynb`, `colab_download.ipynb` | hướng dẫn và notebook của hệ thống trước đó |
| `HUONG_DAN_CHUNG_KET.md`, `START_CONTEST*.bat`, `SESSION_HANDOFF.md` | ghi chú chạy trên máy thi của đội (đường dẫn riêng của máy đó) |

**Kiểm thử** hệ thống thi (DRES và Hugging Face đều là bản giả, không chạm server thật):

```bash
pip install pytest
python -m pytest tests/test_lean_engine.py tests/test_dres_client.py tests/test_dres_task.py tests/test_build_index.py tests/test_video_fetch.py tests/test_translator_settings.py
```

---

## 9. Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Trang không mở | server chưa chạy / đã tắt → chạy lại bước 4, đợi `Engine ready` |
| `Không thấy chỉ mục ở …` | sai `<DATA>` hoặc chưa tải → `python scripts/setup_from_hf.py <DATA>` |
| `"translator": false`, log có `Error parsing line b'\x0e'` | thiếu `protobuf` → `pip install protobuf sentencepiece`, chạy lại server |
| Lỗi CUDA / hết VRAM | cài đúng bản PyTorch có CUDA; đóng chương trình khác đang dùng GPU; hoặc `VI_EN_TRANSLATION_ENABLED=false` |
| Ảnh "không có ảnh" | thiếu keyframe → kiểm tra `KEYFRAME_DIRS`; khung dày N cần có video |
| Video không phát / tải lâu | thiếu file trong `VIDEO_DIR`, hoặc đường truyền chậm (xem mục 5) |
| DRES `412 … non-temporal information` | nộp nhầm evaluation (vd. TRAKE vào evaluation KIS) |
| DRES `401` | phiên hết hạn → đăng nhập lại |
| TRAKE mãi "đúng một phần" | có khung chưa đúng khoảnh khắc: giữ các khung chắc chắn, thay từng khung còn lại |

---

**Giấy phép:** [MIT](LICENSE) (riêng `tulip/` theo giấy phép gốc của nó). Dữ liệu video thuộc ban tổ chức AI Challenge HCMC.

**Ghi nhận:** [PE-Core](https://github.com/facebookresearch/perception_models) (Meta, dùng qua `timm` / `open_clip`) ·
[vinai-translate](https://huggingface.co/vinai/vinai-translate-vi2en-v2) (VinAI) · [DRES](https://github.com/dres-dev/DRES) ·
Ban tổ chức AI Challenge HCMC 2026.
