# Vòng chung kết AIC 2026 — hệ thống tìm kiếm PE-Core + nộp DRES

Tìm kiếm chỉ dùng **PE-Core-bigG-14-448** (không fusion, không BM25, không cross-encoder), nộp **một đáp án** lên DRES.

```
văn bản --dịch vi→en (GPU)--> tháp văn bản PE-Core --> 1 phép nhân ma trận trên TẤT CẢ khung hình (GPU, fp16)
        --> lọc trùng theo thời gian --> giao diện --> chọn 1 khung --> DRES
```

## 1. Dựng chỉ mục (một lần)
Embedding ảnh được làm trên Colab (`colab_embed_pecore.ipynb`) và lưu mỗi video một file `.npz`
(`frame_idx` int32, `pe_core` float32 [n,1280] đã chuẩn hóa) ở `embeddings_pecore/` trên Hugging Face.

```
python scripts/build_index.py --npz-dir <thư mục .npz> --videos <thư mục video> --keyframes <thư mục ảnh keyframe> --out D:\aic_index
python scripts/build_index.py ... --old-dir <thư mục có metadata/embeddings_OLDCORPUS_SAFE.pkl> --out D:\aic_index_all   # kèm bộ cũ
```
Script kiểm từng shard (hữu hạn, chuẩn hóa, frame tăng dần, và **khớp đúng tập ảnh keyframe đã tách**) rồi ghi
`emb.npy` (fp16), `frames.npy`, `vid.npy`, `ms.npy`, `videos.json`. ~230 nghìn khung ≈ 600 MB VRAM.

## 2. Chạy
`START_CONTEST.bat` (hoặc `START_CONTEST_ALL.bat` cho cả bộ cũ + mới) rồi mở http://localhost:8000 sau ~1 phút.
Biến môi trường chính: `INDEX_DIR`, `KEYFRAME_DIRS` (thư mục ảnh, ngăn cách bằng `;`), `VIDEO_DIR`, `HF_HOME`.

- Tìm bằng tiếng Việt hoặc tiếng Anh (tiếng Anh nhanh hơn ~0,3 s vì bỏ bước dịch); `Ctrl+Enter` tìm.
- KIS / Q&A: chọn một khung → **NỘP**. TRAKE: dán các sự kiện `(1) … (2) …`, hệ thống căn chỉnh theo thứ tự thời gian
  (xếp hạng theo từng sự kiện + quy hoạch động, phương pháp của AIC-HQT), đổi từng khung nếu cần.
- Trình xem video **tự chuyển sang keyframe lớn** khi video không phát được (thiếu file, ổ đĩa rớt, codec).

## 3. DRES
- Đăng nhập bằng username/mật khẩu **hoặc** dán `sessionId` lấy ở trang `/user` của DRES. Phiên được lưu ở
  `submission/dres_session.json` (không có mật khẩu, đã nằm trong `.gitignore`) nên khởi động lại server không mất phiên;
  server ping DRES mỗi 30 giây để giữ phiên và tự lấy evaluation ACTIVE mới.
- Tự đăng nhập lại khi phiên chết chỉ có khi đăng nhập bằng mật khẩu (mật khẩu chỉ ở bộ nhớ) và bị chặn nếu xảy ra >2 lần / 10 phút
  (tránh "đá nhau" với trang web DRES nếu DRES chỉ cho một phiên mỗi tài khoản). Tùy chọn: `DRES_USERNAME` / `DRES_PASSWORD`.
- Định dạng nộp (theo `HD-ChungKet-2026.pdf`): KIS `mediaItemName` + `start`/`end` (ms); Q&A `QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>`;
  TRAKE `TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,...` (mặc định ms, đổi sang chỉ số frame trong ⚙ nếu BTC yêu cầu).
  Ô **Tên collection** (tùy chọn) gửi kèm `mediaItemCollectionName` cho KIS.
- Chỉ nộp **một** đáp án mỗi lần, có chặn nộp trùng; mọi đáp án đã gửi được ghi vào `submission/dres_log.jsonl`.
- Các endpoint DRES chỉ trả lời chính máy này (`DRES_ALLOW_REMOTE=true` để mở) và từ chối request từ trang web khác.

## 4. Notebook Colab
| File | Việc |
|---|---|
| `colab_keyframe_pecore.ipynb`, `colab_keyframe_s01.ipynb` | tách keyframe (S01 có xử lý AV1) |
| `colab_embed_pecore.ipynb` | embedding PE-Core, mỗi video một `.npz`, upload HF, có % + thời gian còn lại, cảnh báo khi kẹt |
| `colab_bundle_keyframes.ipynb` | gộp mọi zip keyframe trên HF thành **một** file để tải bằng một cú bấm |

## 5. Kiểm thử
```
python -m pytest tests/test_dres_client.py tests/test_lean_engine.py
```
Chỉ mục, TRAKE (đối chiếu vét cạn với quy hoạch động), thumbnail, API và toàn bộ luồng DRES (bằng DRES giả, không chạm server thật).

## 6. Lưu ý đã biết
- Video dãy **N** (camera giao thông cố định) chỉ có ~1 keyframe/video vì thuật toán chọn keyframe theo đổi cảnh; muốn tìm đúng thời điểm
  nên lấy mẫu đều (mỗi 2–4 giây) rồi embed bổ sung.
- Ổ ngoài có thể rớt kết nối / lật bit: ưu tiên đặt chỉ mục, ảnh keyframe trên ổ trong; có thể kiểm CRC video với mục lục zip của BTC.
