# Hướng dẫn dùng giao diện và nộp bài DRES

Cài đặt và chạy server: xem [README](../README.md#1-bắt-đầu-nhanh). Khi thấy `Engine ready`, mở **http://localhost:8000**.

Sửa giao diện (`frontend/index.html`) không cần khởi động lại server, chỉ cần Ctrl+F5. Sửa code Python thì phải tắt server rồi chạy lại.

---

## 1. Màn hình

**Thanh trên cùng:**
- tab loại câu: **Textual KIS / Video KIS / Q&A / TRAKE**;
- ô **Mở video** (gõ mã video, ví dụ `L26_V156`, để mở ngay);
- đồng hồ và điểm ước tính;
- chip task DRES đang chạy, chip **DRES** (đăng nhập / evaluation);
- 🧾 nhật ký nộp và ⚙ cài đặt.

**Tìm:**
- Gõ mô tả bằng tiếng Việt hoặc tiếng Anh rồi **Enter** (hoặc Ctrl+Enter / nút Tìm). Shift+Enter để xuống dòng.
- Mặc định không tự tìm khi đang gõ; bật lại trong ⚙ nếu muốn.
- Bản dịch tiếng Anh hiện dưới ô. Sửa bản dịch rồi Enter để tìm lại bằng bản đã sửa.
- Gõ thẳng tiếng Anh thì bỏ qua bước dịch.

**Mẹo viết câu tìm:** PE-Core hiểu tiếng Anh tự nhiên. Tả **cái nhìn thấy trong khung hình**: người / vật, màu sắc, hành động,
bối cảnh, chữ trên màn hình. Câu ngắn, cụ thể thường tốt hơn câu dài kể cả câu chuyện.

---

## 2. KIS (Textual / Video) và Q&A

1. Bấm một khung trong lưới. Đáp án hiện ở thanh dưới: **mã video** và **ms ⇄ frame** (sửa ô nào thì ô kia tự đổi).
2. Muốn chính xác hơn: bấm **▶ Xem / chỉnh** (hoặc nút ▶ trên khung). Video mở đúng thời điểm đó:
   - tua bằng thanh thời gian, hoặc các nút −1 s / −0,1 s / +0,1 s / +1 s;
   - hoặc gõ thẳng ms / frame;
   - xong bấm **Chọn thời điểm này**.
3. Q&A: gõ câu trả lời vào ô ở thanh dưới (Enter để nộp).
4. Bấm **NỘP ĐÁP ÁN** (hoặc Ctrl+Shift+Enter).

Nếu video tải chậm, cửa sổ hiện keyframe tạm và tự chuyển sang video khi tải xong. Nút **▶ Thử lại video** dùng để tải lại.
Dải keyframe trong cửa sổ video chỉ nạp lại khi bấm **⟳ Keyframe quanh đây**. Màn hình **không bao giờ tự cuộn**.

---

## 3. TRAKE

1. Dán các sự kiện, mỗi dòng một sự kiện, hoặc cả đoạn dạng `(1) … (2) … (3) …`. Bấm **Ctrl+Enter** (ở tab này Enter là xuống dòng).
2. Mỗi hàng kết quả là một video, kèm khung hệ thống căn chỉnh cho từng sự kiện (theo thứ tự thời gian).
3. Bấm **Chọn video này · xem & chỉnh khung ▶**. Cửa sổ video mở ra kèm **danh sách khung sẽ nộp**:
   - tua tới khoảnh khắc cần nộp rồi bấm **＋ Thêm khung này**. Thêm bao nhiêu khung tuỳ bạn; danh sách tự sắp theo thời gian;
   - sửa ms hoặc frame của từng khung, bấm ✕ để bỏ một khung;
   - bấm khung **đề xuất** của hệ thống để thêm hoặc bỏ; **↺ Về khung hệ thống** để làm lại từ đầu;
   - bấm **NỘP ĐÁP ÁN** ngay trong cửa sổ.
4. **FRAME_ID của TRAKE là chỉ số frame gốc của video**, không phải ms. Lúc thi, nộp bằng ms bị chấm sai, còn cùng đáp án
   nộp bằng frame thì được chấm "đúng một phần". Ô **FRAME_ID là** trong cửa sổ (và trong ⚙) dùng để đổi nếu ban tổ chức đổi quy định.
5. "Đúng một phần" nghĩa là đúng từ 50% số khung trở lên nhưng chưa đủ 100%. Hãy giữ các khung chắc chắn và thay dần từng khung
   còn lại. Khung đúng phải là khoảnh khắc đặc trưng của sự kiện, đừng để lệch vài giây.

---

## 4. Cài đặt (⚙)

| Mục | Ý nghĩa |
|---|---|
| Kích thước ảnh, số kết quả (30/60/100/150) | hiển thị lưới |
| Tối đa khung của một video | mặc định 6, để video camera cố định (N) không chiếm hết lưới; chọn "không giới hạn" khi đã biết đúng video |
| Khoảng ± giây cho KIS | khoảng thời gian gửi kèm đáp án KIS |
| Đơn vị FRAME_ID TRAKE | chỉ số frame (mặc định) hoặc ms |
| Tự tìm khi ngừng gõ | mặc định tắt |
| Theo task đang chạy trên DRES | tự đổi tab, bắt đầu câu mới và lấy thời lượng khi DRES chuyển sang task khác |
| Hỏi xác nhận | phải bấm NỘP hai lần |

---

## 5. DRES

**Đăng nhập:**
- Bấm chip **DRES**, rồi đăng nhập bằng tài khoản đội **hoặc** dán `sessionId` (lấy ở trang người dùng của DRES).
- Phiên được lưu ở `submission/dres_session.json`, nên khởi động lại server không phải đăng nhập lại. File này **không được đưa lên git**
  (đã có trong `.gitignore`).
- Muốn tự đăng nhập khi khởi động: đặt `DRES_USERNAME` / `DRES_PASSWORD` bằng biến môi trường hoặc trong file `.env`
  (cũng đã nằm trong `.gitignore`). **Không** ghi vào code.

**Evaluation:**
- Mỗi loại câu có evaluation riêng (Textual KIS / Video KIS / QA / TRAKE). Giao diện **tự chọn theo tab**.
- Dòng "Nộp vào: …" ở thanh dưới cho biết đang nộp vào đâu. Chữ vàng nghĩa là không khớp, cần kiểm tra trước khi nộp.

**Định dạng gửi lên:**

| Loại | Nội dung | Đơn vị |
|---|---|---|
| KIS | `mediaItemName = <VIDEO_ID>`, `start = end` | ms |
| Q&A | `QA-<ĐÁP ÁN>-<VIDEO_ID>-<ms>` | ms |
| TRAKE | `TR-<VIDEO_ID>-<f1>,<f2>,…` | chỉ số frame |

**An toàn khi nộp:**
- Giao diện chặn nộp trùng. Đáp án đã nộp sai bị làm mờ.
- Mọi lần nộp được ghi vào `submission/dres_log.jsonl`.
- Các endpoint DRES chỉ nhận yêu cầu từ chính máy chạy server (hoặc qua đường hầm SSH). Cả đội dùng chung **một** tài khoản
  nên cần thống nhất **một người bấm nộp**.

**Lỗi thường gặp:**

| Thông báo | Nghĩa |
|---|---|
| `412 … does include non-temporal information` | nộp nhầm evaluation (ví dụ TRAKE vào evaluation KIS) |
| `Could not find media item … collection name` | điền **Tên collection** trong hộp DRES nếu ban tổ chức yêu cầu |
| `401` | phiên hết hạn, cần đăng nhập lại |
| `412` khác | nộp trùng hoặc hết giờ |
| chip DRES đỏ | chưa đăng nhập hoặc phiên hết hạn |

---

## 6. API (cho người muốn tự viết công cụ)

| Endpoint | Việc |
|---|---|
| `GET /api/status` | trạng thái: `ready`, số khung, số video, bộ dịch |
| `POST /api/search` | `{"query": "...", "top_k": 60}` (tối đa 300; `per_video` đổi giới hạn khung/video), trả về danh sách khung (video, frame, ms, điểm) và bản dịch |
| `POST /api/trake` | `{"events": ["...", "..."], "top_k": 20}` (tối đa 12 sự kiện), trả về video, mỗi sự kiện một khung kèm các khung thay thế |
| `GET /api/thumb/{video}/{frame}` | ảnh JPEG của khung |
| `GET /api/video/{video}` | file video (hỗ trợ HTTP Range để tua) |
| `GET /api/videoinfo/{video}` | fps, số khung trong chỉ mục, máy có file video không |
| `/api/dres/*` | đăng nhập, evaluation, task đang chạy, nộp bài (chỉ từ máy chạy server) |

Chi tiết tham số: xem [`api/app.py`](../api/app.py) hoặc mở http://localhost:8000/docs khi server đang chạy.
