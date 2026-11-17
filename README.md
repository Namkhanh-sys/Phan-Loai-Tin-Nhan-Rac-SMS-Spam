# 📩 Phân Loại Tin Nhắn Rác (SMS Spam Classification) - Web Demo

> **Bài tập lớn môn Nhập môn Học máy**  
> **Đề tài 3:** Nhận diện và phân loại tin nhắn rác (SMS Spam) bằng Machine Learning.

---

## 📌 Giới thiệu đề tài

Dự án phát triển hệ thống phát hiện tin nhắn rác (SMS Spam) sử dụng kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và mô hình học máy truyền thống kết hợp giao diện web tương tác trực quan qua **Streamlit**.

- **Thuật toán chính:** Trích xuất đặc trưng **TF-IDF (1–2 gram)** kết hợp bộ phân loại **Logistic Regression** (với `class_weight='balanced'` để xử lý mất cân bằng dữ liệu).
- **Bộ dữ liệu:** SMS Spam Collection (tiếng Anh) gồm **5,169** mẫu tin nhắn sau khi làm sạch cấu trúc và loại trùng lặp (tỉ lệ Spam ~12.63%).
- **Giao diện Web Demo:** Xây dựng bằng Streamlit, hỗ trợ dự đoán theo thời gian thực, điều chỉnh ngưỡng nhạy cảm (Threshold), kiểm tra hàng loạt (Batch CSV) và trực quan hóa các chỉ số đánh giá.

---

## 📊 Kết quả thực nghiệm

Mô hình được đánh giá trên tập kiểm tra độc lập (Test set chiếm 20% dữ liệu, phân tầng theo nhãn `stratified`, `random_state=42`):

| Chỉ số đánh giá | Giá trị trên tập Test | Ghi chú |
|---|---|---|
| **Accuracy** | **99.23%** | Độ chính xác tổng thể |
| **F1-Score (Spam)** | **0.9690** | Điểm F1 cân bằng Precision & Recall cho lớp Spam |
| **Precision (Spam)** | **98.43%** | Tỉ lệ dự đoán là Spam thực sự chính xác (rất ít báo nhầm) |
| **Recall (Spam)** | **95.42%** | Tỉ lệ phát hiện được các tin nhắn Spam thực tế |
| **ROC-AUC** | **0.9951** | Khả năng phân tách giữa hai lớp Ham và Spam |

### Ma trận nhầm lẫn (Confusion Matrix trên 1,034 mẫu test)
```
                Thực tế: HAM    Thực tế: SPAM
Dự đoán: HAM        901 (TN)         6 (FN - bỏ sót spam)
Dự đoán: SPAM         2 (FP - báo nhầm)   125 (TP - bắt đúng spam)
```
*Ghi chú: Mô hình chỉ báo nhầm 2 tin nhắn thường thành rác (False Positive) và chỉ bỏ sót 6 tin nhắn rác trong hơn 1,000 tin nhắn kiểm thử.*

---

## 🏗️ Cấu trúc thư mục

```text
├── app.py                   # Giao diện web demo tương tác (Streamlit)
├── preprocessing.py         # Module tiền xử lý văn bản chuyên sâu (Text Cleaning & Slang normalization)
├── train_model.py           # Script huấn luyện mô hình và lưu trọng số vào model/
├── requirements.txt         # Danh sách thư viện Python cần thiết
├── requirements-lock.txt    # Danh sách phiên bản thư viện cố định chi tiết
├── chay_demo.bat            # Script click-to-run nhanh trên Windows
├── .gitignore               # Cấu hình bỏ qua môi trường ảo và dữ liệu nặng
├── README.md                # Tài liệu hướng dẫn sử dụng và báo cáo kết quả
├── model/                   # Thư mục chứa mô hình đã huấn luyện sẵn
│   ├── spam_model.joblib    # Trọng số Vectorizer + Classifier (~303 KB)
│   └── meta.json            # Siêu dữ liệu và các chỉ số đo lường
└── data/                    # Thư mục chứa dữ liệu đầu vào
    └── DAT_train.csv_VAO_DAY.txt # Hướng dẫn đặt train.csv nếu muốn huấn luyện lại
```

---

## ⚙️ Quy trình xử lý dữ liệu & Mô hình

### 1. Tiền xử lý văn bản (`preprocessing.py`)
Tin nhắn SMS thường ngắn, chứa nhiều ký tự đặc biệt, lỗi chính tả và từ lóng:
- **Chuẩn hóa Unicode & HTML entities**: Giải mã các ký tự html, chuẩn hóa NFC.
- **Mã hóa thực thể đặc biệt (Entity Tokenization)** bằng biểu thức chính quy (Regex):
  - URL / Web domain $\rightarrow$ `urltoken`
  - Địa chỉ Email $\rightarrow$ `emailtoken`
  - Số điện thoại $\rightarrow$ `phonetoken`
  - Đầu số / Shortcode dịch vụ 4–6 số $\rightarrow$ `shortcodetoken`
  - Ký hiệu tiền tệ (`$`, `£`, `€`) $\rightarrow$ `moneysymbol`
  - Chuỗi số thông thường $\rightarrow$ `numtoken`
- **Chuẩn hóa từ lóng tiếng Anh (SMS Slangs)**: Ánh xạ hơn 30 từ viết tắt phổ biến (`u` $\rightarrow$ `you`, `pls` $\rightarrow$ `please`, `2day` $\rightarrow$ `today`, `msg` $\rightarrow$ `message`, ...).
- **Rút gọn ký tự lặp**: Giảm các ký tự kéo dài (`cooool` $\rightarrow$ `cool`).

### 2. Trích xuất đặc trưng & Huấn luyện (`train_model.py`)
- **TF-IDF Vectorizer**: `ngram_range=(1, 2)` (kết hợp cả unigram và bigram), `sublinear_tf=True`, `min_df=2`. Vectorizer chỉ được `fit` trên tập Train để tránh rò rỉ dữ liệu (Data Leakage).
- **Logistic Regression**: `class_weight='balanced'`, tối ưu hóa `max_iter=2000` giúp mô hình hội tụ tốt và phân loại hiệu quả trên tập dữ liệu mất cân bằng.

---

## 🚀 Hướng dẫn cài đặt và chạy Web Demo

### Yêu cầu hệ thống
- Python 3.10 trở lên.
- Hệ điều hành: Windows, macOS hoặc Linux.

### Cách 1: Chạy trực tiếp (Khuyên dùng)

Dự án đã đính kèm sẵn mô hình đã huấn luyện trong thư mục `model/`, bạn có thể chạy ngay web mà không cần tải lại dataset:

1. **Clone repository về máy:**
   ```bash
   git clone https://github.com/Namkhanh-sys/Nh-p-m-n-h-c-m-y---ph-n-lo-i-tin-nh-n-r-c.git
   cd Nh-p-m-n-h-c-m-y---ph-n-lo-i-tin-nh-n-r-c
   ```

2. **Tạo môi trường ảo và cài đặt thư viện:**
   - **Trên Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     pip install -r requirements.txt
     ```
   - **Trên macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```

3. **Khởi chạy Web Demo:**
   ```bash
   streamlit run app.py
   ```
   Hoặc trên Windows bạn chỉ cần nhấp đúp vào file `chay_demo.bat`.  
   Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

### Cách 2: Huấn luyện lại mô hình (Tùy chọn)

Nếu bạn muốn tự huấn luyện lại mô hình từ dữ liệu gốc:

1. Đặt file dữ liệu `train.csv` vào thư mục `data/`.
2. Chạy lệnh:
   ```bash
   python train_model.py
   ```
   *Thêm tham số `--all` (`python train_model.py --all`) nếu muốn huấn luyện mô hình cuối cùng trên 100% dữ liệu sau khi hoàn tất kiểm thử.*

---

## 🖥️ Tính năng nổi bật của Web Demo (`app.py`)

1. **Dự đoán thời gian thực (Real-time Prediction):**
   - Hỗ trợ nhập văn bản tùy ý hoặc chọn nhanh từ các ví dụ mẫu (Spam trúng thưởng, Spam vay tiền, Tin nhắn công việc, Tin nhắn bạn bè...).
   - Hiển thị nhãn dự đoán (SPAM / HAM), xác suất chi tiết và thanh tiến trình trực quan.
   - Cho phép mở rộng xem văn bản sau khi qua các bước làm sạch của `heavy_clean`.
2. **Tùy chỉnh ngưỡng phân loại (Threshold Tuning):**
   - Thanh trượt ở Sidebar cho phép điều chỉnh ngưỡng xác định Spam (từ `0.1` đến `0.9`, mặc định `0.5`).
   - Giúp mô phỏng các chiến lược thực tế: ưu tiên không bỏ sót spam (hạ ngưỡng) hoặc ưu tiên không chặn nhầm tin quan trọng (tăng ngưỡng).
3. **Dự đoán hàng loạt (Batch Prediction):**
   - Tải lên file `.csv` chứa danh sách tin nhắn cần kiểm tra.
   - Tải về kết quả phân loại định dạng CSV ngay trên giao diện.
4. **Bảng thống kê mô hình & Ma trận nhầm lẫn:**
   - Hiển thị đầy đủ thông số siêu dữ liệu, kích thước tập dữ liệu và biểu đồ ma trận nhầm lẫn.
5. **Giao diện thân thiện & Tương thích cao:**
   - Được tối ưu CSS hiển thị sắc nét ở cả chế độ Dark Mode và Light Mode của Streamlit.

---

## 👨‍💻 Tác giả
- Sinh viên thực hiện: Bài tập lớn Nhập môn Học máy (ML).
- Repository: [Namkhanh-sys/Nh-p-m-n-h-c-m-y---ph-n-lo-i-tin-nh-n-r-c](https://github.com/Namkhanh-sys/Nh-p-m-n-h-c-m-y---ph-n-lo-i-tin-nh-n-r-c)
