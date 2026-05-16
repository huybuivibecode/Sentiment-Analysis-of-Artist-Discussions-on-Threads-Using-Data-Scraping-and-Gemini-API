# Social Sentiment Studio (SentimentAnalytist)
**Tác giả: Bùi Quang Huy**

**Liên hệ: buiquanghuy352k5@gmail.com**

**APP: https://toolcrawl-at.streamlit.app/ **

Dự án này là một công cụ toàn diện giúp cào dữ liệu (scrape) bình luận từ nền tảng **Threads** (ví dụ: về một người nổi tiếng, một chủ đề cụ thể), tiền xử lý ngôn ngữ tự nhiên (NLP), và thực hiện phân tích cảm xúc (Sentiment Analysis) sử dụng **Google Gemini API**. 

Ứng dụng cung cấp một giao diện người dùng trực quan qua **Streamlit** để thực hiện cào dữ liệu, cũng như phân tích và trực quan hóa dữ liệu theo thời gian thực (ví dụ như biểu đồ tròn, biểu đồ cột, Treemap, Word Cloud, phân tích Bigram).

## 🌟 Các tính năng chính

1. **Thu thập dữ liệu (Scraping):**
   - Tự động tìm kiếm, cuộn trang và trích xuất bình luận từ các bài viết trên Threads sử dụng `playwright`.
   - Lưu trữ state đăng nhập qua Cookie/Storage State để cào dữ liệu an toàn.

2. **Tiền xử lý văn bản tiếng Việt:**
   - Sử dụng thư viện `pyvi` để tách từ (word segmentation).
   - Loại bỏ các stop-words tiếng Việt kết hợp cùng danh sách từ nhiễu (noise/filler words) riêng cho mạng xã hội.

3. **Phân tích cảm xúc & Khai phá chủ đề bằng AI:**
   - Sử dụng mô hình **Gemini API** (ví dụ: `gemini-2.5-flash`) để gán nhãn cho từng bình luận.
   - **Sentiment:** Tích cực (positive), Tiêu cực (negative), Trung lập (neutral), Trái chiều (mixed).
   - **Emotion:** Ngưỡng mộ, Thất vọng, Tức giận, Hài hước, Đồng cảm,...
   - **Topic:** Âm nhạc, Drama, Hẹn hò, Sự nghiệp, Nhan sắc,...
   - **Controversy:** Phát hiện các bình luận mang tính tranh cãi, độc hại (toxic).

4. **Giao diện phân tích (Streamlit Dashboard):**
   - Hỗ trợ nhập và luân chuyển (rotate) nhiều API Key trực tiếp trên giao diện để tránh lỗi giới hạn (Quota Limit).
   - Giao diện Dashboard đẹp mắt, biểu đồ động qua `plotly`.

5. **Cơ chế Checkpoint tự động:**
   - Tiến trình gọi API LLM được lưu tự động xuống tệp `sentiment_checkpoint.csv`.
   - Khi bị gián đoạn do lỗi mạng hoặc hết token, hệ thống có thể tiếp tục tự động mà không cần bắt đầu lại.

## 🛠️ Cài đặt môi trường

**Bước 1: Clone hoặc tải mã nguồn về máy**
```bash
git clone https://github.com/huybuivibecode/Sentiment-Analysis-of-Artist-Discussions-on-Threads-Using-Data-Scraping-and-Gemini-API
cd SentimentAnalytist
```

**Bước 2: Cài đặt các thư viện Python**
Đảm bảo bạn đang sử dụng Python 3.9+ (khuyến nghị 3.10 trở lên).
```bash
pip install -r requirements.txt
```

**Bước 3: Cài đặt trình duyệt Playwright**
Để script cào dữ liệu hoạt động, bạn cần cài đặt các trình duyệt cho Playwright.
```bash
playwright install chromium
```

## 🚀 Hướng dẫn sử dụng

### 1. Khởi chạy ứng dụng Streamlit

Mở terminal tại thư mục dự án và chạy:
```bash
streamlit run app.py
```

Ứng dụng sẽ mở trên trình duyệt tại địa chỉ `http://localhost:8501`.

### 2. Cấu hình Gemini API Key
- Tại menu **Cấu hình hệ thống** bên trái (Sidebar), bạn có thể nhập các Gemini API Key.
- Nếu có nhiều API Key (để bypass rate limit/quota limit), hãy nhập mỗi key trên một dòng. Hệ thống sẽ tự động xoay vòng (rotate) qua các key nếu một key bị lỗi hoặc hết hạn mức.

### 3. Cào dữ liệu và Phân tích
- Tại giao diện chính, nhập tên người nổi tiếng hoặc từ khóa (ví dụ: `Miu Le, miule`).
- Chọn số lượng bài post và số comment mong muốn.
- Bấm **Chạy Threads**. Hệ thống sẽ tự động bật Playwright (Headless/Non-headless) tìm và quét data, sau đó tiến hành phân tích qua Gemini API và hiển thị kết quả trên Dashboard trực quan.

### 4. Tùy chỉnh (dành cho Developer)
- `threads/threads_runner.py`: Logic scrape Threads. Bạn có thể set `HEADLESS = False` để hiển thị cửa sổ Chromium khi cào dữ liệu, hữu ích cho việc quét mã QR/login lần đầu.
- `gemini_sentiment.py`: Prompt logic và cơ chế xoay vòng API, checkpoint lưu dữ liệu tạm thời.
- `data_preprocessing.py`: Bộ quy tắc loại bỏ rác/nhiễu từ HTML scraping và các thẻ filler.

## 🗂️ Cấu trúc thư mục

```text
SentimentAnalytist/
├── app.py                      # Ứng dụng Streamlit chính (UI + Visualizations)
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── data_preprocessing.py       # Module tiền xử lý văn bản, dọn rác và stop-words
├── gemini_sentiment.py         # Module kết nối Gemini API (Prompt, Retry, Checkpoint)
├── threads/
│   ├── threads_runner.py       # Crawler Threads bằng Playwright
│   ├── vietnamese-stopwords.txt# File từ vựng stop-words tiếng Việt (nếu có)
│   └── threads_storage_state.json # Cache cookie/session đăng nhập Threads
├── generated/                  # Nơi lưu trữ tự động các file Data output (CSV)
└── Jupyter Notebooks (.ipynb)  # File Notebook (nếu có) hỗ trợ phân tích thử nghiệm
```

## 📝 Chú ý về Đăng nhập Threads
Lần đầu tiên cào Threads, cookie/session có thể chưa hợp lệ dẫn đến không tìm thấy kết quả hoặc bị chặn hiển thị bình luận:
1. Bạn hãy đổi `HEADLESS = False` trong `threads/threads_runner.py`.
2. Khi chạy, Playwright sẽ hiện trình duyệt. Bạn hãy tiến hành đăng nhập Threads (bằng web). Trạng thái (cookie) sẽ được tự động lưu vào file `threads_storage_state.json` để sử dụng cho các lần sau.

---
**Chúc bạn có trải nghiệm tuyệt vời với Social Sentiment Studio!**
