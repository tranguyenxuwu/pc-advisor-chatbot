# RAG Chatbot Tư Vấn Cấu Hình PC

Đây là một chatbot sử dụng kiến trúc RAG (Retrieval-Augmented Generation) để tư vấn xây dựng và lựa chọn máy tính (PC) dựa trên nhu cầu và ngân sách của người dùng. Chatbot được xây dựng bằng Streamlit và sử dụng ChromaDB làm cơ sở dữ liệu vector.

## Tính năng chính

- **Tư vấn linh kiện**: Gợi ý CPU, GPU, RAM, SSD, ... phù hợp.
- **Tư vấn máy bộ**: Gợi ý các bộ máy tính đã được lắp ráp sẵn.
- **Hiểu ngôn ngữ tự nhiên**: Phân tích câu hỏi của người dùng để đưa ra truy vấn tìm kiếm tối ưu.
- **Giao diện web**: Giao diện trực quan dễ sử dụng với Streamlit.

## Yêu cầu hệ thống

- Python 3.11+
- `pip` và `venv`
- đề xuất sử dụng `conda`

## Hướng dẫn cài đặt

Thực hiện các bước sau để cài đặt và chạy ứng dụng trên máy của bạn.

### 1. Tạo môi trường ảo

Mở terminal và chạy các lệnh sau để tạo và kích hoạt môi trường ảo. Điều này giúp quản lý các thư viện Python một cách độc lập.

```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường ảo
# Trên macOS/Linux:
source .venv/bin/activate
# Trên Windows:
# .\.venv\Scripts\activate
```

### 2. Cài đặt PyTorch

PyTorch là một thư viện học sâu quan trọng cho ứng dụng này. Lệnh cài đặt sẽ khác nhau tùy thuộc vào hệ điều hành và phần cứng của bạn (đặc biệt là GPU).

Mở terminal (đã kích hoạt môi trường ảo) và chạy lệnh phù hợp với hệ thống của bạn:

**A. Dành cho máy có GPU NVIDIA (CUDA):**
_Lưu ý: Lệnh sau dành cho CUDA 12.1. Để biết lệnh chính xác cho phiên bản CUDA của bạn, hãy truy cập trang chủ PyTorch._

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**B. Dành cho máy Mac có chip Apple Silicon (M1/M2/M3):**

```bash
pip install torch torchvision torchaudio
```

**C. Dành cho máy chỉ sử dụng CPU (Windows, Linux):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> Để có lệnh cài đặt mới nhất và phù hợp nhất, bạn nên truy cập trang chủ của PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### 3. Cài đặt các thư viện còn lại

Sau khi cài đặt PyTorch, hãy cài đặt các thư viện khác được định nghĩa trong file `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị dữ liệu và tạo Embeddings

Đây là bước quan trọng để chatbot có dữ liệu để tìm kiếm. Chạy script `embedding.py` để đọc dữ liệu từ các file CSV trong thư mục `data/`, tạo vector embeddings và lưu vào ChromaDB.

**Lưu ý**: Quá trình này có thể mất vài phút tùy thuộc vào cấu hình máy tính của bạn, vì nó cần tải về các mô hình embedding từ Hugging Face.

```bash
python embedding.py
```

Sau khi chạy xong, bạn sẽ thấy một thư mục `data/chromadb` được tạo ra chứa cơ sở dữ liệu vector.

### 5. Cấu hình API Key

Ứng dụng cần API key từ một nhà cung cấp mô hình ngôn ngữ lớn (LLM) tương thích với OpenAI API (ví dụ: Groq, Together AI, hoặc chính OpenAI).

Bạn cần cung cấp **API Key** và **Base URL** trong giao diện của ứng dụng khi chạy.

## Chạy ứng dụng

Sau khi hoàn tất các bước trên, bạn có thể khởi động ứng dụng Streamlit bằng lệnh sau:

```bash
streamlit run app.py
```

Khi chạy script sẽ thông báo đng chạy trên thiết bị nào : `Using device: MPS (Apple Silicon GPU)`

Trình duyệt sẽ tự động mở một tab mới với địa chỉ `http://localhost:8501`. Tại đây bạn có thể bắt đầu trò chuyện với chatbot.

## Cấu trúc thư mục

```
.
├── app.py              # File chính của ứng dụng Streamlit (giao diện người dùng)
├── embedding.py        # Script để tạo và lưu trữ vector embeddings vào ChromaDB
├── retrieval.py        # Module xử lý logic truy xuất và tái xếp hạng (RAG)
├── requirements.txt    # Danh sách các thư viện Python cần thiết
├── README.md           # File hướng dẫn này
└── data/
    ├── parts.csv       # Dữ liệu về các linh kiện máy tính
    ├── prebuilt-pc.csv # Dữ liệu về các bộ máy tính lắp sẵn
    └── chromadb/       # Thư mục chứa cơ sở dữ liệu vector của ChromaDB
```
