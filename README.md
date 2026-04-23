# nhom-chatbot-weather
Sinh Viên Thực Hiện: 

    LÊ QUANG LINH
              
    TỪ TẤT DUY
              
    NGUYỄN VĂN HÀ
Giảng Viên Hướng Dẫn: TRẦN QUỐC HƯNG

# Hướng dẫn: Chạy mô hình Chatbot Thời tiết

1) Yêu cầu trước
- Python 3.10+
- Cài Ollama và chọn mô hình (mô hình đang chọn là vinallama)

2) Thiết lập môi trường (Windows PowerShell
```
1. Tạo virtualenv (tuỳ chọn) 
      python -m venv .venv
      .\.venv\Scripts\Activate.ps1

2. Dùng anaconda khởi tạo môi trường (phiên bản python 3.10+) -> trỏ vào foder đang dùng

    Cài phụ thuộc (bash)
    pip install -r requirements.txt

3) Biến môi trường cần thiết
Tạo file `.env` ở thư mục gốc (đọc file .env(mau))

Tùy chọn: dùng Ngrok để expose server (Đăng kí ngrok để láy token)
NGROK_AUTH_TOKEN=your_ngrok_token_here
# Môi trường
ENVIRONMENT=development
```
3) Lưu ý: `config.py` kiểm tra `WEATHERAPI_API_KEY` và đường dẫn tới file trong `data/`.

4) Chạy server (API / web)
```
- Chạy local (uvicorn):
python run.py

- Chạy kèm Ngrok (yêu cầu `NGROK_AUTH_TOKEN` trong `.env`): local -> global
python run.py --ngrok

Sau khi server chạy, ứng dụng FastAPI được khởi tạo từ `main:app`.
```
5) Chạy chế độ tương tác trên Terminal (developer CLI) bỏ qua giao diện để chạy nhanh
```
python run_flast.py
Chế độ này khởi một phiên CLI tương tác, hữu ích để thử luồng hội thoại và debug.
```
6) LLM / mô hình
```
- Mặc định `config.py` dùng `LLM_MODEL = "vinallama-chat"`.
- Đảm bảo thư viện Ollama / model local có sẵn. Nếu dùng Ollama server, khởi server Ollama trước khi chạy ứng dụng.
- Nếu dùng model cục bộ khác, kiểm tra cấu hình ở `llm.py`.
```
7) Kiểm tra cấu hình và lỗi phổ biến
```
- Nếu thiếu `WEATHERAPI_API_KEY` (đăng kí WeatherAPI để lấy API) hoặc file trong `data/` không tồn tại, `run.py` sẽ in lỗi từ `validate_config()`.
- Khi bật `--ngrok`, nếu không có token trong `.env`, Ngrok sẽ không khởi chạy được.
```
# link tải file trọng số và data
[Click here](https://drive.google.com/drive/folders/15ll3g_xB-Jq5PxB3m8zMqupUZPewFFML?usp=sharing)
