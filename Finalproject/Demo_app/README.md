# RA-Rec Food Recommender System

Hệ thống gợi ý món ăn đơn giản với Streamlit + FastAPI.

## Cài đặt

```bash
cd RA_Rec_app
pip install -r requirements.txt #có thể tạo môi trường ảo nếu muốn
```

## Cấu hình

Tạo file `.env`:

```bash
cp .env.example .env
```

Thêm Google API key vào `.env`:

```
GOOGLE_API_KEY=your_key_here
```

Tải model xuống:
Link drive: https://drive.google.com/drive/folders/1WNkUjkH3JJWx1xfEcgKcnx32QAn3aVkj?usp=drive_link

Lưu ý: tải toàn bộ folder Saved_models, giải nén và bỏ vào folder notebooks (notebooks/Saved_models)

## Sử dụng

**Terminal 1 - Backend:**
```bash
python api.py
```

**Terminal 2 - Frontend:**
```bash
streamlit run app.py
```

Truy cập: http://localhost:8501

## Hướng dẫn

1. Mở giao diện Streamlit
2. Chat với bot để tìm món ăn
3. Nói "restart" để bắt đầu lại
