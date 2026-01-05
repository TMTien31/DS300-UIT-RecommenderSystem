# RA-Rec Food Recommender System

Hệ thống gợi ý món ăn đơn giản với Streamlit + FastAPI.

## Cài đặt

```bash
cd RA_Rec_app
pip install -r requirements.txt #có thể tạo môi trường ảo nếu muốn
```

## Cấu hình

1. Tạo file `.env`:

```bash
cp .env.example .env
```

2. Thêm Google API key vào `.env`:

```
GOOGLE_API_KEY=your_key_here
```

3. Tải model xuống:
6 model cơ bản: Link GG Drive: https://drive.google.com/drive/folders/1WNkUjkH3JJWx1xfEcgKcnx32QAn3aVkj?usp=drive_link
Lưu ý: tải toàn bộ folder Saved_models (vì các model nặng nên có thể cần tải và giải nén từng folder riêng), giải nén và bỏ vào folder Finalproject\notebooks\Saved_models\...

Model RA-REC: https://drive.google.com/file/d/1KPPWXGv52jamRULpjw2EFVYMdx24IG2x/view?usp=drive_link
Lưu ý: model RA-Rec được đặt vào thư mục Finalproject\notebooks\RA_Rec\recipes_embeddings_list.pkl

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
