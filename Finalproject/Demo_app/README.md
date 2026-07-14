# Recipe Recommender Demo App

Demo app gồm hai phần:

- FastAPI backend: load dữ liệu, load model, chạy search và chat.
- Streamlit frontend: giao diện tìm kiếm và hội thoại.

## 1. Chuẩn bị môi trường

Chạy từ root của repository:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu bạn đã có `.venv` rồi thì chỉ cần activate lại.

## 2. Chuẩn bị dữ liệu và model

Demo app cần các file sau:

```text
Finalproject/data/all_recipes_final.csv
Finalproject/notebooks/Saved_models/TFIDF/
Finalproject/notebooks/Saved_models/Ingredient_TFIDF/
Finalproject/notebooks/Saved_models/SBERT_FAISS/
Finalproject/notebooks/Saved_models/Hybrid/
Finalproject/notebooks/Saved_models/Hybrid_TFIDF_SBERT/
Finalproject/notebooks/Saved_models/RA_Rec/recipes_embeddings_list.pkl
```

Nếu thiếu model, hãy chạy lại các notebook training trong:

```text
Finalproject/notebooks/recommend_and_evaluation/
Finalproject/notebooks/RA_Rec/
```

## 3. Cấu hình LLM

Demo app dùng OpenAI-compatible endpoint, phù hợp với LiteLLM. App đọc file .env ở root repository:

`	ext
DS300-UIT-RecommenderSystem/.env
`

Nếu chưa có file này, tạo từ template root:

`ash
copy .env.example .env
`

Các biến cần có trong root .env:

`env
LLM_BASE_URL=litellm.imt-soft
LLM_MODEL_NAME=local-std-03
LLM_API_KEY=your-llm-api-key
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=4096
`

LLM_BASE_URL có thể ghi litellm.imt-soft hoặc URL đầy đủ như https://litellm.imt-soft/v1; app sẽ tự chuẩn hóa scheme và /v1.

## 4. Chạy backend

Terminal 1:

```bash
cd Finalproject\Demo_app
..\..\.venv\Scripts\activate
python api.py
```

Backend mặc định chạy ở:

```text
http://localhost:8000
```

Kiểm tra health:

```text
http://localhost:8000/health
```

## 5. Chạy frontend

Terminal 2:

```bash
cd Finalproject\Demo_app
..\..\.venv\Scripts\activate
streamlit run app.py
```

Frontend mặc định chạy ở:

```text
http://localhost:8501
```

## 6. Chế độ sử dụng

Search mode:

- Chọn thuật toán.
- Nhập query tự nhiên, ví dụ `bánh trung thu nhân đậu xanh`.
- App trả về danh sách món theo thuật toán đã chọn.

Conversational Chat mode:

- User nhập yêu cầu món ăn.
- LLM cập nhật dialogue state.
- RA-Rec late fusion tìm món phù hợp.
- LLM trình bày gợi ý tự nhiên cho user.

## 7. Ghi chú

- Search mode vẫn có thể chạy nếu LLM endpoint chưa dùng tới, nhưng backend hiện vẫn khởi tạo LLM handler để phục vụ chat.
- Nếu chat trả lỗi LLM, kiểm tra lại `LLM_BASE_URL`, `LLM_MODEL_NAME`, và `LLM_API_KEY`.
- Nếu backend báo thiếu file, kiểm tra lại `all_recipes_final.csv` và folder `Saved_models`.
