# DS300 - Hệ thống Gợi ý Món Ăn

## Giới thiệu

Đây là đồ án cuối kỳ môn DS300 - Hệ thống Gợi ý (Recommender System) tại Trường Đại học Công nghệ Thông tin, ĐHQG-HCM. Hệ thống được xây dựng để gợi ý công thức nấu ăn cho người dùng dựa trên nhiều phương pháp khác nhau, bao gồm Content-Based Filtering, Collaborative Filtering và LLM-Based Recommendation.

## Mục lục

- [Tổng quan hệ thống](#tổng-quan-hệ-thống)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Thu thập dữ liệu](#thu-thập-dữ-liệu)
- [Phương pháp gợi ý](#phương-pháp-gợi-ý)
- [Cài đặt và sử dụng](#cài-đặt-và-sử-dụng)
- [Demo Application](#demo-application)
- [Đánh giá hệ thống](#đánh-giá-hệ-thống)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Tác giả](#tác-giả)

## Tổng quan hệ thống

Hệ thống gợi ý món ăn được phát triển với các tính năng chính:

- **Thu thập dữ liệu tự động**: Crawl dữ liệu công thức nấu ăn từ nhiều nguồn khác nhau
- **Xử lý và làm sạch dữ liệu**: Pipeline xử lý dữ liệu hoàn chỉnh với EDA chi tiết
- **Đa dạng phương pháp gợi ý**: Kết hợp nhiều thuật toán gợi ý khác nhau
- **Tích hợp LLM**: Sử dụng Google Generative AI để cải thiện chất lượng gợi ý
- **Ứng dụng demo**: Giao diện web tương tác với người dùng
- **Đánh giá toàn diện**: Metrics và evaluation framework đầy đủ

## Cấu trúc thư mục

### Finalproject/

Thư mục chứa đồ án chính với các thành phần:

#### data/
Chứa notebooks và dữ liệu liên quan đến thu thập và xử lý:

- **0_Crawl_vnexpress_raw_data.ipynb**: Crawl dữ liệu thô từ VNExpress
- **1_Crawl_vnexpress_multi_item_pages.ipynb**: Crawl nhiều trang từ VNExpress
- **2_Crawl_vncooking.ipynb**: Thu thập công thức từ VNCooking
- **3_Crawl_dienmayxanh.ipynb**: Crawl dữ liệu từ Điện Máy Xanh
- **4_Merge_and_preprocess_data.ipynb**: Gộp và tiền xử lý dữ liệu
- **5_EDA_data.ipynb**: Phân tích khám phá dữ liệu (Exploratory Data Analysis)

Các file CSV:
- `all_recipes_final.csv`: Tập dữ liệu cuối cùng sau xử lý
- `vnexpress_foods_detail.csv`, `vncooking_final_data.csv`, `dienmayxanh_foods_detail.csv`: Dữ liệu từ từng nguồn

#### Demo_app/
Ứng dụng demo web với các module:

- **api.py**: API endpoints cho hệ thống
- **app.py**: Ứng dụng chính (Streamlit/Flask)
- **config.py**: Cấu hình hệ thống
- **data_loader.py**: Module load dữ liệu
- **dialogue_manager.py**: Quản lý hội thoại với người dùng
- **llm_handler.py**: Xử lý tích hợp LLM
- **recommender.py**: Core recommendation engine
- **search_recommender.py**: Tìm kiếm và gợi ý
- **state_manager.py**: Quản lý trạng thái ứng dụng
- **requirements.txt**: Dependencies cho demo app

#### notebooks/
Notebooks phân tích và thử nghiệm:

- **RA_Rec/**: Notebooks về Reinforcement Learning-based Recommendation
- **recommend_and_evaluation/**: Notebooks đánh giá hệ thống
- **Saved_models/**: Models đã train và lưu lại

#### rec_and_eval/

Pipeline xây dựng ground truth và đánh giá retrieval:

- **1_build_groundtruth.ipynb**: Tạo query set, sinh method runs, và pooling candidate.
- **2_llm_label_groundtruth.ipynb**: Gán nhãn relevance 0-3 bằng LLM.
- **4_validate_groundtruth.ipynb**: Human validation/audit cho nhãn LLM.
- **5_manual_inspect_llm_relevance.ipynb**: Manual inspection trực quan các nhãn LLM.
- **6_evaluate_retrieval_runs.ipynb**: Report đánh giá cuối cùng.

### LAB/

Các bài thực hành trong học phần:

- **LAB1-LAB5**: Notebooks bài tập và datasets tương ứng
- Mỗi lab bao gồm: notebook thực hành, dữ liệu, và kết quả

## Thu thập dữ liệu

### Nguồn dữ liệu

Dữ liệu được thu thập từ 3 nguồn chính:

1. **VNExpress Ẩm thực**: Công thức nấu ăn và bài viết ẩm thực
2. **VNCooking**: Cộng đồng chia sẻ công thức nấu ăn
3. **Điện Máy Xanh**: Công thức và hướng dẫn nấu ăn

### Pipeline thu thập

1. Crawl URLs từ trang chủ
2. Lấy chi tiết từng công thức
3. Lưu checkpoint để tránh mất dữ liệu
4. Xử lý failed requests và retry
5. Merge và clean data

### Đặc điểm dữ liệu

Mỗi công thức bao gồm:
- Tên món ăn
- Nguyên liệu
- Các bước thực hiện
- Thời gian nấu
- Độ khó
- Thông tin dinh dưỡng (nếu có)
- Hình ảnh

## Phương pháp gợi ý

### 1. Content-Based Filtering

Dựa trên đặc trưng của món ăn:
- Sử dụng TF-IDF để vector hóa nguyên liệu và mô tả
- Tính toán độ tương đồng Cosine
- Gợi ý món ăn tương tự dựa trên preferences

### 2. Collaborative Filtering

Dựa trên hành vi người dùng:
- Matrix Factorization (SVD, NMF)
- User-based và Item-based approaches
- Khai thác patterns từ user interactions

### 3. LLM-Based Recommendation

Tích hợp Google Generative AI:
- Hiểu ngữ cảnh và ý định người dùng
- Gợi ý thông minh dựa trên hội thoại
- Cá nhân hóa cao hơn

### 4. Reinforcement Learning (RA-Rec)

Học từ feedback:
- Cải thiện gợi ý theo thời gian
- Tối ưu hóa user satisfaction
- Exploration-exploitation balance

## Cài đặt và sử dụng

### Yêu cầu

```
Python 3.8+
pip hoặc conda
```

### Cài đặt môi trường

1. Clone repository:
```bash
git clone <repository-url>
cd DS300-UIT-RecommenderSystem
```

2. Tạo virtual environment:
```bash
python -m venv DS300-venv
```

3. Kích hoạt environment:

Windows:
```bash
DS300-venv\Scripts\activate
```

Linux/Mac:
```bash
source DS300-venv/bin/activate
```

4. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

### Chạy notebooks

1. Khởi động Jupyter:
```bash
jupyter notebook
```

2. Mở và chạy các notebooks theo thứ tự trong thư mục `data/` để xử lý dữ liệu

3. Chạy các notebooks trong `Finalproject/notebooks/recommend_and_evaluation/` để train models

### Chạy demo application

1. Cài dependencies từ root repository:
```bash
pip install -r requirements.txt
```

2. Cấu hình LLM bằng file `.env` ở root repository:
```env
LLM_BASE_URL=litellm.imt-soft
LLM_MODEL_NAME=local-std-03
LLM_API_KEY=your-llm-api-key
```

3. Chạy backend:
```bash
cd Finalproject/Demo_app
python api.py
```

4. Chạy frontend ở terminal khác:
```bash
cd Finalproject/Demo_app
streamlit run app.py
```

Backend mặc định chạy ở `http://localhost:8000`, frontend chạy ở `http://localhost:8501`.

## Demo Application

Ứng dụng demo cung cấp:

- **Giao diện thân thiện**: Tìm kiếm và duyệt công thức dễ dàng
- **Gợi ý cá nhân hóa**: Dựa trên preferences và history
- **Hội thoại tự nhiên**: Chat với AI để tìm món phù hợp
- **Lọc nâng cao**: Theo nguyên liệu, thời gian, độ khó
- **Lưu favorites**: Quản lý công thức yêu thích

## Đánh giá hệ thống

### Metrics sử dụng

- **nDCG@K (Normalized Discounted Cumulative Gain)**: Metric chính cho relevance score 0-3
- **Precision@K**: Tỷ lệ kết quả liên quan trong top-K
- **MRR@K**: Vị trí kết quả liên quan đầu tiên
- **MAP@K**: Chất lượng ranking với relevance nhị phân từ ngưỡng `relevance >= 1`

### So sánh các phương pháp

Chi tiết trong notebook `Finalproject/notebooks/rec_and_eval/6_evaluate_retrieval_runs.ipynb`.


---
