# 🎨 Graphic Design Quality Evaluation System

Ứng dụng đánh giá chất lượng thiết kế đồ họa bằng Machine Learning kết hợp Deep Learning.

Hệ thống sử dụng mô hình LayoutLMv3 để trích xuất đặc trưng đa phương thức (hình ảnh, văn bản và bố cục), kết hợp với các đặc trưng hình học (Alignment, Overlap, Whitespace) và mô hình hồi quy Gradient Boosting để dự đoán điểm chất lượng thiết kế trên thang điểm từ 0 đến 10.

---

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_LINK_HERE)

---

## Project Overview

Mục tiêu của dự án là xây dựng hệ thống tự động đánh giá chất lượng bố cục thiết kế đồ họa như:

- Poster
- Banner
- Infographic
- Social Media Design

Thay vì đánh giá thủ công bởi chuyên gia thiết kế, hệ thống sử dụng Machine Learning để:

- Dự đoán điểm chất lượng thiết kế (0 - 10)
- Phân tích mức độ căn chỉnh (Alignment)
- Đánh giá hiện tượng chồng lấp (Overlap)
- Phân tích không gian trắng (Whitespace)
- Giải thích nguyên nhân ảnh hưởng đến điểm số

---

## Dataset

Nguồn dữ liệu:

GraphicDesignEvaluation Dataset (HuggingFace)

Dataset gồm:

- 700 mẫu thiết kế
- 100 thiết kế gốc
- 600 thiết kế bị biến đổi

Ba tiêu chí đánh giá chính:

| Dataset | Tiêu chí |
|----------|----------|
| Alignment | Căn chỉnh bố cục |
| Overlap | Chồng lấp thành phần |
| Whitespace | Không gian trắng |

Ground Truth được sinh tự động bằng GPT và chuẩn hóa theo thang điểm từ 0 đến 10.

---

## System Architecture

Pipeline xử lý:

```text
Input Design Image
        │
        ▼
 OCR + Bounding Boxes
        │
        ▼
Geometric Feature Extraction
        │
        ▼
 LayoutLMv3 Feature Extraction
        │
        ▼
Hybrid Feature Construction
        │
        ▼
StandardScaler + PCA
        │
        ▼
Regression Model
        │
        ▼
Predicted Quality Score
```

---

## Feature Extraction

### 1. Geometric Features

Hệ thống tính toán 12 đặc trưng hình học:

#### Alignment

- Left Edge Std
- Right Edge Std
- Center X Std
- Vertical Gap Std

#### Overlap

- Overlap Ratio
- Max IoU
- Overlap Pair Ratio

#### Whitespace

- Whitespace Ratio
- Left Margin
- Right Margin
- Top Margin
- Density Variance

---

### 2. Deep Features

Sử dụng:

```text
microsoft/layoutlmv3-base
```

Trích xuất:

- CLS Token Embedding (768 chiều)
- Visual Patch Mean Features
- Visual Patch Std Features

Sau đó chuẩn hóa bằng L2-Normalization.

---

### 3. Hybrid Features

Kết hợp:

```text
Hybrid Feature =
CLS Embedding
+ Visual Features
+ Geometric Features
```

Tạo vector biểu diễn cuối cùng trước khi đưa vào mô hình học máy.

---

## Machine Learning Models

Các mô hình được thử nghiệm:

### Gradient Boosting Regressor

Mô hình chính của hệ thống.

Ưu điểm:

- Hiệu quả với dữ liệu nhỏ
- Học tốt quan hệ phi tuyến
- Có khả năng giải thích đặc trưng

---

### Random Forest Regressor

Mô hình baseline để so sánh.

---

### Support Vector Regression (SVR)

Khai thác khả năng học biên tối ưu trong không gian đặc trưng.

---

## Evaluation Metrics

Mô hình được đánh giá bằng:

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Pearson Correlation
- Spearman Correlation
- Standard Deviation Difference

Các chỉ số giúp đánh giá cả độ chính xác lẫn khả năng tránh hiện tượng "Collapse to Mean".

---

## Streamlit Features

Ứng dụng cho phép:

✅ Upload ảnh thiết kế

✅ Tự động OCR

✅ Trích xuất đặc trưng bố cục

✅ Dự đoán điểm chất lượng

✅ Xếp loại thiết kế

✅ Hiển thị báo cáo giải thích

---

## Project Structure

```text
project/
│
├── app.py
├── model/
│   ├── gradient_boosting.pkl
│   ├── scaler.pkl
│   └── pca.pkl
│
├── utils/
│   ├── feature_extraction.py
│   ├── geometric_features.py
│   └── scoring.py
│
├── assets/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone project:

```bash
git clone YOUR_REPOSITORY_URL
cd project
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Chạy Streamlit:

```bash
streamlit run app.py
```

---

## Technologies

- Python
- Streamlit
- Scikit-Learn
- LayoutLMv3
- HuggingFace Transformers
- Tesseract OCR
- Pandas
- NumPy

---

## Future Improvements

- Fine-tuning LayoutLMv3
- SHAP Explainability
- Multi-criteria Scoring
- Real-time Design Feedback
- Design Recommendation System

---

## Author

Nguyễn Hồng Đăng

Machine Learning for Design Quality Evaluation

Assignment Project - 2026
