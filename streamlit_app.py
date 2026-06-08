import streamlit as st
import numpy as np
import torch
import io
import os
from PIL import Image
import joblib
import warnings
import pytesseract

if os.name == 'nt' and os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Đánh giá Thiết kế Đồ họa",
    page_icon="🎨",
    layout="wide",
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
GEO_FEATURE_NAMES = [
    "geo_left_edge_std",
    "geo_right_edge_std",
    "geo_center_x_std",
    "geo_center_y_regularity",
    "geo_overlap_ratio",
    "geo_max_iou",
    "geo_overlap_pair_ratio",
    "geo_whitespace_ratio",
    "geo_margin_left",
    "geo_margin_right",
    "geo_margin_top",
    "geo_density_variance",
]

GEO_DESCRIPTIONS = {
    "geo_left_edge_std":       ("Alignment",  "Độ lệch cạnh trái — nhỏ = căn đều trái"),
    "geo_right_edge_std":      ("Alignment",  "Độ lệch cạnh phải — nhỏ = căn đều phải"),
    "geo_center_x_std":        ("Alignment",  "Độ lệch tâm ngang — nhỏ = căn giữa đều"),
    "geo_center_y_regularity": ("Alignment",  "Khoảng cách hàng đều — lớn = nhịp đọc tốt"),
    "geo_overlap_ratio":       ("Overlap",    "Tỉ lệ pixel chồng lấp / canvas"),
    "geo_max_iou":             ("Overlap",    "IoU lớn nhất giữa 2 phần tử"),
    "geo_overlap_pair_ratio":  ("Overlap",    "% cặp phần tử có chồng lấp"),
    "geo_whitespace_ratio":    ("Whitespace", "Tỉ lệ diện tích trống — ~0.4-0.6 là tối ưu"),
    "geo_margin_left":         ("Whitespace", "Lề trái nhỏ nhất"),
    "geo_margin_right":        ("Whitespace", "Lề phải nhỏ nhất"),
    "geo_margin_top":          ("Whitespace", "Lề trên nhỏ nhất"),
    "geo_density_variance":    ("Whitespace", "Variance mật độ lưới 4×4 — nhỏ = phân bố đều"),
}


def compute_geometric_features(boxes_raw: np.ndarray) -> np.ndarray:
    CANVAS = 1000.0
    valid = boxes_raw[
        (boxes_raw[:, 2] > boxes_raw[:, 0]) & (boxes_raw[:, 3] > boxes_raw[:, 1])
    ]
    if len(valid) < 2:
        return np.zeros(12, dtype=np.float32)

    x0, y0, x1, y1 = valid[:, 0], valid[:, 1], valid[:, 2], valid[:, 3]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0).clip(min=0)
    h = (y1 - y0).clip(min=0)
    area = w * h

    rng_x0 = x0.max() - x0.min() + 1e-6
    rng_x1 = x1.max() - x1.min() + 1e-6
    rng_cx = cx.max() - cx.min() + 1e-6

    left_edge_std = float(np.std(x0) / rng_x0)
    right_edge_std = float(np.std(x1) / rng_x1)
    center_x_std = float(np.std(cx) / rng_cx)

    sorted_cy = np.sort(np.unique(cy.round(-1)))
    if len(sorted_cy) >= 2:
        gaps = np.diff(sorted_cy)
        center_y_regularity = float(1.0 - np.std(gaps) / (np.mean(gaps) + 1e-6))
    else:
        center_y_regularity = 1.0

    n = len(valid)
    overlap_pixels = 0.0
    iou_list = []
    overlap_pairs = 0

    sample = valid[: min(n, 60)]
    ns = len(sample)
    sx0, sy0, sx1, sy1 = sample[:, 0], sample[:, 1], sample[:, 2], sample[:, 3]

    for i in range(ns):
        for j in range(i + 1, ns):
            ix0 = max(sx0[i], sx0[j])
            iy0 = max(sy0[i], sy0[j])
            ix1 = min(sx1[i], sx1[j])
            iy1 = min(sy1[i], sy1[j])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if inter > 0:
                a_i = (sx1[i] - sx0[i]) * (sy1[i] - sy0[i])
                a_j = (sx1[j] - sx0[j]) * (sy1[j] - sy0[j])
                union = a_i + a_j - inter
                iou_list.append(inter / (union + 1e-6))
                overlap_pixels += inter
                overlap_pairs += 1

    canvas_area = CANVAS * CANVAS
    overlap_ratio = float(overlap_pixels / canvas_area)
    max_iou = float(max(iou_list)) if iou_list else 0.0
    total_pairs = ns * (ns - 1) / 2
    overlap_pair_ratio = float(overlap_pairs / (total_pairs + 1e-6))

    total_element_area = float(np.sum(area))
    whitespace_ratio = float(1.0 - total_element_area / canvas_area)

    margin_left = float(x0.min() / CANVAS)
    margin_right = float((CANVAS - x1.max()) / CANVAS)
    margin_top = float(y0.min() / CANVAS)

    grid_n = 4
    density = np.zeros((grid_n, grid_n))
    for k in range(len(valid)):
        col = min(int(cx[k] / CANVAS * grid_n), grid_n - 1)
        row = min(int(cy[k] / CANVAS * grid_n), grid_n - 1)
        density[row, col] += area[k] / canvas_area
    density_variance = float(np.var(density))

    return np.array(
        [
            left_edge_std,
            right_edge_std,
            center_x_std,
            center_y_regularity,
            overlap_ratio,
            max_iou,
            overlap_pair_ratio,
            whitespace_ratio,
            margin_left,
            margin_right,
            margin_top,
            density_variance,
        ],
        dtype=np.float32,
    )


# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải LayoutLMv3 và hệ thống 9 mô hình ML nâng cao...")
def load_all_models():
    from transformers import LayoutLMv3Processor, LayoutLMv3Model

    base = os.path.dirname(os.path.abspath(__file__))

    def _load(name):
        path = os.path.join(base, name)
        if not os.path.exists(path):
            path = name
        return joblib.load(path)

    # Load bộ tiền xử lý dùng chung
    scaler = _load("scaler.pkl")
    pca = _load("pca.pkl")

    # Đọc cấu trúc cây toàn bộ 9 file pkl đã huấn luyện riêng biệt
    models_db = {
        "GradientBoosting": {
            "alignment": _load("gb_model_alignment.pkl"),
            "whitespace": _load("gb_model_whitespace.pkl"),
            "overlap": _load("gb_model_overlap.pkl")
        },
        "RandomForest": {
            "alignment": _load("rf_model_alignment.pkl"),
            "whitespace": _load("rf_model_whitespace.pkl"),
            "overlap": _load("rf_model_overlap.pkl")
        },
        "SVR": {
            "alignment": _load("svr_model_alignment.pkl"),
            "whitespace": _load("svr_model_whitespace.pkl"),
            "overlap": _load("svr_model_overlap.pkl")
        }
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=True
    )
    model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(device)
    model.eval()

    return scaler, pca, models_db, processor, model, device


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def extract_features(img: Image.Image, processor, model, device):
    inputs = processor(
        img, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state
    seq_len = hidden.shape[1]
    text_end = seq_len - 196

    cls_vec = hidden[0, 0, :].cpu().float().numpy()
    cls_vec = cls_vec / (np.linalg.norm(cls_vec) + 1e-8)

    patch_tokens = hidden[0, text_end:, :].cpu().float().numpy()
    vis_mean = patch_tokens.mean(axis=0)
    vis_mean = vis_mean / (np.linalg.norm(vis_mean) + 1e-8)
    vis_std = patch_tokens.std(axis=0)
    vis_std = vis_std / (np.linalg.norm(vis_std) + 1e-8)

    boxes = inputs["bbox"][0].cpu().numpy()
    mask = inputs["attention_mask"][0].cpu().numpy().astype(bool)
    geo_vec = compute_geometric_features(boxes[mask])

    feat_raw = np.concatenate([cls_vec, vis_mean, vis_std, geo_vec]).reshape(1, -1)
    return feat_raw, geo_vec


def predict_scores(feat_raw, scaler, pca, active_models, weights):
    # Tiến hành transform đặc trưng đầu vào qua Scaler và PCA
    feat_sc = scaler.transform(feat_raw)
    feat_pca = pca.transform(feat_sc[:, :-12])
    feat_f = np.hstack([feat_pca, feat_sc[:, -12:]])

    # Kích hoạt đúng mô hình chuyên biệt cho từng tiêu chí để dự đoán điểm số độc lập
    score_alignment = float(np.clip(active_models["alignment"].predict(feat_f)[0], 0, 10))
    score_whitespace = float(np.clip(active_models["whitespace"].predict(feat_f)[0], 0, 10))
    score_overlap = float(np.clip(active_models["overlap"].predict(feat_f)[0], 0, 10))

    # Tính điểm tổng hợp dựa trên trọng số mà người dùng cấu hình ở Sidebar
    score_overall = round(
        weights["alignment"] * score_alignment
        + weights["whitespace"] * score_whitespace
        + weights["overlap"] * score_overlap,
        2,
    )

    verdict_map = [
        (8, "🟢 Xuất sắc"),
        (6.5, "🟡 Tốt"),
        (5, "🟠 Trung bình"),
        (0, "🔴 Cần cải thiện"),
    ]
    verdict = next(v for t, v in verdict_map if score_overall >= t)

    return {
        "score_alignment": score_alignment,
        "score_whitespace": score_whitespace,
        "score_overlap": score_overlap,
        "score_overall": score_overall,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🎨 Đánh giá Chất lượng Thiết kế Đồ họa")
st.caption(
    "Mô hình: LayoutLMv3 + Geometric Features → Đa mô hình chuyên biệt  |  "
    "Tiêu chí: Alignment · Whitespace · Overlap  |  "
    "MSSV: 24022956 – Nguyễn Hồng Đăng"
)

st.divider()

# ── Sidebar: settings ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")

    model_choice = st.selectbox(
        "Chọn mô hình ML",
        ["GradientBoosting", "RandomForest", "SVR"],
        index=0,
        help="Hệ thống sẽ tự động gọi 3 file mô hình con tương ứng với thuật toán được lựa chọn.",
    )

    st.subheader("Trọng số tiêu chí")
    w_align = st.slider("Alignment", 0, 100, 33)
    w_white = st.slider("Whitespace", 0, 100, 33)
    w_over = st.slider("Overlap", 0, 100, 34)
    total_w = w_align + w_white + w_over

    if total_w != 100:
        st.warning(f"⚠️ Tổng trọng số = {total_w}% (phải = 100%). Đang tự động chuẩn hóa.")
    weights = {
        "alignment": w_align / 100,
        "whitespace": w_white / 100,
        "overlap": w_over / 100,
    }
    # normalize safely
    s = sum(weights.values())
    weights = {k: v / s for k, v in weights.items()}

    st.divider()
    st.info(
        "**Về mô hình nâng cấp**\n\n"
        "- Đã kích hoạt hệ thống **9 mô hình song song**.\n"
        "- Mỗi tiêu chí chấm điểm sở hữu một mô hình trí tuệ nhân tạo riêng biệt.\n"
        "- Thanh trượt trọng số phản ánh chính xác điểm số tổng hợp dựa trên độ ưu tiên bố cục."
    )

# ── Main: upload ──────────────────────────────────────────────────────
col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("📁 Tải ảnh thiết kế lên")
    uploaded = st.file_uploader(
        "Chọn file ảnh (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        img_bytes = uploaded.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Upscale nếu quá nhỏ
        w, h = img.size
        if w < 224 or h < 224:
            scale = max(224 / w, 224 / h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        with col_preview:
            st.subheader("🖼️ Xem trước")
            st.image(img, use_container_width=True, caption=f"{uploaded.name}  ({img.size[0]}×{img.size[1]})")

        run_btn = st.button("▶️ Chấm điểm thiết kế", type="primary", use_container_width=True)

        if run_btn:
            # Load tất cả artifacts của hệ thống vào cache
            try:
                scaler, pca, models_db, processor, lm_model, device = load_all_models()
            except Exception as e:
                st.error(f"❌ Không thể tải hệ thống mô hình: {e}")
                st.stop()

            # Lấy ra bộ 3 mô hình thuộc thuật toán người dùng vừa click lựa chọn ở Sidebar
            active_models = models_db[model_choice]

            with st.spinner("⏳ Đang trích xuất đặc trưng (LayoutLMv3)..."):
                try:
                    feat_raw, geo_vec = extract_features(img, processor, lm_model, device)
                except Exception as e:
                    st.error(f"❌ Lỗi trích xuất đặc trưng: {e}")
                    st.stop()

            with st.spinner("🤖 Đang chạy tính toán điểm số qua các mô hình chuyên biệt..."):
                result = predict_scores(feat_raw, scaler, pca, active_models, weights)

            # ── Results ──────────────────────────────────────────
            st.divider()
            st.subheader("📊 Kết quả chấm điểm")

            # Overall score big card
            verdict_color = {
                "🟢 Xuất sắc": "green",
                "🟡 Tốt": "#c8a800",
                "🟠 Trung bình": "orange",
                "🔴 Cần cải thiện": "red",
            }.get(result["verdict"], "gray")

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1e1e2e, #2d2d44);
                    border-radius: 16px;
                    padding: 24px 32px;
                    text-align: center;
                    margin-bottom: 20px;
                    border: 1px solid #444;
                ">
                    <div style="font-size: 42px; font-weight: 800; color: white;">
                        {result['score_overall']:.2f} <span style="font-size:20px;color:#aaa;">/ 10</span>
                    </div>
                    <div style="font-size: 20px; margin-top: 8px; color: {verdict_color}; font-weight: 600;">
                        {result['verdict']}
                    </div>
                    <div style="font-size: 13px; margin-top: 6px; color: #888;">
                        Mô hình cốt lõi: {model_choice}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Per-criterion scores
            c1, c2, c3 = st.columns(3)
            criteria = [
                (c1, "📐 Alignment",  result["score_alignment"],  weights["alignment"]),
                (c2, "⬜ Whitespace", result["score_whitespace"], weights["whitespace"]),
                (c3, "🔀 Overlap",    result["score_overlap"],    weights["overlap"]),
            ]
            for col, title, score, w_val in criteria:
                bar_pct = int(np.clip(score * 10, 0, 100))
                bar_color = "#4ade80" if score >= 7 else "#facc15" if score >= 5 else "#f87171"
                with col:
                    st.markdown(
                        f"""
                        <div style="
                            background:#1e1e2e; border-radius:12px;
                            padding:16px; text-align:center; border:1px solid #333;
                        ">
                            <div style="font-size:14px;color:#aaa;">{title}</div>
                            <div style="font-size:28px;font-weight:700;color:white;margin:6px 0;">
                                {score:.2f}
                            </div>
                            <div style="background:#333;border-radius:8px;height:8px;overflow:hidden;">
                                <div style="width:{bar_pct}%;height:100%;background:{bar_color};
                                            border-radius:8px;transition:width 0.5s;"></div>
                            </div>
                            <div style="font-size:11px;color:#666;margin-top:4px;">
                                trọng số {w_val:.0%}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Geometric features table
            st.divider()
            st.subheader("🔍 Chi tiết Geometric Features")

            import pandas as pd

            rows = []
            for i, name in enumerate(GEO_FEATURE_NAMES):
                val = geo_vec[i]
                group, desc = GEO_DESCRIPTIONS.get(name, ("–", "–"))
                rows.append(
                    {
                        "Tiêu chí": group,
                        "Tên đặc trưng": name,
                        "Giá trị": f"{val:.4f}",
                        "Mô tả": desc,
                    }
                )

            df_geo = pd.DataFrame(rows)
            st.dataframe(
                df_geo,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Tiêu chí": st.column_config.TextColumn(width="small"),
                    "Tên authoritarian đặc trưng": st.column_config.TextColumn(width="medium"),
                    "Giá trị": st.column_config.TextColumn(width="small"),
                    "Mô tả": st.column_config.TextColumn(width="large"),
                },
            )

    else:
        with col_preview:
            st.subheader("🖼️ Xem trước")
            st.info("Tải ảnh lên để xem trước và chấm điểm.")

# ── Footer ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Đồ án môn Machine Learning for Design · ĐHCN - ĐHQGHN · "
    "MSSV 24022956 Nguyễn Hồng Đăng"
)
