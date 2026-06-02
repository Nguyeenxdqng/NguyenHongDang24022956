import streamlit as st
import numpy as np
import torch
import io
import os
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

# ── Must be first Streamlit call ──────────────────────────────────────────
st.set_page_config(
    page_title="DesignScore AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEO_FEATURE_NAMES = [
    "geo_left_edge_std", "geo_right_edge_std", "geo_center_x_std",
    "geo_center_y_regularity", "geo_overlap_ratio", "geo_max_iou",
    "geo_overlap_pair_ratio", "geo_whitespace_ratio", "geo_margin_left",
    "geo_margin_right", "geo_margin_top", "geo_density_variance",
]

GEO_META = {
    "geo_left_edge_std":       {"group":"Alignment",  "label":"Lệch cạnh trái",   "good":"low",  "desc":"Nhỏ = các phần tử căn trái đều"},
    "geo_right_edge_std":      {"group":"Alignment",  "label":"Lệch cạnh phải",   "good":"low",  "desc":"Nhỏ = các phần tử căn phải đều"},
    "geo_center_x_std":        {"group":"Alignment",  "label":"Lệch tâm ngang",   "good":"low",  "desc":"Nhỏ = canh giữa đều"},
    "geo_center_y_regularity": {"group":"Alignment",  "label":"Đều hàng dọc",     "good":"high", "desc":"Lớn = khoảng cách hàng đều nhau"},
    "geo_overlap_ratio":       {"group":"Overlap",    "label":"Tỉ lệ chồng lấp",  "good":"low",  "desc":"Nhỏ = ít phần tử che nhau"},
    "geo_max_iou":             {"group":"Overlap",    "label":"Chồng lấp lớn nhất","good":"low", "desc":"Nhỏ = không có cặp nào bị che nặng"},
    "geo_overlap_pair_ratio":  {"group":"Overlap",    "label":"% cặp chồng nhau", "good":"low",  "desc":"Nhỏ = thiết kế gọn gàng"},
    "geo_whitespace_ratio":    {"group":"Whitespace", "label":"Không gian trắng",  "good":"mid",  "desc":"0.4–0.6 = cân đối nhất"},
    "geo_margin_left":         {"group":"Whitespace", "label":"Lề trái",           "good":"high", "desc":"Lớn = có không gian thở"},
    "geo_margin_right":        {"group":"Whitespace", "label":"Lề phải",           "good":"high", "desc":"Cân đối lề hai bên"},
    "geo_margin_top":          {"group":"Whitespace", "label":"Lề trên",           "good":"high", "desc":"Có khoảng thở phía trên"},
    "geo_density_variance":    {"group":"Whitespace", "label":"Variance mật độ",   "good":"low",  "desc":"Nhỏ = phân bố phần tử đều"},
}

# ══════════════════════════════════════════════════════════════════════════
# CSS INJECTION — Dark editorial theme
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:       #07090F;
    --surface:  #0D1019;
    --surface2: #141824;
    --border:   #1C2235;
    --border2:  #252D42;
    --blue:     #4F8EF7;
    --blue-glow:#4F8EF720;
    --green:    #22D18B;
    --yellow:   #F5C518;
    --orange:   #F5892A;
    --red:      #F25C5C;
    --text:     #E2E8F8;
    --muted:    #5C6A88;
    --font:     'DM Sans', sans-serif;
    --mono:     'DM Mono', monospace;
}

/* ── App shell ── */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: var(--bg) !important;
    font-family: var(--font) !important;
}
[data-testid="stToolbar"] { display: none; }
footer { display: none; }
#MainMenu { display: none; }

/* ── Main content ── */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--blue) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-family: var(--font) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--blue) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: .3px !important;
    box-shadow: 0 0 24px var(--blue-glow) !important;
    transition: all .2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 32px var(--blue-glow) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--blue) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Text overrides ── */
h1, h2, h3, h4 { font-family: var(--font) !important; color: var(--text) !important; }
p, li, span { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# GEOMETRIC FEATURES COMPUTATION
# ══════════════════════════════════════════════════════════════════════════
def compute_geometric_features(boxes_raw: np.ndarray) -> np.ndarray:
    CANVAS = 1000.0
    valid = boxes_raw[(boxes_raw[:, 2] > boxes_raw[:, 0]) &
                      (boxes_raw[:, 3] > boxes_raw[:, 1])]
    if len(valid) < 2:
        return np.zeros(12, dtype=np.float32)

    x0, y0, x1, y1 = valid[:,0], valid[:,1], valid[:,2], valid[:,3]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w  = (x1 - x0).clip(min=0)
    h  = (y1 - y0).clip(min=0)
    area = w * h

    rng_x0 = x0.max() - x0.min() + 1e-6
    rng_x1 = x1.max() - x1.min() + 1e-6
    rng_cx = cx.max() - cx.min() + 1e-6
    left_edge_std  = float(np.std(x0) / rng_x0)
    right_edge_std = float(np.std(x1) / rng_x1)
    center_x_std   = float(np.std(cx) / rng_cx)

    sorted_cy = np.sort(np.unique(cy.round(-1)))
    if len(sorted_cy) >= 2:
        gaps = np.diff(sorted_cy)
        center_y_regularity = float(1.0 - np.std(gaps) / (np.mean(gaps) + 1e-6))
    else:
        center_y_regularity = 1.0

    n = len(valid)
    overlap_pixels, iou_list, overlap_pairs = 0.0, [], 0
    sample = valid[:min(n, 60)]
    ns = len(sample)
    sx0,sy0,sx1,sy1 = sample[:,0],sample[:,1],sample[:,2],sample[:,3]
    for i in range(ns):
        for j in range(i+1, ns):
            ix0 = max(sx0[i], sx0[j]); iy0 = max(sy0[i], sy0[j])
            ix1 = min(sx1[i], sx1[j]); iy1 = min(sy1[i], sy1[j])
            inter = max(0, ix1-ix0) * max(0, iy1-iy0)
            if inter > 0:
                a_i = (sx1[i]-sx0[i]) * (sy1[i]-sy0[i])
                a_j = (sx1[j]-sx0[j]) * (sy1[j]-sy0[j])
                iou_list.append(inter / (a_i + a_j - inter + 1e-6))
                overlap_pixels += inter
                overlap_pairs += 1

    canvas_area = CANVAS * CANVAS
    overlap_ratio      = float(overlap_pixels / canvas_area)
    max_iou            = float(max(iou_list)) if iou_list else 0.0
    overlap_pair_ratio = float(overlap_pairs / (ns*(ns-1)/2 + 1e-6))

    total_element_area = float(np.sum(area))
    whitespace_ratio   = float(1.0 - total_element_area / canvas_area)
    margin_left  = float(x0.min() / CANVAS)
    margin_right = float((CANVAS - x1.max()) / CANVAS)
    margin_top   = float(y0.min() / CANVAS)

    grid_n = 4
    density = np.zeros((grid_n, grid_n))
    for k in range(len(valid)):
        col = min(int(cx[k] / CANVAS * grid_n), grid_n - 1)
        row = min(int(cy[k] / CANVAS * grid_n), grid_n - 1)
        density[row, col] += area[k] / canvas_area
    density_variance = float(np.var(density))

    return np.array([left_edge_std, right_edge_std, center_x_std,
                     center_y_regularity, overlap_ratio, max_iou, overlap_pair_ratio,
                     whitespace_ratio, margin_left, margin_right, margin_top,
                     density_variance], dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_ml_models():
    import joblib
    try:
        gb    = joblib.load("gb_model.pkl")
        rf    = joblib.load("rf_model.pkl")
        svr   = joblib.load("svr_model.pkl")
        sc    = joblib.load("scaler.pkl")
        pca   = joblib.load("pca.pkl")
        return gb, rf, svr, sc, pca, None
    except FileNotFoundError as e:
        return None, None, None, None, None, str(e)

@st.cache_resource(show_spinner=False)
def load_layoutlm():
    try:
        import pytesseract
        from transformers import LayoutLMv3Processor, LayoutLMv3Model
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        proc  = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
        model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(DEVICE)
        model.eval()
        return proc, model, None
    except Exception as e:
        return None, None, str(e)

# ══════════════════════════════════════════════════════════════════════════
# SCORE FUNCTION (PCA bug fixed: :-12 instead of :1536)
# ══════════════════════════════════════════════════════════════════════════
def score_image(img: Image.Image, gb_model, rf_model, svr_model, scaler, pca, processor, lm_model):
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w < 224 or h < 224:
        scale = max(224/w, 224/h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    inputs = processor(img, return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = lm_model(**inputs)

    hidden   = outputs.last_hidden_state
    seq_len  = hidden.shape[1]
    N_PATCHES = 196
    text_end = seq_len - N_PATCHES

    cls_vec = hidden[0, 0, :].cpu().float().numpy()
    cls_vec = cls_vec / (np.linalg.norm(cls_vec) + 1e-8)

    patch_tokens = hidden[0, text_end:, :].cpu().float().numpy()
    vis_mean = patch_tokens.mean(axis=0); vis_mean /= (np.linalg.norm(vis_mean)+1e-8)
    vis_std  = patch_tokens.std(axis=0);  vis_std  /= (np.linalg.norm(vis_std)+1e-8)

    boxes = inputs['bbox'][0].cpu().numpy()
    mask  = inputs['attention_mask'][0].cpu().numpy().astype(bool)
    geo_vec = compute_geometric_features(boxes[mask])

    feat_raw = np.concatenate([cls_vec, vis_mean, vis_std, geo_vec]).reshape(1, -1)
    feat_sc  = scaler.transform(feat_raw)

    # ✅ BUG FIX: dùng :-12 thay vì :1536
    feat_pca   = pca.transform(feat_sc[:, :-12])          # 2304 → 100
    feat_final = np.hstack([feat_pca, feat_sc[:, -12:]])   # 100 + 12 = 112

    s_gb  = float(np.clip(gb_model.predict(feat_final)[0],  0, 10))
    s_rf  = float(np.clip(rf_model.predict(feat_final)[0],  0, 10))
    s_svr = float(np.clip(svr_model.predict(feat_final)[0], 0, 10))
    s_avg = round((s_gb + s_rf + s_svr) / 3, 2)

    return {"gb": s_gb, "rf": s_rf, "svr": s_svr, "avg": s_avg,
            "geo": dict(zip(GEO_FEATURE_NAMES, geo_vec.tolist()))}

# ══════════════════════════════════════════════════════════════════════════
# HTML COMPONENTS
# ══════════════════════════════════════════════════════════════════════════
def score_color(s):
    if s >= 8:   return "#22D18B"
    elif s >= 6.5: return "#F5C518"
    elif s >= 5:   return "#F5892A"
    else:          return "#F25C5C"

def verdict_text(s):
    if s >= 8:    return ("Xuất sắc",    "✦")
    elif s >= 6.5: return ("Tốt",         "◈")
    elif s >= 5:   return ("Trung bình",  "◇")
    else:          return ("Cần cải thiện","▲")

def html_big_gauge(score, title="Điểm tổng"):
    r  = 78
    cx = cy = 100
    circ   = 2 * 3.14159 * r          # 490.09
    arc    = circ * 0.75               # 270° arc
    fill   = arc * (score / 10)
    gap    = circ - fill
    offset = -(circ * 0.375)          # start at 135°
    col    = score_color(score)
    v_text, v_icon = verdict_text(score)

    return f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:4px">
  <svg width="200" height="200" viewBox="0 0 200 200">
    <!-- Glow filter -->
    <defs>
      <filter id="glow">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <!-- Track -->
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
            stroke="#1C2235" stroke-width="13"
            stroke-dasharray="{arc:.2f} {circ-arc:.2f}"
            stroke-dashoffset="{offset:.2f}"
            stroke-linecap="round"/>
    <!-- Fill -->
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
            stroke="{col}" stroke-width="13"
            stroke-dasharray="{fill:.2f} {circ-fill:.2f}"
            stroke-dashoffset="{offset:.2f}"
            stroke-linecap="round"
            filter="url(#glow)"/>
    <!-- Score number -->
    <text x="{cx}" y="{cy+6}" text-anchor="middle"
          font-family="DM Mono,monospace" font-size="38"
          font-weight="500" fill="{col}">{score:.1f}</text>
    <text x="{cx}" y="{cy+26}" text-anchor="middle"
          font-family="DM Sans,sans-serif" font-size="13"
          fill="#5C6A88">/10</text>
  </svg>
  <div style="font-family:'DM Sans',sans-serif;font-size:13px;
              color:#5C6A88;letter-spacing:.5px;text-transform:uppercase;
              margin-top:-12px">{title}</div>
  <div style="font-family:'DM Sans',sans-serif;font-size:15px;
              font-weight:600;color:{col};margin-top:2px">
    {v_icon} {v_text}
  </div>
</div>"""

def html_mini_gauge(score, label):
    r  = 40
    cx = cy = 50
    circ   = 2 * 3.14159 * r
    arc    = circ * 0.75
    fill   = arc * (score / 10)
    offset = -(circ * 0.375)
    col    = score_color(score)
    return f"""
<div style="display:flex;flex-direction:column;align-items:center">
  <svg width="100" height="100" viewBox="0 0 100 100">
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
            stroke="#1C2235" stroke-width="8"
            stroke-dasharray="{arc:.2f} {circ-arc:.2f}"
            stroke-dashoffset="{offset:.2f}" stroke-linecap="round"/>
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
            stroke="{col}" stroke-width="8"
            stroke-dasharray="{fill:.2f} {circ-fill:.2f}"
            stroke-dashoffset="{offset:.2f}" stroke-linecap="round"/>
    <text x="{cx}" y="{cy+5}" text-anchor="middle"
          font-family="DM Mono,monospace" font-size="18"
          font-weight="500" fill="{col}">{score:.1f}</text>
  </svg>
  <div style="font-family:'DM Sans',sans-serif;font-size:11px;
              color:#5C6A88;margin-top:-8px;text-align:center">{label}</div>
</div>"""

def html_card(content, pad="24px"):
    return f"""
<div style="background:#0D1019;border:1px solid #1C2235;border-radius:16px;
            padding:{pad};margin-bottom:12px">{content}</div>"""

def html_feature_bar(name, val, meta):
    good = meta["good"]
    label = meta["label"]
    desc  = meta["desc"]

    # Normalize val to 0–1 for display (cap at 1)
    bar_pct = min(float(val), 1.0) * 100

    # Score this feature: is it "good" or "bad"?
    if good == "low":
        quality = 1.0 - min(float(val), 1.0)
    elif good == "high":
        quality = min(float(val), 1.0)
    else:  # mid: optimal around 0.5
        quality = 1.0 - abs(float(val) - 0.5) * 2

    quality = max(0, min(1, quality))

    if quality >= 0.7:   bar_col = "#22D18B"
    elif quality >= 0.4: bar_col = "#F5C518"
    else:                bar_col = "#F25C5C"

    status_icon = "✓" if quality >= 0.7 else ("~" if quality >= 0.4 else "!")
    status_col  = bar_col

    return f"""
<div style="margin-bottom:14px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
    <span style="font-family:'DM Sans',sans-serif;font-size:13px;
                 color:#B0BAD0;font-weight:500">{label}</span>
    <span style="font-family:'DM Mono',monospace;font-size:12px;color:{status_col};
                 font-weight:500">{status_icon} {float(val):.3f}</span>
  </div>
  <div style="background:#1C2235;border-radius:4px;height:6px;overflow:hidden">
    <div style="background:{bar_col};width:{bar_pct:.1f}%;height:100%;
                border-radius:4px;transition:width .4s ease"></div>
  </div>
  <div style="font-family:'DM Sans',sans-serif;font-size:11px;
              color:#3D4F6A;margin-top:3px">{desc}</div>
</div>"""

def html_group_header(group, color):
    icons = {"Alignment": "⊞", "Overlap": "⊠", "Whitespace": "□"}
    return f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;margin-top:8px">
  <span style="font-size:16px">{icons.get(group,'•')}</span>
  <span style="font-family:'DM Sans',sans-serif;font-size:13px;font-weight:700;
               color:{color};text-transform:uppercase;letter-spacing:1.5px">{group}</span>
  <div style="flex:1;height:1px;background:#1C2235;margin-left:4px"></div>
</div>"""

# ══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════
def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:36px">
      <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:6px">
        <span style="font-family:'DM Mono',monospace;font-size:28px;
                     font-weight:500;color:#4F8EF7">DesignScore</span>
        <span style="font-family:'DM Mono',monospace;font-size:28px;
                     font-weight:300;color:#1C2235">AI</span>
        <span style="font-family:'DM Sans',sans-serif;font-size:12px;
                     color:#3D4F6A;font-weight:500;letter-spacing:2px;
                     text-transform:uppercase;margin-left:8px">v2.0</span>
      </div>
      <p style="font-family:'DM Sans',sans-serif;font-size:14px;
                color:#5C6A88;margin:0">
        Đánh giá chất lượng thiết kế đồ họa tự động — Alignment · Overlap · Whitespace
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load models ────────────────────────────────────────────────────────
    with st.spinner("Đang tải models..."):
        gb_model, rf_model, svr_model, scaler, pca, ml_err = load_ml_models()
        processor, lm_model, lm_err = load_layoutlm()

    # ── Model status banner ────────────────────────────────────────────────
    model_ok  = ml_err is None
    layout_ok = lm_err is None

    status_html = '<div style="display:flex;gap:10px;margin-bottom:28px;flex-wrap:wrap">'
    def badge(label, ok, detail=""):
        col = "#22D18B" if ok else "#F25C5C"
        icon = "●" if ok else "○"
        return f'<div style="background:#0D1019;border:1px solid {col}30;border-radius:8px;padding:6px 14px;display:flex;align-items:center;gap:6px"><span style="color:{col};font-size:10px">{icon}</span><span style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:#B0BAD0">{label}</span></div>'
    status_html += badge("ML Models", model_ok)
    status_html += badge("LayoutLMv3", layout_ok)
    status_html += badge(f"Device: {DEVICE.upper()}", True)
    status_html += '</div>'
    st.markdown(status_html, unsafe_allow_html=True)

    if not model_ok:
        st.error(f"⚠️ Không tìm thấy file model: `{ml_err}`\n\nHãy chạy notebook v2 để train và lưu model trước.", icon="🔴")
        st.code("# Chạy Cell 11 trong notebook:\njoblib.dump(gb_model, 'gb_model.pkl')\njoblib.dump(scaler,   'scaler.pkl')\njoblib.dump(pca,      'pca.pkl')")
        return

    if not layout_ok:
        st.error(f"⚠️ Lỗi tải LayoutLMv3: {lm_err}", icon="🔴")
        return

    # ── Upload section ─────────────────────────────────────────────────────
    st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:#5C6A88;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:10px">Upload thiết kế</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Kéo thả hoặc click để chọn file ảnh",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        # Empty state
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    background:#0D1019;border:1px solid #1C2235;border-radius:16px;
                    padding:60px 24px;margin-top:12px">
          <div style="font-size:48px;margin-bottom:16px;opacity:.3">⬆</div>
          <p style="font-family:'DM Sans',sans-serif;font-size:15px;color:#3D4F6A;margin:0">
            Upload ảnh thiết kế để bắt đầu đánh giá
          </p>
          <p style="font-family:'DM Sans',sans-serif;font-size:12px;color:#2A3448;margin-top:6px">
            Hỗ trợ: PNG · JPG · JPEG · WEBP
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Process ────────────────────────────────────────────────────────────
    img_bytes = uploaded.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Two-column layout: image | controls
    col_img, col_ctrl = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:#3D4F6A;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">Thiết kế đầu vào</p>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:11px;color:#2A3448;margin-top:6px">{uploaded.name}  ·  {img.size[0]}×{img.size[1]}px</p>', unsafe_allow_html=True)

    with col_ctrl:
        st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:#3D4F6A;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px">Phân tích</p>', unsafe_allow_html=True)

        # Info cards
        st.markdown(html_card(f"""
          <p style="font-family:'DM Sans',sans-serif;font-size:13px;color:#5C6A88;margin:0 0 8px 0">Pipeline</p>
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            {"".join(f'<span style="background:#141824;border:1px solid #252D42;border-radius:6px;padding:4px 10px;font-family:DM Mono,monospace;font-size:11px;color:#4F8EF7">{t}</span>'
                     for t in ["LayoutLMv3","PCA(100)","GBM","RF","SVR"])}
          </div>
        """), unsafe_allow_html=True)

        run = st.button("🔍  Đánh giá thiết kế", use_container_width=True)

    # ── Run inference ──────────────────────────────────────────────────────
    if run or "result" in st.session_state:

        if run:
            with st.spinner("Đang phân tích thiết kế..."):
                try:
                    result = score_image(img, gb_model, rf_model, svr_model,
                                         scaler, pca, processor, lm_model)
                    st.session_state["result"] = result
                    st.session_state["img"]    = img
                except Exception as e:
                    st.error(f"Lỗi khi phân tích: {e}")
                    return

        result = st.session_state.get("result")
        if result is None:
            return

        st.markdown("<hr style='margin:32px 0;border-color:#1C2235'>", unsafe_allow_html=True)
        st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:#3D4F6A;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:20px">Kết quả đánh giá</p>', unsafe_allow_html=True)

        # ── Score gauges ──────────────────────────────────────────────────
        g_col1, g_col2, g_col3, g_col4 = st.columns([1.5, 1, 1, 1])

        with g_col1:
            st.components.v1.html(html_card(html_big_gauge(result["avg"], "Điểm tổng"), "32px"), height=260)
        with g_col2:
            st.components.v1.html(html_card(html_mini_gauge(result["gb"],  "Gradient\nBoosting"), "20px"), height=160)
        with g_col3:
            st.components.v1.html(html_card(html_mini_gauge(result["rf"],  "Random\nForest"), "20px"), height=160)
        with g_col4:
            st.components.v1.html(html_card(html_mini_gauge(result["svr"], "SVR\n(RBF)"), "20px"), height=160)

        # ── Geometric features ─────────────────────────────────────────────
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:#3D4F6A;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Phân tích đặc trưng thiết kế</p>', unsafe_allow_html=True)

        feat_cols = st.columns(3)
        groups = [
            ("Alignment",  "#4F8EF7", feat_cols[0]),
            ("Overlap",    "#F25C5C", feat_cols[1]),
            ("Whitespace", "#22D18B", feat_cols[2]),
        ]

        for group_name, group_col, col in groups:
            features_in_group = [n for n in GEO_FEATURE_NAMES if GEO_META[n]["group"] == group_name]
            bars_html = html_group_header(group_name, group_col)
            for fname in features_in_group:
                val = result["geo"].get(fname, 0)
                bars_html += html_feature_bar(fname, val, GEO_META[fname])
            with col:
                st.components.v1.html(html_card(bars_html, "20px 20px 8px"), height=360)

        # ── Suggestions ────────────────────────────────────────────────────
        suggestions = []
        geo = result["geo"]

        if geo.get("geo_left_edge_std", 0) > 0.25:
            suggestions.append(("Alignment", "Các phần tử văn bản chưa căn đều bên trái. Hãy dùng snap/align tool.", "#4F8EF7"))
        if geo.get("geo_overlap_ratio", 0) > 0.05:
            suggestions.append(("Overlap", "Một số phần tử đang chồng lấp nhau. Kiểm tra lại z-order và vị trí.", "#F25C5C"))
        if geo.get("geo_max_iou", 0) > 0.3:
            suggestions.append(("Overlap", "Phát hiện cặp phần tử chồng lấp nặng (IoU > 0.3). Cần tách ra.", "#F25C5C"))
        ws = geo.get("geo_whitespace_ratio", 0)
        if ws < 0.25:
            suggestions.append(("Whitespace", "Thiết kế quá dày đặc. Hãy tăng khoảng cách giữa các phần tử.", "#22D18B"))
        elif ws > 0.75:
            suggestions.append(("Whitespace", "Quá nhiều không gian trắng. Thiết kế có thể trông trống rỗng.", "#22D18B"))
        if geo.get("geo_density_variance", 0) > 0.005:
            suggestions.append(("Whitespace", "Phân bố phần tử không đều. Một số vùng trống, một số quá dày.", "#22D18B"))

        if suggestions:
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:#3D4F6A;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Gợi ý cải thiện</p>', unsafe_allow_html=True)
            sug_html = ""
            for group, text, col in suggestions:
                sug_html += f"""
                <div style="display:flex;gap:12px;align-items:flex-start;
                            padding:10px 0;border-bottom:1px solid #141824">
                  <span style="font-family:'DM Mono',monospace;font-size:11px;
                               color:{col};font-weight:500;padding-top:1px;
                               min-width:80px">{group}</span>
                  <span style="font-family:'DM Sans',sans-serif;font-size:13px;
                               color:#8898B8">{text}</span>
                </div>"""
            st.markdown(html_card(sug_html), unsafe_allow_html=True)
        else:
            st.markdown(html_card("""
            <div style="display:flex;align-items:center;gap:10px">
              <span style="color:#22D18B;font-size:18px">✦</span>
              <span style="font-family:'DM Sans',sans-serif;font-size:13px;color:#8898B8">
                Không phát hiện vấn đề rõ ràng. Thiết kế đạt các tiêu chí cơ bản.
              </span>
            </div>"""), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()