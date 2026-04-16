"""
ConvNeXt-Base Image Classifier — Streamlit Dashboard
=====================================================
Replicates the inference logic from the notebook and adds rich
performance visualisations for general testing/demoing purposes.

Usage
-----
    pip install streamlit torch torchvision plotly pandas pillow
    streamlit run convnext_dashboard.py

Place  best_model_finetuning.pt  in the same directory before launching.
"""

import io
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from huggingface_hub import hf_hub_download, login

# ── Hugging Face Authentication ────────────────────────────────────────────────
hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Token cargado correctamente")
else:
    print("Corriendo sin token (modo limitado)")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ConvNeXt Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = [
    "Aroma22", "CLL18", "Clusters", "FM", "JazTec",
    "RT7221FP", "RT7301", "RT7521FP", "RTv7303",
    "Taurus", "XP760MA", "XP784",
]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "best_model_finetuning.pt"

# ── Representative per-class metrics (replace with actual validation numbers) ──
_PRECISION = [0.97, 0.97, 0.98, 0.82, 0.93, 0.84, 0.94, 0.81, 0.93, 0.99, 0.80, 0.98]
_RECALL    = [0.95, 0.99, 1.00, 0.75, 0.95, 0.88, 0.94, 0.78, 0.91, 0.99, 0.83, 0.97]
_F1        = [0.96, 0.98, 0.99, 0.78, 0.94, 0.86, 0.94, 0.79, 0.92, 0.99, 0.82, 0.97]
_SUPPORT   = [776, 798, 113, 12, 795, 348, 791, 835, 801, 794, 829, 778]

PERF_DF = pd.DataFrame(
    {"Class": CLASS_NAMES, "Precision": _PRECISION, "Recall": _RECALL,
     "F1-Score": _F1, "Support": _SUPPORT}
)

# Approximate confusion-matrix diagonal (accuracy per class ≈ recall)
CONF_MATRIX_DIAG = [int(r * s) for r, s in zip(_RECALL, _SUPPORT)]

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import distinctive fonts */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #A0192E, #C49A6C);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-family: 'Space Mono', monospace; font-size: 2.1rem; margin:0; letter-spacing: -1px; }
.hero p  { font-size: 1rem; opacity: 0.75; margin: 8px 0 0; }

/* Prediction badge */
.pred-badge {
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
    margin-bottom: 18px;
    box-shadow: 0 6px 24px rgba(0,0,0,.18);
}
.pred-badge .cls  { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; }
.pred-badge .conf { font-size: 1.25rem; opacity: .9; }
.pred-badge .lbl  { font-size: .82rem; opacity: .7; margin-top: 4px; }

/* KPI cards */
.kpi-card {
    background: #A0192E;
    border: 1px solid #C49A6C;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    color: #ffffff;
}
.kpi-card .val { font-family: 'Space Mono', monospace; font-size: 1.9rem; color: #ffffff; }
.kpi-card .lbl { font-size: .78rem; color: #ffffff; opacity: .8; margin-top: 4px; text-transform: uppercase; letter-spacing: .06em; }

/* Section headers */
.sec-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    letter-spacing: .04em;
    color: #ffffff;
    text-transform: uppercase;
    margin: 0 0 12px;
    border-bottom: 1px solid #C49A6C;
    padding-bottom: 6px;
}

/* Info banner */
.info-banner {
    background: #A0192E;
    border-left: 4px solid #C49A6C;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: .88rem;
    color: #ffffff;
    margin-bottom: 16px;
}

/* Tag pills */
.tag {
    display: inline-block;
    background: #C49A6C;
    color: #ffffff;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: .78rem;
    margin: 2px 3px;
    border: 1px solid #C49A6C;
}

/* Sidebar */
[data-testid="stSidebar"] { background: #A0192E; }
[data-testid="stSidebar"] * { color: #ffffff !important; }

/* Code blocks in sidebar */
[data-testid="stSidebar"] .stCodeBlock pre,
[data-testid="stSidebar"] .stCodeBlock code {
    color: #212529 !important;
    background-color: #f8f9fa !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("UARPP-LogoV2.png", width='stretch', caption="University of Arkansas System")
    st.image("itesm_logo.png", width='stretch', caption="Instituto Tecnológico y de Estudios Superiores de Monterrey")
    st.divider()
    st.markdown("## 🧬 ConvNeXt-Base")
    st.markdown("**Image Classification Dashboard**")
    st.divider()
    st.markdown("### ⚙️ Configuration")
    top_k = st.slider("Top-K predictions", 1, NUM_CLASSES, 5, key="topk")
    show_all = st.checkbox("Show full probability table", value=True)
    st.divider()
    st.markdown("### 🏷️ Classes")
    for i, cls in enumerate(CLASS_NAMES):
        st.markdown(f'<span class="tag">{i:02d} · {cls}</span>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📐 Input pipeline")
    st.code("Resize(256)\nCenterCrop(224)\nToTensor()\nNormalize(\n  μ=[.485,.456,.406]\n  σ=[.229,.224,.225]\n)", language="python")
    st.divider()
    st.markdown("### 👥 Corresponding Authors")
    st.markdown("**Model training, Research & Dashboard:** \n* Atoany Nazareth Fierro Radilla (afierror@tec.mx)\n\n* Francisco Javier Navarro (fj.navarro.barron@tec.mx)\n\n**Dataset curation & project supervision:** \n* Alan Roberto Vázquez Alcocer (alanrvazquez@tec.mx)")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧬 ConvNeXt-Base Classifier</h1>
  <p>Fine-tuned model · 12 classes · 224 × 224 input · CPU / CUDA inference</p>
</div>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.convnext_base(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m, device


eval_tfms = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_ok = False
model = device = None

if os.path.exists(MODEL_PATH):
    try:
        with st.spinner("Loading ConvNeXt-Base weights…"):
            model, device = load_model(MODEL_PATH)
        model_ok = True
        st.success(f"✅ Model ready on **{device}** — {NUM_CLASSES} classes detected", icon="🚀")
    except Exception as exc:
        st.error(f"❌ Could not load model: {exc}")
else:
    try:
        with st.spinner("Model file not found locally. Attempting to download from Hugging Face Hub…"):
            MODEL_PATH = hf_hub_download(
                repo_id="fjnavarro/rice_variety_convnext",
                filename="best_model_finetuning.pt",
                repo_type="model",
            )
            model, device = load_model(MODEL_PATH)
        model_ok = True
        st.success(f"✅ Model downloaded and loaded successfully on **{device}**", icon="🚀")
    except Exception as exc:
        st.error(f"❌ Could not download model: {exc}")
        st.warning(
        f"⚠️ `{MODEL_PATH}` not found.  "
        "Place the file in the same directory and restart the app.",
        icon="📂")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_inf, tab_perf, tab_arch = st.tabs(["🖼️  Inference", "📊  Performance", "🏛️  Architecture"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — INFERENCE
# ════════════════════════════════════════════════════════════════════════════════
with tab_inf:
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown('<p class="sec-title">Upload image</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "JPG · JPEG · PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded:
            img_pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            st.image(img_pil, width='stretch',
                     caption=f"{uploaded.name}  ·  {img_pil.size[0]} × {img_pil.size[1]} px")

    with col_right:
        if uploaded and model_ok:
            # ── Run inference ────────────────────────────────────────────────
            tensor = eval_tfms(img_pil).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            sorted_idx  = np.argsort(probs)[::-1]
            top_idx     = sorted_idx[:top_k]
            top_classes = [CLASS_NAMES[i] for i in top_idx]
            top_probs   = probs[top_idx]

            pred_cls  = top_classes[0]
            pred_conf = top_probs[0]

            # ── Badge ────────────────────────────────────────────────────────
            if pred_conf >= 0.90:
                bg, lbl = "#065f46", "High confidence ✅"
            elif pred_conf >= 0.70:
                bg, lbl = "#78350f", "Medium confidence ⚠️"
            else:
                bg, lbl = "#7f1d1d", "Low confidence ❌"

            st.markdown(f"""
            <div class="pred-badge" style="background:{bg};">
              <div class="cls">{pred_cls}</div>
              <div class="conf">{pred_conf * 100:.2f}% confidence</div>
              <div class="lbl">{lbl} &nbsp;·&nbsp; ⏱ {elapsed_ms:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Top-K bar chart ──────────────────────────────────────────────
            st.markdown(f'<p class="sec-title">Top-{top_k} predictions</p>', unsafe_allow_html=True)
            bar_colors = ["#A0192E"] + ["#C49A6C"] * (top_k - 1)
            fig_bar = go.Figure(go.Bar(
                x=[p * 100 for p in top_probs],
                y=top_classes,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{p * 100:.2f}%" for p in top_probs],
                textposition="outside",
                hovertemplate="%{y}: %{x:.3f}%<extra></extra>",
            ))
            fig_bar.update_layout(
                xaxis_title="Confidence (%)",
                yaxis=dict(autorange="reversed"),
                height=280,
                margin=dict(l=0, r=70, t=8, b=32),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="#9ca3af"),
                yaxis_tickfont=dict(size=12, family="Space Mono"),
            )
            st.plotly_chart(fig_bar, width='stretch')

            # ── All-class probability table ──────────────────────────────────
            if show_all:
                st.markdown('<p class="sec-title">Full probability distribution</p>', unsafe_allow_html=True)
                prob_df = pd.DataFrame({
                    "Rank":        range(1, NUM_CLASSES + 1),
                    "Class":       [CLASS_NAMES[i] for i in sorted_idx],
                    "Probability": [f"{probs[i] * 100:.4f}%" for i in sorted_idx],
                    "Raw logit":   [f"{logits[0][i].item():.4f}" for i in sorted_idx],
                })
                st.dataframe(prob_df, width='stretch', hide_index=True)

            # ── Probability gauge (all classes) ─────────────────────────────
            st.markdown('<p class="sec-title">Class probability gauge</p>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Bar(
                x=CLASS_NAMES,
                y=[probs[i] * 100 for i in range(NUM_CLASSES)],
                marker_color=[
                    "#A0192E" if CLASS_NAMES[i] == pred_cls else "#C49A6C"
                    for i in range(NUM_CLASSES)
                ],
                hovertemplate="%{x}: %{y:.4f}%<extra></extra>",
            ))
            fig_gauge.update_layout(
                xaxis_title="Class",
                yaxis_title="Probability (%)",
                height=280,
                margin=dict(l=0, r=0, t=8, b=60),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(tickangle=40, color="#9ca3af"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="#9ca3af"),
            )
            st.plotly_chart(fig_gauge, width='stretch')

        elif not model_ok:
            st.info("📦 Load the model file to enable inference.", icon="ℹ️")
        else:
            st.info("📂 Upload an image to begin.", icon="ℹ️")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown(
        '<div class="info-banner">'
        "ℹ️ The metrics below can be interpreted as follows: "
        "<strong>Macro Precision (0.92)</strong>: On average, when the model predicts a rice variety, it's correct 92% of the time. "
        "<strong>Macro Recall (0.91)</strong>: The model successfully identifies 91% of actual rice varieties in the validation set. "
        "<strong>Macro F1-Score (0.91)</strong>: A balanced measure showing the model performs consistently well across all 12 rice varieties."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── KPI row ─────────────────────────────────────────────────────────────
    avg_p  = 0.92 #np.mean(_PRECISION)
    avg_r  = 0.91 #np.mean(_RECALL)
    avg_f1 = 0.91 #np.mean(_F1)
    tot_s  = sum(_SUPPORT)

    k1, k2, k3, k4 = st.columns(4)
    for col, val, label in [
        (k1, f"{avg_p:.2f}", "Macro Precision"),
        (k2, f"{avg_r:.2f}", "Macro Recall"),
        (k3, f"{avg_f1:.2f}", "Macro F1-Score"),
        (k4, f"{tot_s:,}",   "Validation samples"),
    ]:
        col.markdown(
            f'<div class="kpi-card"><div class="val">{val}</div>'
            f'<div class="lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Per-class table ──────────────────────────────────────────────────────
    st.markdown('<p class="sec-title">Per-class metrics</p>', unsafe_allow_html=True)
    display_df = PERF_DF.copy()
    for col in ("Precision", "Recall", "F1-Score"):
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    st.dataframe(display_df, width='stretch', hide_index=True)

    st.divider()

    # ── Charts row 1 ────────────────────────────────────────────────────────
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<p class="sec-title">F1-Score per class</p>', unsafe_allow_html=True)
        fig_f1 = px.bar(
            PERF_DF, x="Class", y="F1-Score",
            color="F1-Score", color_continuous_scale=['#C49A6C', '#A0192E'],
            range_y=[0.8, 1.0],
            text=PERF_DF["F1-Score"].apply(lambda v: f"{v:.3f}"),
        )
        fig_f1.update_traces(textposition="outside")
        fig_f1.update_layout(
            height=360, margin=dict(t=10, b=60),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(tickangle=40, color="#9ca3af"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="#9ca3af"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_f1, width='stretch')

    with c2:
        st.markdown('<p class="sec-title">Precision vs Recall</p>', unsafe_allow_html=True)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Bar(
            name="Precision", x=CLASS_NAMES, y=_PRECISION,
            marker_color="#A0192E",
        ))
        fig_pr.add_trace(go.Bar(
            name="Recall", x=CLASS_NAMES, y=_RECALL,
            marker_color="#19A08B", # color complementario
        ))
        fig_pr.update_layout(
            barmode="group",
            yaxis=dict(range=[0.8, 1.0], gridcolor="rgba(255,255,255,0.07)", color="#9ca3af"),
            xaxis=dict(tickangle=40, color="#9ca3af"),
            height=360, margin=dict(t=10, b=60),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_pr, width='stretch')

    # ── Charts row 2 ────────────────────────────────────────────────────────
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown('<p class="sec-title">Radar: per-class F1</p>', unsafe_allow_html=True)
        cats   = CLASS_NAMES + [CLASS_NAMES[0]]
        vals   = _F1 + [_F1[0]]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            line_color="#C49A6C", fillcolor="rgba(196,154,108,.6)",
            marker=dict(size=6, color="#C49A6C"),
        ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.8, 1.0],
                                color="#9ca3af", gridcolor="rgba(255,255,255,0.1)"),#A0192E
                angularaxis=dict(color="#9ca3af"),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig_r, width='stretch')

    with c4:
        st.markdown('<p class="sec-title">Support distribution</p>', unsafe_allow_html=True)
        fig_sup = px.pie(
            PERF_DF, names="Class", values="Support",
            color_discrete_sequence=px.colors.qualitative.Light24,
            hole=0.45,
        )
        fig_sup.update_traces(textposition="outside", textinfo="label+percent")
        fig_sup.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_sup, width='stretch')

    # ── Accuracy per class (approx) ──────────────────────────────────────────
    st.divider()
    st.markdown('<p class="sec-title">Correct predictions per class (approx.)</p>', unsafe_allow_html=True)

    acc_df = pd.DataFrame({
        "Class":    CLASS_NAMES,
        "Correct":  CONF_MATRIX_DIAG,
        "Total":    _SUPPORT,
        "Accuracy": [c / t for c, t in zip(CONF_MATRIX_DIAG, _SUPPORT)],
    })
    fig_acc = px.bar(
        acc_df, x="Class", y="Accuracy",
        color="Accuracy", color_continuous_scale=['#C49A6C', '#A0192E'],
        range_y=[0.8, 1.0],
        custom_data=["Correct", "Total"],
        text=acc_df["Accuracy"].apply(lambda v: f"{v*100:.1f}%"),
    )
    fig_acc.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.3f}<br>%{customdata[0]}/%{customdata[1]}<extra></extra>",
    )
    fig_acc.update_layout(
        height=320, margin=dict(t=10, b=60),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(tickangle=40, color="#9ca3af"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="#9ca3af"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_acc, width='stretch')

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown('<p class="sec-title">ConvNeXt-Base overview</p>', unsafe_allow_html=True)
    st.markdown("""
    **ConvNeXt** is a pure convolutional architecture that incorporates design principles
    from Vision Transformers (ViT).  The *Base* variant strikes a strong accuracy/compute
    balance.  The final linear head was replaced with a `Linear(1024 → 12)` layer for
    this 12-class fine-tuning task.
    """)

    # ── Stage table ──────────────────────────────────────────────────────────
    st.markdown('<p class="sec-title">Stage breakdown</p>', unsafe_allow_html=True)
    stage_df = pd.DataFrame({
        "Stage":       ["Stem", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Classifier head"],
        "Operation":   [
            "Patchify Conv 4×4, stride 4",
            "LayerNorm downsample + 3 ConvNeXt blocks",
            "LayerNorm downsample + 3 ConvNeXt blocks",
            "LayerNorm downsample + 27 ConvNeXt blocks",
            "LayerNorm downsample + 3 ConvNeXt blocks",
            "Global Avg Pool → LayerNorm → Linear(1024, 12)",
        ],
        "Out channels": [128, 128, 256, 512, 1024, 12],
        "# Blocks":     [1,   3,   3,   27,  3,    1],
    })
    st.dataframe(stage_df, width='stretch', hide_index=True)

    # ── ConvNeXt block diagram ───────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="sec-title">ConvNeXt block internals</p>', unsafe_allow_html=True)
    st.code("""
ConvNeXt Block (per stage):
  ┌───────────────────────────────────────────────────┐
  │  Input: (B, C, H, W)                              │
  │    ↓                                              │
  │  Depthwise Conv 7×7 (groups=C, same padding)      │
  │    ↓                                              │
  │  Permute → (B, H, W, C)                           │
  │    ↓                                              │
  │  LayerNorm (eps=1e-6)                             │
  │    ↓                                              │
  │  Linear(C → 4C)   [point-wise expand]             │
  │    ↓                                              │
  │  GELU                                             │
  │    ↓                                              │
  │  Linear(4C → C)   [point-wise contract]           │
  │    ↓                                              │
  │  Permute → (B, C, H, W)                           │
  │    ↓                                              │
  │  Layer Scale (γ · x, γ learnable, init 1e-6)      │
  │    ↓                                              │
  │  Stochastic Depth (drop_path)                     │
  │    ↓                                              │
  │  + Residual ──────────────────────────────────────┘
  │    ↓
  │  Output: (B, C, H, W)
    """, language="text")

    # ── Training config table ────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="sec-title">Fine-tuning configuration</p>', unsafe_allow_html=True)
    cfg_df = pd.DataFrame({
        "Parameter":  [
            "Model file",
            "Number of classes",
            "Input resolution",
            "Preprocessing — resize",
            "Preprocessing — crop",
            "Norm mean",
            "Norm std",
            "Serialisation",
            "Inference device",
        ],
        "Value": [
            "best_model_finetuning.pt",
            "12",
            "224 × 224",
            "Resize(256)",
            "CenterCrop(224)",
            "[0.485, 0.456, 0.406]",
            "[0.229, 0.224, 0.225]",
            "torch.save (state_dict)",
            "CUDA if available, else CPU",
        ],
    })
    st.dataframe(cfg_df, width='stretch', hide_index=True)

    # ── Class list ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="sec-title">Class registry</p>', unsafe_allow_html=True)
    cls_df = pd.DataFrame({
        "Index":      list(range(NUM_CLASSES)),
        "Class name": CLASS_NAMES,
    })
    st.dataframe(cls_df, width='stretch', hide_index=True)

    # ── Loading code snippet ──────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="sec-title">Minimal inference snippet</p>', unsafe_allow_html=True)
    st.code("""
import torch, torch.nn as nn
from torchvision import models
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

CLASS_NAMES = [
    'Aroma22','CLL18','Clusters','FM','JazTec',
    'RT7221FP','RT7301','RT7521FP','RTv7303','Taurus','XP760MA','XP784'
]

# Build model
model = models.convnext_base(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 12)
model.load_state_dict(torch.load('best_model_finetuning.pt', map_location='cpu'))
model.eval()

# Transform
tfm = Compose([
    Resize(256), CenterCrop(224), ToTensor(),
    Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Infer
img = Image.open('your_image.jpg').convert('RGB')
with torch.no_grad():
    probs = torch.softmax(model(tfm(img).unsqueeze(0)), dim=1).squeeze()
print(CLASS_NAMES[probs.argmax()], f'{probs.max()*100:.2f}%')
""", language="python")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='opacity:.45;font-size:.8rem;'>"
    "ConvNeXt-Base · 12-class classifier · Built with Streamlit"
    "</center>",
    unsafe_allow_html=True,
)
