import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CineSense | Movie Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS - MODERN DARK THEME DESIGN
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        opacity: 0.3;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        position: relative;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-weight: 300;
        position: relative;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        color: white;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.3);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }

    /* Input Area */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 2px solid #334155 !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    .stTextArea label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6) !important;
    }
    .stButton button:active {
        transform: translateY(0) !important;
    }

    /* Result Cards */
    .result-positive {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(21, 128, 61, 0.1) 100%);
        border: 2px solid rgba(34, 197, 94, 0.5);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: slideIn 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    .result-positive::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #22c55e, #4ade80);
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.1) 100%);
        border: 2px solid rgba(239, 68, 68, 0.5);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: slideIn 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    .result-negative::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ef4444, #f87171);
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    .emoji-big {
        font-size: 4rem;
        animation: pulse 2s infinite;
        display: block;
        margin-bottom: 0.5rem;
    }
    .sentiment-text {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .sentiment-positive { color: #4ade80; }
    .sentiment-negative { color: #f87171; }

    /* Circular Gauge */
    .gauge-container {
        position: relative;
        width: 150px;
        height: 150px;
        margin: 0 auto;
    }
    .gauge-bg {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: conic-gradient(
            from 0deg,
            rgba(99, 102, 241, 0.2) 0deg,
            rgba(99, 102, 241, 0.2) 360deg
        );
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }
    .gauge-fill {
        position: absolute;
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: conic-gradient(
            from 0deg,
            #6366f1 0deg,
            #8b5cf6 var(--fill-degree, 0deg),
            rgba(99, 102, 241, 0.1) var(--fill-degree, 0deg),
            rgba(99, 102, 241, 0.1) 360deg
        );
    }
    .gauge-inner {
        position: absolute;
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: #0f172a;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 2;
    }
    .gauge-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        line-height: 1;
    }
    .gauge-label {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Sample Cards */
    .sample-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .sample-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.9rem;
        color: #e2e8f0;
    }
    .sample-card:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
    .sample-card.active {
        background: rgba(99, 102, 241, 0.3);
        border-color: #6366f1;
    }

    /* Metrics */
    .metric-box {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* File Uploader */
    .stFileUploader {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed #334155;
    }
    .stFileUploader:hover {
        border-color: #6366f1;
    }

    /* DataFrame */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Footer */
    .footer-modern {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255,255,255,0.1);
        color: #64748b;
        font-size: 0.85rem;
    }
    .footer-modern a {
        color: #6366f1;
        text-decoration: none;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Spinner */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# BULLETPROOF MODEL LOADER
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_all_models():
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"

    def load_file(filepath):
        errors = []
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f), "pickle"
        except Exception as e:
            errors.append(f"pickle: {str(e)[:80]}")
        try:
            return joblib.load(filepath), "joblib"
        except Exception as e:
            errors.append(f"joblib: {str(e)[:80]}")
        try:
            obj = np.load(filepath, allow_pickle=True)
            return obj.item() if hasattr(obj, "item") else obj, "numpy"
        except Exception as e:
            errors.append(f"numpy: {str(e)[:80]}")
        raise RuntimeError(" | ".join(errors))

    def patch_scaler(obj):
        if hasattr(obj, "__class__") and obj.__class__.__name__ == "MaxAbsScaler":
            if not hasattr(obj, "clip"):
                obj.clip = False
        return obj

    files = {"tfidf": "tfidf_vectorizer.pkl", "scaler": "scaler.pkl", "model": "linearsvc_tuned_model.pkl"}
    loaded = {}
    load_info = {}

    try:
        for key, filename in files.items():
            filepath = models_dir / filename
            if not filepath.exists():
                return None, None, None, False, f"File not found: {filepath}", {}
            obj, method = load_file(filepath)
            obj = patch_scaler(obj)
            loaded[key] = obj
            load_info[key] = f"{method} ({type(obj).__name__})"
        return loaded["tfidf"], loaded["scaler"], loaded["model"], True, "", load_info
    except Exception as e:
        return None, None, None, False, str(e), {}

tfidf_vectorizer, scaler, model, MODELS_LOADED, LOAD_ERROR, LOAD_INFO = load_all_models()

# ═══════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def predict_sentiment(text: str):
    if not MODELS_LOADED:
        raise RuntimeError(f"Models not loaded: {LOAD_ERROR}")
    X_tfidf = tfidf_vectorizer.transform([text])
    try:
        X_scaled = scaler.transform(X_tfidf)
    except TypeError:
        X_scaled = scaler.transform(X_tfidf.toarray())
    prediction = model.predict(X_scaled)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        confidence = float(max(proba)) * 100
    else:
        confidence = 50.0
    label_map = {0: "Negative", 1: "Positive", "neg": "Negative", "pos": "Positive"}
    sentiment = label_map.get(prediction, str(prediction))
    return {
        "sentiment": sentiment,
        "label": int(prediction) if isinstance(prediction, (int, np.integer)) else prediction,
        "confidence": round(confidence, 2),
    }

def batch_predict(texts: list):
    if not MODELS_LOADED:
        raise RuntimeError(f"Models not loaded: {LOAD_ERROR}")
    X_tfidf = tfidf_vectorizer.transform(texts)
    try:
        X_scaled = scaler.transform(X_tfidf)
    except TypeError:
        X_scaled = scaler.transform(X_tfidf.toarray())
    predictions = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_scaled)
        confidences = np.max(probas, axis=1) * 100
    else:
        confidences = np.ones(len(texts)) * 50.0
    results = []
    for pred, conf in zip(predictions, confidences):
        sentiment = "Positive" if pred in [1, "pos"] else "Negative"
        results.append({"sentiment": sentiment, "confidence": round(float(conf), 2)})
    return results

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎬</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: white;">CineSense</div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem;">AI-Powered Movie Review Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🧠 Model Pipeline**")
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.5rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="background: #6366f1; color: white; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">01</span>
            <span style="color: #e2e8f0; font-size: 0.9rem;">TF-IDF Vectorizer</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="background: #8b5cf6; color: white; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">02</span>
            <span style="color: #e2e8f0; font-size: 0.9rem;">MaxAbsScaler</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="background: #a855f7; color: white; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">03</span>
            <span style="color: #e2e8f0; font-size: 0.9rem;">Calibrated LinearSVC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if MODELS_LOADED and LOAD_INFO:
        st.markdown("---")
        st.markdown("**✅ System Status**")
        for k, v in LOAD_INFO.items():
            st.markdown(f"<div style='color: #4ade80; font-size: 0.85rem;'>● {k}: {v}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📊 Stats**")
    if "history" not in st.session_state:
        st.session_state.history = []
    total = len(st.session_state.history)
    pos = sum(1 for h in st.session_state.history if h["sentiment"] == "Positive")
    neg = total - pos
    st.markdown(f"<div style='color: #94a3b8; font-size: 0.9rem;'>Reviews analyzed: <b style='color: white;'>{total}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color: #94a3b8; font-size: 0.9rem;'>😊 Positive: <b style='color: #4ade80;'>{pos}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color: #94a3b8; font-size: 0.9rem;'>😠 Negative: <b style='color: #f87171;'>{neg}</b></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #475569; font-size: 0.75rem; margin-top: 2rem;'>CineSense v1.0<br>Built with Streamlit & Scikit-Learn</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎬 CineSense</div>
    <div class="hero-subtitle">Instant AI-powered sentiment analysis for movie reviews</div>
    <div class="hero-badge">🤖 LinearSVC • 📊 TF-IDF • 🎯 Calibrated</div>
</div>
""", unsafe_allow_html=True)

if not MODELS_LOADED:
    st.error("❌ Failed to load model files from `models/` folder")
    st.code(LOAD_ERROR, language="text")
    st.info("""
    **Troubleshooting:**
    1. Make sure your `.pkl` files are inside a folder named exactly `models/` (next to `app.py`)
    2. Check that the files aren't corrupted
    3. If you saved them with `joblib.dump()`, they should still load — the app tries both pickle and joblib
    """)
    st.stop()

# ═══════════════════════════════════════════════════════════════
# INPUT SECTION
# ═══════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>✍️ Enter Review</div>", unsafe_allow_html=True)

sample_reviews = {
    "Inception (Positive)": "Inception is a masterpiece of modern cinema. Christopher Nolan delivers a mind-bending narrative that keeps you on the edge of your seat. The visual effects are stunning, the performances are top-notch, and Hans Zimmer's score is absolutely iconic. I've watched it five times and still discover new details. A true cinematic achievement.",
    "The Room (Negative)": "This movie makes no sense. The acting is wooden, the plot is nonsensical, and the dialogue is painfully awkward. The pacing is so slow it feels like the movie will never end. I walked out after 45 minutes and demanded a refund. Complete waste of time and money.",
    "Shawshank (Positive)": "The Shawshank Redemption is simply one of the greatest films ever made. Tim Robbins and Morgan Freeman deliver performances that stay with you long after the credits roll. The story of hope and friendship is beautifully told, and the ending is one of the most satisfying in film history. I cry every single time I watch it.",
    "Disaster Movie (Negative)": "I wanted to love this movie but it was a complete disaster. The CGI looks like it was made in 2002, the dialogue is cringe-worthy, and the plot has more holes than Swiss cheese. The actors seem like they don't even want to be there. Two and a half hours of my life I'll never get back."
}

# Sample selector as clickable cards
cols = st.columns(4)
selected_sample = None
for i, (name, text) in enumerate(sample_reviews.items()):
    with cols[i]:
        if st.button(name, use_container_width=True, key=f"sample_{i}"):
            selected_sample = text

# Text area with selected sample or manual input
if "review_input" not in st.session_state:
    st.session_state.review_input = ""

if selected_sample:
    st.session_state.review_input = selected_sample

review_text = st.text_area(
    "",
    value=st.session_state.review_input,
    height=180,
    placeholder="Type or paste a movie review here...",
    key="review_input",
    label_visibility="collapsed"
)

analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═══════════════════════════════════════════════════════════════

if analyze_btn and review_text.strip():
    with st.spinner("🧠 Analyzing sentiment..."):
        try:
            result = predict_sentiment(review_text)
            sentiment = result["sentiment"]
            confidence = result["confidence"]

            # Save to history
            st.session_state.history.append({
                "text": review_text[:80] + "...",
                "sentiment": sentiment,
                "confidence": confidence
            })

            st.markdown("---")
            st.markdown("<div class='section-title'>📊 Analysis Result</div>", unsafe_allow_html=True)

            # Result card with emoji and animation
            if sentiment == "Positive":
                st.markdown(f"""
                <div class="result-positive">
                    <span class="emoji-big">😊</span>
                    <div class="sentiment-text sentiment-positive">Positive Review</div>
                    <p style="color: #86efac; margin-top: 0.5rem;">This review expresses a favorable opinion about the movie.</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <span class="emoji-big">😠</span>
                    <div class="sentiment-text sentiment-negative">Negative Review</div>
                    <p style="color: #fca5a5; margin-top: 0.5rem;">This review expresses an unfavorable opinion about the movie.</p>
                </div>
                """, unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: {'#4ade80' if sentiment == 'Positive' else '#f87171'};">{sentiment}</div>
                    <div class="metric-label">Sentiment</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: #6366f1;">{confidence}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: #e2e8f0;">{len(review_text.split())}</div>
                    <div class="metric-label">Words</div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence gauge using CSS
            gauge_color = "#4ade80" if sentiment == "Positive" else "#f87171"
            fill_deg = int((confidence / 100) * 360)
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin-top: 2rem;">
                <div class="gauge-container">
                    <div class="gauge-fill" style="--fill-degree: {fill_deg}deg; background: conic-gradient(from 0deg, {gauge_color} 0deg, {gauge_color} {fill_deg}deg, rgba(255,255,255,0.05) {fill_deg}deg, rgba(255,255,255,0.05) 360deg);"></div>
                    <div class="gauge-inner">
                        <div class="gauge-value">{confidence}%</div>
                        <div class="gauge-label">Confidence</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Technical details
            with st.expander("🔧 Technical Details"):
                st.json(result)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.exception(e)

elif analyze_btn and not review_text.strip():
    st.warning("⚠️ Please enter a review text before analyzing.")

# ═══════════════════════════════════════════════════════════════
# BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("<div class='section-title'>📁 Batch Analysis</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a CSV file with a 'review' column",
    type=["csv"],
    help="Upload a CSV file containing movie reviews. Each row should have a 'review' column with the text to analyze."
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if "review" not in df.columns:
            st.error("❌ CSV must contain a column named 'review'")
        else:
            st.markdown(f"<div style='color: #94a3b8; margin-bottom: 1rem;'>📄 Found <b style='color: white;'>{len(df)}</b> reviews in uploaded file</div>", unsafe_allow_html=True)
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)

            if st.button("🚀 Analyze All Reviews", use_container_width=True):
                with st.spinner("Processing batch..."):
                    texts = df["review"].astype(str).tolist()
                    results = batch_predict(texts)

                    df["sentiment"] = [r["sentiment"] for r in results]
                    df["confidence"] = [r["confidence"] for r in results]

                    # Summary stats
                    pos_count = sum(1 for r in results if r["sentiment"] == "Positive")
                    neg_count = len(results) - pos_count
                    avg_conf = np.mean([r["confidence"] for r in results])

                    st.markdown("---")
                    st.markdown("<div class='section-title'>📊 Batch Results Summary</div>", unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #6366f1;">{len(df)}</div>
                            <div class="metric-label">Total Reviews</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #4ade80;">{pos_count}</div>
                            <div class="metric-label">Positive</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #f87171;">{neg_count}</div>
                            <div class="metric-label">Negative</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c4:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #e2e8f0;">{avg_conf:.1f}%</div>
                            <div class="metric-label">Avg Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Full results table
                    st.markdown("<div class='section-title' style='margin-top: 1.5rem;'>📋 Detailed Results</div>", unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Download
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Results as CSV",
                        csv,
                        "sentiment_results.csv",
                        "text/csv",
                        use_container_width=True
                    )

                    # Chart
                    st.markdown("<div class='section-title' style='margin-top: 1.5rem;'>📈 Sentiment Distribution</div>", unsafe_allow_html=True)
                    chart_data = pd.DataFrame({
                        "Sentiment": ["Positive", "Negative"],
                        "Count": [pos_count, neg_count]
                    })
                    st.bar_chart(chart_data.set_index("Sentiment"), color=["#4ade80", "#f87171"], use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# ═══════════════════════════════════════════════════════════════
# HISTORY SECTION
# ═══════════════════════════════════════════════════════════════

if st.session_state.history:
    st.markdown("---")
    st.markdown("<div class='section-title'>🕐 Recent Analysis History</div>", unsafe_allow_html=True)

    hist_df = pd.DataFrame(st.session_state.history[-10:][::-1])
    hist_df["confidence"] = hist_df["confidence"].astype(str) + "%"

    def color_sentiment(val):
        color = "#4ade80" if val == "Positive" else "#f87171"
        return f"color: {color}; font-weight: 600;"

    st.dataframe(
        hist_df.style.map(color_sentiment, subset=["sentiment"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "text": st.column_config.TextColumn("Review Preview", width="large"),
            "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "confidence": st.column_config.TextColumn("Confidence", width="small"),
        }
    )

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer-modern">
    <p>🎬 <b>CineSense</b> — Movie Review Sentiment Analysis</p>
    <p>Powered by <span style="color: #6366f1;">Streamlit</span> • 
    <span style="color: #8b5cf6;">Calibrated LinearSVC</span> • 
    <span style="color: #a855f7;">TF-IDF</span> • 
    <span style="color: #c084fc;">MaxAbsScaler</span></p>
    <p style="margin-top: 0.5rem; font-size: 0.75rem;">All three .pkl model files are loaded and utilized in the prediction pipeline</p>
</div>
""", unsafe_allow_html=True)
