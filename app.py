import streamlit as st
import pickle
import numpy as np
import os
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineScope · Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e0d0;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero-wrap {
    text-align: center;
    padding: 3rem 0 2rem;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-weight: 900;
    color: #f5c842;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0;
    text-shadow: 0 0 60px rgba(245,200,66,0.3);
}

.hero-sub {
    margin-top: 0.6rem;
    font-size: 1rem;
    font-weight: 300;
    color: #8a8070;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.divider {
    border: none;
    border-top: 1px solid #2a2820;
    margin: 1.6rem auto;
    width: 60%;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #13121a !important;
    border: 1px solid #2e2c3a !important;
    border-radius: 12px !important;
    color: #e8e0d0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem 1.2rem !important;
    min-height: 160px !important;
    transition: border-color 0.25s;
    resize: vertical !important;
}

.stTextArea textarea:focus {
    border-color: #f5c842 !important;
    box-shadow: 0 0 0 2px rgba(245,200,66,0.12) !important;
}

.stTextArea label {
    color: #8a8070 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #f5c842 0%, #e6a800 100%) !important;
    color: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 24px rgba(245,200,66,0.25) !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result cards ── */
.result-card {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.6rem;
    text-align: center;
    animation: fadeUp 0.5s ease;
}

.result-positive {
    background: linear-gradient(135deg, #0d1f0f 0%, #102215 100%);
    border: 1px solid #1e4a22;
}

.result-negative {
    background: linear-gradient(135deg, #1f0d0d 0%, #221010 100%);
    border: 1px solid #4a1e1e;
}

.result-emoji {
    font-size: 3rem;
    margin-bottom: 0.4rem;
}

.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0;
}

.label-positive { color: #4ade80; }
.label-negative { color: #f87171; }

.result-confidence {
    font-size: 0.88rem;
    color: #7a7268;
    margin-top: 0.5rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.conf-bar-wrap {
    background: #1a1820;
    border-radius: 50px;
    height: 6px;
    margin: 0.8rem auto 0;
    max-width: 280px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.8s ease;
}

/* ── Info boxes ── */
.info-pill {
    display: inline-block;
    background: #13121a;
    border: 1px solid #2a2820;
    border-radius: 50px;
    padding: 0.3rem 1rem;
    font-size: 0.78rem;
    color: #5a5248;
    margin: 0.3rem 0.2rem;
    letter-spacing: 0.05em;
}

.pills-row {
    text-align: center;
    margin-top: 0.6rem;
}

/* ── Sidebar / stats ── */
.stat-block {
    background: #13121a;
    border: 1px solid #2a2820;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

.stat-label {
    font-size: 0.72rem;
    color: #5a5248;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #f5c842;
}

/* ── History ── */
.history-item {
    background: #13121a;
    border: 1px solid #2a2820;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.88rem;
    color: #8a8070;
}

.history-sentiment {
    font-size: 1.1rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Spinner color ── */
.stSpinner > div { border-top-color: #f5c842 !important; }

/* ── Selectbox / radio ── */
[data-testid="stExpander"] {
    background: #13121a !important;
    border: 1px solid #2a2820 !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all three pickled artefacts once and cache them."""
    try:
        with open("linearsvc_tuned_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, scaler, vectorizer, None
    except FileNotFoundError as e:
        return None, None, None, str(e)


def predict(text: str, model, scaler, vectorizer):
    """Run the full pipeline and return (label, confidence_score)."""
    vec = vectorizer.transform([text])
    scaled = scaler.transform(vec)
    label = model.predict(scaled)[0]
    # Decision function → confidence proxy (sigmoid-like stretch to 0-1)
    try:
        dec = model.decision_function(scaled)[0]
        confidence = float(1 / (1 + np.exp(-abs(dec))))  # soft confidence
    except Exception:
        confidence = 1.0
    return label, confidence


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts
if "count_pos" not in st.session_state:
    st.session_state.count_pos = 0
if "count_neg" not in st.session_state:
    st.session_state.count_neg = 0


# ── Load models ───────────────────────────────────────────────────────────────
model, scaler, vectorizer, load_error = load_models()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <p class="hero-title">🎬 CineScope</p>
  <p class="hero-sub">AI · Movie Review · Sentiment Analyzer</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)


# ── Model status ──────────────────────────────────────────────────────────────
if load_error:
    st.error(f"⚠️ Could not load model files. Make sure these files are in the same folder as `app.py`:\n\n"
             f"- `linearsvc_tuned_model.pkl`\n- `scaler.pkl`\n- `tfidf_vectorizer.pkl`\n\n**Error:** {load_error}")
    st.stop()

# Success indicator (subtle)
st.markdown("""
<div class="pills-row">
  <span class="info-pill">✓ LinearSVC model</span>
  <span class="info-pill">✓ TF-IDF vectorizer</span>
  <span class="info-pill">✓ Scaler</span>
</div>
""", unsafe_allow_html=True)


# ── Main input ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

review_text = st.text_area(
    "Your Movie Review",
    placeholder="e.g. The cinematography was breathtaking, and the performances were absolutely riveting — a masterpiece from start to finish.",
    height=180,
    key="review_input",
    label_visibility="visible",
)

# Example buttons
col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    if st.button("✨ Try positive", use_container_width=True, key="ex_pos"):
        st.session_state["review_input"] = (
            "An absolute masterpiece! The storytelling was gripping, the performances were flawless, "
            "and the cinematography left me speechless. One of the best films I've seen in years."
        )
        st.rerun()
with col_e2:
    if st.button("💔 Try negative", use_container_width=True, key="ex_neg"):
        st.session_state["review_input"] = (
            "Terrible waste of two hours. The plot made no sense, the acting was wooden, "
            "and the dialogue was cringe-worthy. I walked out halfway through."
        )
        st.rerun()
with col_e3:
    if st.button("🗑️ Clear", use_container_width=True, key="ex_clr"):
        st.session_state["review_input"] = ""
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ── Analyse button ─────────────────────────────────────────────────────────────
analyse_clicked = st.button("🔍 Analyse Sentiment", key="analyse")

if analyse_clicked:
    text = review_text.strip()
    if not text:
        st.warning("Please enter a review before analysing.")
    elif len(text.split()) < 3:
        st.warning("Review seems too short. Add a few more words for a reliable result.")
    else:
        with st.spinner("Analysing your review…"):
            time.sleep(0.5)   # brief moment so the spinner is visible
            label, confidence = predict(text, model, scaler, vectorizer)

        # Normalise label to a string
        label_str = str(label).strip().lower()
        is_positive = label_str in ("1", "positive", "pos", "good")

        emoji   = "🎉" if is_positive else "💔"
        word    = "Positive" if is_positive else "Negative"
        css_cls = "result-positive" if is_positive else "result-negative"
        lbl_cls = "label-positive" if is_positive else "label-negative"
        bar_col = "#4ade80" if is_positive else "#f87171"
        conf_pct = int(confidence * 100)

        st.markdown(f"""
<div class="result-card {css_cls}">
  <div class="result-emoji">{emoji}</div>
  <p class="result-label {lbl_cls}">{word} Review</p>
  <p class="result-confidence">Confidence · {conf_pct}%</p>
  <div class="conf-bar-wrap">
    <div class="conf-bar-fill" style="width:{conf_pct}%; background:{bar_col};"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Save to history
        short = text[:60] + ("…" if len(text) > 60 else "")
        st.session_state.history.insert(0, {
            "text": short, "label": word, "emoji": emoji, "conf": conf_pct
        })
        if is_positive:
            st.session_state.count_pos += 1
        else:
            st.session_state.count_neg += 1


# ── Stats row ─────────────────────────────────────────────────────────────────
total = st.session_state.count_pos + st.session_state.count_neg
if total > 0:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-block">
          <div class="stat-label">Total Analysed</div>
          <div class="stat-value">{total}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-block">
          <div class="stat-label">Positive</div>
          <div class="stat-value" style="color:#4ade80">{st.session_state.count_pos}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-block">
          <div class="stat-label">Negative</div>
          <div class="stat-value" style="color:#f87171">{st.session_state.count_neg}</div>
        </div>""", unsafe_allow_html=True)


# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    with st.expander("📜 Recent analyses", expanded=False):
        for item in st.session_state.history[:10]:
            color = "#4ade80" if item["label"] == "Positive" else "#f87171"
            st.markdown(f"""
<div class="history-item">
  <span>{item['text']}</span>
  <span class="history-sentiment" style="color:{color}">{item['emoji']} {item['label']} ({item['conf']}%)</span>
</div>""", unsafe_allow_html=True)
        if st.button("Clear history", key="clr_hist"):
            st.session_state.history = []
            st.session_state.count_pos = 0
            st.session_state.count_neg = 0
            st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<br><hr class="divider">
<p style="text-align:center;color:#3a3830;font-size:0.78rem;letter-spacing:0.06em;">
  CINESCOPE · POWERED BY LINEARSVC + TF-IDF · DATA SCIENCE PROJECT
</p>
""", unsafe_allow_html=True)