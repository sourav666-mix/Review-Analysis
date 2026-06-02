# 🎬 CineSense — Movie Review Sentiment Analysis

A modern, interactive **Streamlit web application** that classifies movie reviews as **Positive** or **Negative** using a pre-trained Machine Learning pipeline. Built with a sleek dark UI, real-time confidence gauges, batch CSV processing, and animated results.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🖼️ Screenshot

```
┌─────────────────────────────────────────────────────────────┐
│  🎬 CineSense — AI-Powered Movie Review Analysis            │
├─────────────────────────────────────────────────────────────┤
│  [😊 Inception] [😠 The Room] [😊 Shawshank] [😠 Disaster]  │
│                                                             │
│  [  Type or paste a movie review here...        ]           │
│                                                             │
│  [        🔍 Analyze Sentiment (full width)       ]         │
├─────────────────────────────────────────────────────────────┤
│  📊 Result: 😊 Positive Review  |  Confidence: 94.2%        │
│              ╭───────────╮                                    │
│             ╱    94.2%    ╲                                   │
│            │  Confidence   │                                 │
│             ╲─────────────╱                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **🎯 Single Review Analysis** | Paste any movie review and get instant sentiment + confidence score |
| **📁 Batch CSV Processing** | Upload a CSV with a `review` column to analyze hundreds of reviews at once |
| **📊 Circular Confidence Gauge** | Animated CSS gauge showing prediction confidence (0–100%) |
| **🧪 4 Sample Reviews** | One-click test buttons for quick validation |
| **📈 Sentiment Distribution Chart** | Bar chart showing Positive vs Negative counts for batch uploads |
| **🕐 Analysis History** | Sidebar tracks your last 10 predictions live |
| **🔧 Technical Details** | Expandable JSON view of raw prediction data |
| **⬇️ Export Results** | Download batch analysis as CSV |
| **🌙 Dark Theme UI** | Modern glassmorphism design with gradient backgrounds |

---

## 🧠 Model Pipeline

The app uses **three serialized ML artifacts** loaded in sequence:

| Step | File | Purpose | Type |
|------|------|---------|------|
| **1** | `tfidf_vectorizer.pkl` | Converts raw text → TF-IDF feature vectors | `CountVectorizer` / `TfidfVectorizer` |
| **2** | `scaler.pkl` | Normalizes feature values to [-1, 1] range | `MaxAbsScaler` |
| **3** | `linearsvc_tuned_model.pkl` | Classifies sentiment using calibrated probabilities | `CalibratedClassifierCV` (wraps `LinearSVC`) |

**Pipeline flow:**
```
Raw Text → TF-IDF Vectorizer → MaxAbsScaler → Calibrated LinearSVC → Sentiment + Confidence
```

---

## 📁 Folder Structure

```
movie_review/
│
├── app.py                          ← Main Streamlit application (single file)
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
│
└── models/                         ← MUST be named exactly "models"
    ├── linearsvc_tuned_model.pkl   ← Trained CalibratedClassifierCV
    ├── tfidf_vectorizer.pkl        ← Trained TF-IDF vectorizer
    └── scaler.pkl                  ← Trained MaxAbsScaler
```

> ⚠️ **Important:** The `models/` folder must be in the **same directory** as `app.py`.

---

## 🛠️ Installation

### 1. Clone or Download

```bash
git clone https://github.com/yourusername/cinesense.git
cd cinesense
```

Or manually create the folder structure and place your `.pkl` files in `models/`.

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Start the App

```bash
streamlit run app.py
```

The app will open automatically at:
- **Local:** `http://localhost:8501`
- **Network:** `http://192.168.1.x:8501`

### Single Review Analysis

1. **Type or paste** a movie review in the text area
2. **Click** "🔍 Analyze Sentiment"
3. View the **animated result card** with sentiment, confidence %, and circular gauge

### Quick Test with Samples

Click any of the **4 sample review buttons** at the top:
- 😊 **Inception** — Positive
- 😠 **The Room** — Negative
- 😊 **Shawshank** — Positive
- 😠 **Disaster Movie** — Negative

### Batch Analysis (CSV)

1. Scroll to **📁 Batch Analysis**
2. **Upload** a CSV file with a column named `review`
3. Click **"🚀 Analyze All Reviews"**
4. View the **summary dashboard** + download results

**Example CSV format:**
```csv
review
"This movie was absolutely incredible!"
"I fell asleep halfway through."
"Best film I've seen this year."
```

---

## ⚙️ requirements.txt

```txt
streamlit>=1.28.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
joblib>=1.3.0
```

---

## 🔧 Troubleshooting

### ❌ "Failed to load model files from models/ folder"

**Cause:** The app can't find your `.pkl` files.

**Fix:**
```bash
# Check your folder structure
ls models/
# Should show: linearsvc_tuned_model.pkl  scaler.pkl  tfidf_vectorizer.pkl
```

### ❌ "'MaxAbsScaler' object has no attribute 'clip'"

**Cause:** Your models were saved with **scikit-learn 1.6.1** but you're running **1.8.0**.

**Fix:** The app auto-patches this. If it still fails, downgrade sklearn:
```bash
pip install scikit-learn==1.6.1
```

### ❌ "invalid load key, '\x01'"

**Cause:** The `.pkl` file is actually a **joblib** file or is corrupted.

**Fix:** The app tries `pickle`, `joblib`, `numpy`, and `gzip` loaders automatically. If all fail, verify the file isn't corrupted:
```python
import os
print(os.path.getsize("models/scaler.pkl"))  # Should be > 1 KB
```

### ⚠️ sklearn InconsistentVersionWarning

**Cause:** Version mismatch between training and runtime.

**Fix:** This is a **warning**, not an error. The app suppresses it. For best accuracy, match versions:
```bash
pip install scikit-learn==1.6.1
```

---

## 🧪 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | Linear Support Vector Classification (LinearSVC) |
| **Calibration** | CalibratedClassifierCV with sigmoid calibration |
| **Vectorization** | TF-IDF (Term Frequency-Inverse Document Frequency) |
| **Scaling** | MaxAbsScaler (preserves sparsity) |
| **Output** | Binary classification (Positive / Negative) |
| **Confidence** | True probability via `predict_proba()` |

---

## 📝 Example Predictions

| Review | Sentiment | Confidence |
|--------|-----------|------------|
| "A masterpiece of modern cinema. Stunning visuals and iconic score." | **Positive** | ~95% |
| "The acting is wooden, the plot is nonsensical. Complete waste of time." | **Negative** | ~92% |
| "I cry every single time I watch it. The ending is so satisfying." | **Positive** | ~97% |
| "CGI looks like it was made in 2002. The actors don't want to be there." | **Negative** | ~89% |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs via Issues
- Submit Pull Requests for new features
- Suggest UI improvements

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine Learning powered by [scikit-learn](https://scikit-learn.org/)
- UI design inspired by modern glassmorphism trends

---

<p align="center">
  <b>🎬 CineSense</b> — Know the sentiment before you watch.<br>
  <sub>All three .pkl files are loaded and utilized in the prediction pipeline.</sub>
</p>
