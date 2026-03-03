<div align="center">

# 🏥 Detecting Service Gaps & Bias Signals<br>in Hospital Reviews

**Interpretable NLP · Google Maps Reviews · Bengaluru, India**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit App](https://img.shields.io/badge/🖥%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](http://localhost:8501)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Interview--Ready-brightgreen?style=for-the-badge)]()

<br>

> *"Sentiment scores tell you **that** patients are unhappy — not **why**, not **where**, and not **what** can be fixed."*

<br>

</div>

---

## 🎯 Problem

Hospital star ratings and binary sentiment labels are blunt instruments. A 3-star review hides whether the complaint was about **staff rudeness**, **a 4-hour wait**, **accessibility barriers**, or **a hygiene issue** — each of which demands a completely different operational response.

This project builds an **interpretable NLP pipeline** that:

- 🔍 Flags **specific concern types** in patient language (fully traceable to exact words)
- 📊 Identifies **which hospitals and concern categories** cluster negative experiences
- ⚠️ Surfaces **mismatch cases** — reviews where the sentiment label contradicts the language — highest value for quality assurance
- 🤖 Validates signals with a **TF-IDF + Logistic Regression baseline**

---

## 📊 Dataset

**Google Maps Hospital Reviews — Bengaluru, India**

| Field | Description |
|-------|-------------|
| `Feedback` | Raw review text (English, mixed) |
| `Sentiment Label` | `1` = positive · `0` = negative |
| `Ratings` | Star rating 1–5 (where available) |

> ⚠️ **No raw data is committed to this repository.** See [`data/README_DATA.md`](data/README_DATA.md) for dataset source and download instructions.

---

## 🔬 Methodology

### Step 1 — Text Preprocessing
Lowercase normalisation, whitespace cleanup, tokenisation

### Step 2 — Interpretable Concern Signals

Every review is scanned against **4 hand-curated lexicon categories**. Each flag is **100% traceable** — you can always point to the specific word that triggered it.

```
Review: "We waited 3 hours with no update. The staff was rude and ignored us."
         └── ⏱ delays_wait  : ['wait', 'waited', 'hours']
         └── 👊 mistreatment : ['rude', 'ignored']
```

| 🏷️ Category | 🔑 Example Trigger Words |
|-------------|--------------------------|
| 👊 **Mistreatment** | `rude`, `ignored`, `dismissive`, `arrogant`, `negligent`, `unprofessional` |
| ♿ **Access Barriers** | `wheelchair`, `ramp`, `accessible`, `lift`, `disabled`, `handicap` |
| ⏱ **Delays / Wait** | `waiting`, `queue`, `hours`, `delayed`, `slow`, `forever` |
| 🧹 **Safety / Hygiene** | `dirty`, `unhygienic`, `infested`, `contaminated`, `smell`, `infection` |

### Step 3 — Pattern Analysis

```
MISMATCH TYPE A  →  Positive label  +  concern flags  =  Hidden dissatisfaction
MISMATCH TYPE B  →  Negative label  +  no flags       =  Experience outside lexicon
```

### Step 4 — LDA Topic Modelling
Latent Dirichlet Allocation on negative reviews — surfaces complaint themes beyond the fixed lexicon.

### Step 5 — Baseline ML Model
TF-IDF + Logistic Regression for sentiment prediction. Compared against the lexicon-only heuristic to quantify interpretability trade-offs.

---

## 📈 Key Results (on 977 real reviews)

<div align="center">

| Metric | Value |
|--------|-------|
| 🟢 Positive Reviews | 712 (72.9%) |
| 🔴 Negative Reviews | 265 (27.1%) |
| 🚩 Reviews with ≥1 Concern Flag | 137 (14.0%) |
| ⚠️ Type A Mismatches (hidden dissatisfaction) | 60 (6.1%) |
| ❓ Type B Mismatches (unexplained negatives) | 188 (19.2%) |
| 📉 Delays/Wait — Highest concern flag rate | ~8% of all reviews |
| 🤖 Model Accuracy (TF-IDF + LR) | 87% |
| 📐 ROC-AUC | 0.91 |

</div>

---

## 📁 Project Structure

```
hospital-review-bias-service-gaps-nlp/
│
├── 📓 notebooks/
│   ├── 01_eda_overview.ipynb              ← data quality, distributions, top words
│   ├── 02_interpretable_signals.ipynb     ← lexicon flags, explainability demo, mismatch
│   ├── 03_topics_and_drivers.ipynb        ← LDA topics, concern lift, co-occurrence
│   └── 04_model_baselines.ipynb           ← TF-IDF + LR, ROC, feature importance, errors
│
├── 🧠 src/
│   ├── config.py       ← paths + lexicon config
│   ├── data_io.py      ← load & validate data
│   ├── preprocess.py   ← clean & tokenise text
│   ├── signals.py      ← concern flag logic
│   ├── modeling.py     ← TF-IDF + LR baseline
│   ├── evaluation.py   ← metrics + error analysis
│   └── reporting.py    ← chart export + summary writer
│
├── 📊 reports/
│   ├── figures/                           ← all exported charts
│   └── executive_summary.md              ← manager-facing findings
│
├── 📦 data/
│   ├── raw/                               ← put dataset here locally
│   ├── processed/                         ← generated by notebooks
│   └── README_DATA.md
│
└── 🖥️ app/
    └── streamlit_app.py                   ← interactive 4-page demo
```

---

## 🖥️ Streamlit App — Live Demo

> **▶️ [Launch the App on Streamlit Cloud](https://detecting-service-gaps-and-bias-signals-in-tbtgkafcnjwzsgra2rh.streamlit.app)** ← *live, no install needed!*

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://detecting-service-gaps-and-bias-signals-in-tbtgkafcnjwzsgra2rh.streamlit.app)

The app has **4 interactive pages**:

| Page | Description |
|------|-------------|
| 📊 **Overview** | Dataset KPIs, sentiment distribution, concern flag rates by category |
| 🔬 **Signal Explorer** | Grouped concern rates by sentiment + filterable flagged review samples |
| 🔍 **Live Review Analyser** | Paste any review → instant concern flag detection with keyword highlights |
| 📋 **Mismatch Cases** | Browse Type A (hidden dissatisfaction) and Type B (unexplained negatives) |

**To run locally:**
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```


---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Divyadhole/Detecting-Service-Gaps-and-Bias-Signals-in-Hospital-Reviews.git
cd Detecting-Service-Gaps-and-Bias-Signals-in-Hospital-Reviews

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset
#    Copy CSV to data/ — columns expected: Feedback, Sentiment Label, Ratings

# 4. Run notebooks in order
jupyter lab
# Open notebooks/01_eda_overview.ipynb → run all → proceed through 02, 03, 04

# 5. Launch Streamlit demo
streamlit run app/streamlit_app.py
```

---

## 🎤 Interview Demo Flow

> *Open `02_interpretable_signals.ipynb`*

1. Show the **lexicon categories** — four categories, real keywords, zero black box
2. Run the **explainability cell** — point to exact trigger words per review
3. Show the **concern rate × sentiment** chart — which types drive negative labels
4. Show a **Type A mismatch** — positive-labelled review with concern flags
5. Close with the **executive summary** — operational recommendations

*Total time: ~5 minutes. No ML degree required to follow.*

---

## ⚖️ Ethics & Limitations

> **Patterns ≠ Proof of Intent.** This tool surfaces signals for investigation — not for automated decisions.

| Limitation | Why It Matters |
|------------|----------------|
| 🔵 **Reviewer bias** | Online reviews skew toward extreme experiences; the silent majority is invisible |
| 🟡 **Keyword limitations** | Sarcasm, code-switching (Kannada/Hindi), and negations are missed |
| 🟠 **Correlation ≠ causation** | Concern clusters do not establish institutional negligence |
| 🔴 **Small samples** | Hospitals with few reviews have volatile flag rates |

**Recommended use**: All outputs should initiate an **audit trail with human review** — not replace clinical or operational judgment.

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

---

<div align="center">

**Divya Dhole** · M.S. Data Science  
[GitHub](https://github.com/Divyadhole) · [Portfolio](https://github.com/Divyadhole)

<br>

*Built for interpretability, reproducibility, and responsible deployment.*

</div>
