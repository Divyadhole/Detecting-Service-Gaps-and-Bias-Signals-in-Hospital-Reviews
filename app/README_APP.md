# App — Streamlit Demo

## How to Run

```bash
# From the repo root
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Pages

| Page | What it shows |
|------|---------------|
| 📊 Overview | Dataset KPIs, sentiment distribution, flag rates |
| 🔬 Signal Explorer | Concern rates by sentiment + sample flagged reviews |
| 🔍 Live Review Analyser | Paste any text and see which flags trigger (interview demo) |
| 📋 Mismatch Cases | Type A (positive + flags) and Type B (negative + no flags) |

## Requirements

Streamlit is included in `requirements.txt`. No extra configuration needed.
The app reads from `data/processed/reviews_flagged.csv` if it exists, otherwise falls back to `data/hospital.csv`.
