"""
streamlit_app.py
-----------------
Lightweight interactive demo for the Hospital Review Bias / Service Gaps project.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Review Insights",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lexicon ────────────────────────────────────────────────────────────────────
CONCERN_LEXICON = {
    "mistreatment": [
        "rude", "rudely", "ignored", "dismissive", "disrespectful",
        "insulted", "humiliated", "arrogant", "negligent",
        "unprofessional", "attitude", "refused",
    ],
    "access_barriers": [
        "wheelchair", "ramp", "accessible", "accessibility", "lift",
        "elevator", "stairs", "disabled", "handicap", "parking", "mobility",
    ],
    "delays_wait": [
        "waiting", "wait", "waited", "hours", "queue",
        "delayed", "delay", "slow", "late", "forever", "long time",
    ],
    "safety_hygiene": [
        "dirty", "unclean", "unhygienic", "hygiene", "infection",
        "infested", "smell", "smelly", "contaminated", "unsanitary",
    ],
}

CATEGORY_LABELS = {
    "mistreatment": "👊 Mistreatment",
    "access_barriers": "♿ Access Barriers",
    "delays_wait": "⏱ Delays / Wait",
    "safety_hygiene": "🧹 Safety / Hygiene",
}

CATEGORY_COLORS = {
    "mistreatment": "#e74c3c",
    "access_barriers": "#f39c12",
    "delays_wait": "#3498db",
    "safety_hygiene": "#8e44ad",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def flag_concern(text: str, keywords: list) -> int:
    t = str(text).lower()
    return int(any(kw in t for kw in keywords))


def get_matched(text: str, keywords: list) -> list:
    t = str(text).lower()
    return [kw for kw in keywords if kw in t]


def apply_flags(df: pd.DataFrame) -> pd.DataFrame:
    for cat, kws in CONCERN_LEXICON.items():
        df[f"flag_{cat}"] = df["review_text"].apply(lambda x: flag_concern(x, kws))
    flag_cols = [f"flag_{c}" for c in CONCERN_LEXICON]
    df["flag_any_concern"] = (df[flag_cols].sum(axis=1) > 0).astype(int)
    return df


@st.cache_data
def load_data():
    # Try processed first, then fallback to raw
    processed = ROOT / "data" / "processed" / "reviews_flagged.csv"
    raw = ROOT / "data" / "hospital.csv"

    if processed.exists():
        df = pd.read_csv(processed)
        if "review_text" not in df.columns:
            df = None
    else:
        df = None

    if df is None and raw.exists():
        df = pd.read_csv(raw)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            "Feedback":        "review_text",
            "Sentiment Label": "sentiment_label",
            "Ratings":         "rating",
        })
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df.drop_duplicates().dropna(subset=["review_text"])
        df["sentiment_label"] = pd.to_numeric(df["sentiment_label"], errors="coerce").fillna(0).astype(int)
        df["sentiment"] = df["sentiment_label"].map({1: "positive", 0: "negative"})
        df = apply_flags(df)
        df = df.reset_index(drop=True)

    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=60)
st.sidebar.title("Hospital Review Insights")
st.sidebar.markdown("*Interpretable NLP — Service Gaps & Bias Signals*")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔬 Signal Explorer", "🔍 Live Review Analyser", "📋 Mismatch Cases"],
)

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_data()

if df is None:
    st.error(
        "No dataset found. Please place `hospital.csv` in `data/` or run "
        "`01_eda_overview.ipynb` first to generate the processed file."
    )
    st.stop()

flag_cols = [c for c in df.columns if c.startswith("flag_") and c != "flag_any_concern"]
categories = [c.replace("flag_", "") for c in flag_cols]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Dataset Overview")

    # KPI row
    n_total  = len(df)
    n_pos    = (df["sentiment_label"] == 1).sum()
    n_neg    = (df["sentiment_label"] == 0).sum()
    n_flagged = df["flag_any_concern"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{n_total:,}")
    col2.metric("Positive", f"{n_pos:,}", f"{n_pos/n_total*100:.1f}%")
    col3.metric("Negative", f"{n_neg:,}", f"-{n_neg/n_total*100:.1f}%", delta_color="inverse")
    col4.metric("With ≥1 Concern Flag", f"{n_flagged:,}", f"{n_flagged/n_total*100:.1f}%")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        sent_counts = df["sentiment"].value_counts() if "sentiment" in df.columns else pd.Series({"positive": n_pos, "negative": n_neg})
        colors = ["#2ecc71", "#e74c3c"]
        ax.bar(sent_counts.index, sent_counts.values, color=colors, edgecolor="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col_b:
        st.subheader("Concern Flag Rates")
        rates = df[flag_cols].mean() * 100
        rates.index = [CATEGORY_LABELS.get(c.replace("flag_",""), c) for c in rates.index]
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.barh(rates.index, rates.values, color=[CATEGORY_COLORS[c] for c in categories], edgecolor="white")
        ax2.set_xlabel("Flagged Reviews (%)")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Signal Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Signal Explorer":
    st.title("🔬 Concern Signal Explorer")
    st.markdown("Concern rate by category, broken down by sentiment label.")

    concern_by_sent = df.groupby("sentiment")[flag_cols].mean() * 100 if "sentiment" in df.columns else None

    if concern_by_sent is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(categories))
        width = 0.35
        pos_values = concern_by_sent.loc["positive"].values if "positive" in concern_by_sent.index else [0]*len(categories)
        neg_values = concern_by_sent.loc["negative"].values if "negative" in concern_by_sent.index else [0]*len(categories)
        ax.bar(x - width/2, pos_values, width, label="Positive Reviews", color="#2ecc71", edgecolor="white")
        ax.bar(x + width/2, neg_values, width, label="Negative Reviews", color="#e74c3c", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories], rotation=10)
        ax.set_ylabel("Concern Rate (%)")
        ax.set_title("Concern Rate by Category × Sentiment", fontweight="bold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)

    st.divider()
    st.subheader("Sample Flagged Reviews")
    selected_cat = st.selectbox("Filter by concern category", ["Any"] + [CATEGORY_LABELS[c] for c in categories])
    n_samples = st.slider("Number of samples", 3, 20, 5)

    if selected_cat == "Any":
        sample_df = df[df["flag_any_concern"] == 1]
    else:
        raw_cat = [c for c in categories if CATEGORY_LABELS[c] == selected_cat][0]
        sample_df = df[df[f"flag_{raw_cat}"] == 1]

    for _, row in sample_df.sample(min(n_samples, len(sample_df)), random_state=42).iterrows():
        sentiment_emoji = "✅" if row.get("sentiment_label", 0) == 1 else "❌"
        st.markdown(f"**{sentiment_emoji} Sentiment:** {row.get('sentiment','unknown')}")
        st.write(str(row["review_text"])[:400])
        matched_info = []
        for cat, kws in CONCERN_LEXICON.items():
            matched = get_matched(row["review_text"], kws)
            if matched:
                matched_info.append(f"**{CATEGORY_LABELS[cat]}**: `{', '.join(matched)}`")
        if matched_info:
            st.markdown("Triggered by: " + " | ".join(matched_info))
        st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Live Review Analyser
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Review Analyser":
    st.title("🔍 Live Review Analyser")
    st.markdown("Paste any hospital review to see which concern signals it triggers.")

    sample_reviews = [
        "The staff was incredibly rude and we waited for over 3 hours without any update.",
        "Great doctors and clean facilities. Highly recommend.",
        "The hospital is dirty and the nurses ignored our requests for help.",
        "Accessible entrance with ramps, very accommodating for my wheelchair.",
    ]

    selected_sample = st.selectbox("Or pick a sample review:", ["(type your own)"] + sample_reviews)
    default_text = "" if selected_sample == "(type your own)" else selected_sample
    review_input = st.text_area("Review text:", value=default_text, height=120)

    if review_input.strip():
        st.subheader("Analysis Results")
        any_flag = False
        for cat, kws in CONCERN_LEXICON.items():
            matched = get_matched(review_input, kws)
            if matched:
                any_flag = True
                color = CATEGORY_COLORS[cat]
                st.markdown(
                    f"<div style='background:{color}22;border-left:4px solid {color};"
                    f"padding:8px;border-radius:4px;margin:4px 0'>"
                    f"<b>{CATEGORY_LABELS[cat]}</b> — triggered by: "
                    f"<code>{', '.join(matched)}</code></div>",
                    unsafe_allow_html=True,
                )

        if not any_flag:
            st.success("✅ No concern signals detected in this review.")
        else:
            st.warning("⚠️ Concern signals detected — recommended for human review.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Mismatch Cases
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Mismatch Cases":
    st.title("📋 Mismatch Analysis")
    st.markdown(
        "**Type A**: Positive sentiment label + ≥1 concern flag → potential hidden dissatisfaction  \n"
        "**Type B**: Negative sentiment label + 0 concern flags → negative experience outside our lexicon"
    )

    type_a = df[(df["sentiment_label"] == 1) & (df["flag_any_concern"] == 1)]
    type_b = df[(df["sentiment_label"] == 0) & (df["flag_any_concern"] == 0)]

    col1, col2 = st.columns(2)
    col1.metric("Type A (positive + flags)", len(type_a), f"{len(type_a)/len(df)*100:.1f}% of reviews")
    col2.metric("Type B (negative + no flags)", len(type_b), f"{len(type_b)/len(df)*100:.1f}% of reviews")

    mismatch_type = st.radio("Show:", ["Type A — Hidden Dissatisfaction", "Type B — Unexplained Negatives"])
    n_show = st.slider("Reviews to show", 3, 15, 5)

    sample = type_a if "Type A" in mismatch_type else type_b
    for _, row in sample.sample(min(n_show, len(sample)), random_state=10).iterrows():
        st.write(f"**Review:** {str(row['review_text'])[:350]}")
        if "Type A" in mismatch_type:
            for cat, kws in CONCERN_LEXICON.items():
                matched = get_matched(row["review_text"], kws)
                if matched:
                    st.caption(f"↳ {CATEGORY_LABELS[cat]}: {matched}")
        st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("⚠️ Patterns ≠ proof of intent. For investigative purposes only. Human review required before operational decisions.")
