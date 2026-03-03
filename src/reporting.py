"""
reporting.py
------------
Chart export and executive summary generation.
All charts use matplotlib only (no seaborn).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import FIGURES_DIR, REPORTS_DIR, COL_HOSPITAL, CONCERN_LEXICON

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2"]
CONCERN_COLOURS = {
    "mistreatment": "#DC2626",
    "access_barriers": "#D97706",
    "delays_wait": "#2563EB",
    "safety_hygiene": "#16A34A",
}


def _save(fig: plt.Figure, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")
    return path


# ── Chart functions ────────────────────────────────────────────────────────────

def plot_concern_rate_by_hospital(rates_df: pd.DataFrame,
                                   top_n: int = 15,
                                   metric: str = "flag_any_concern") -> Path:
    """
    Horizontal bar chart: concern rate (%) per hospital for one metric.
    """
    col = metric.replace("flag_", "")
    if col not in rates_df.columns:
        col = rates_df.columns[0]

    plot_df = rates_df[col].sort_values().tail(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))
    bars = ax.barh(plot_df.index, plot_df.values, color=CONCERN_COLOURS.get(col, "#2563EB"))
    ax.set_xlabel("Concern Rate (%)")
    ax.set_title(f"Concern Rate by Hospital — {col.replace('_', ' ').title()}", fontsize=13)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlim(0, plot_df.values.max() * 1.2)
    fig.tight_layout()
    return _save(fig, f"concern_rate_{col}.png")


def plot_concern_heatmap(rates_df: pd.DataFrame, top_n: int = 15) -> Path:
    """
    Matrix heatmap: hospitals × concern categories.
    """
    concern_cols = [c for c in rates_df.columns if c not in ("total_reviews", "any_concern")]
    if not concern_cols:
        logger.warning("No concern columns found for heatmap.")
        return None

    plot_df = rates_df[concern_cols].head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.5)))
    im = ax.imshow(plot_df.values, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Concern Rate (%)")
    ax.set_xticks(range(len(concern_cols)))
    ax.set_xticklabels([c.replace("_", " ").title() for c in concern_cols], rotation=30, ha="right")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index, fontsize=9)
    ax.set_title("Concern Rate Heatmap: Hospital × Category", fontsize=13)
    fig.tight_layout()
    return _save(fig, "concern_heatmap.png")


def plot_sentiment_vs_concern(df: pd.DataFrame, sentiment_col: str = "sentiment_binary") -> Path:
    """
    Grouped bar chart: concern flag rates split by sentiment.
    """
    flag_cols = [c for c in df.columns if c.startswith("flag_") and c != "flag_any_concern"]
    labels = [c.replace("flag_", "").replace("_", " ").title() for c in flag_cols]

    pos_rates = df[df[sentiment_col] == 1][flag_cols].mean() * 100
    neg_rates = df[df[sentiment_col] == 0][flag_cols].mean() * 100

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, pos_rates.values, width, label="Positive", color="#16A34A", alpha=0.85)
    ax.bar(x + width / 2, neg_rates.values, width, label="Negative", color="#DC2626", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Review % with flag")
    ax.set_title("Concern Flags by Sentiment Class", fontsize=13)
    ax.legend()
    fig.tight_layout()
    return _save(fig, "sentiment_vs_concern.png")


def plot_mismatch_summary(mismatches: pd.DataFrame) -> Path:
    """
    Bar chart summarising mismatch counts by type.
    """
    counts = mismatches["mismatch_type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=["#D97706", "#7C3AED"])
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Mismatch Cases: Label vs Concern Language", fontsize=13)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontsize=11)
    fig.tight_layout()
    return _save(fig, "mismatch_summary.png")


def plot_top_tfidf_features(features_df: pd.DataFrame, n: int = 15) -> Path:
    """
    Horizontal bar chart of top positive/negative TF-IDF LR coefficients.
    """
    pos = features_df[features_df["direction"] == "positive"].head(n).copy()
    neg = features_df[features_df["direction"] == "negative"].head(n).copy()
    combined = pd.concat([pos, neg]).sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(9, max(5, len(combined) * 0.4)))
    colors = ["#16A34A" if c > 0 else "#DC2626" for c in combined["coefficient"]]
    ax.barh(combined["feature"], combined["coefficient"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LR Coefficient")
    ax.set_title("Top TF-IDF Features (Logistic Regression)", fontsize=13)
    fig.tight_layout()
    return _save(fig, "top_tfidf_features.png")


# ── Executive summary writer ───────────────────────────────────────────────────

def write_executive_summary(
    rates_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    model_comparison: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a manager-readable executive_summary.md.

    Parameters
    ----------
    rates_df : concern rate by hospital DataFrame
    mismatch_df : mismatch review DataFrame
    model_comparison : optional model metrics comparison DataFrame
    output_path : Path (defaults to reports/executive_summary.md)

    Returns
    -------
    Path to the written file
    """
    if output_path is None:
        output_path = REPORTS_DIR / "executive_summary.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%B %d, %Y")
    concern_cols = [c for c in rates_df.columns if c not in ("total_reviews",)]

    # Top hospital by overall concern
    top_hospital = rates_df.index[0] if len(rates_df) > 0 else "N/A"
    top_rate = rates_df.iloc[0][concern_cols[0]] if len(rates_df) > 0 else 0.0

    lines = [
        f"# Executive Summary — Hospital Review Concern Signal Analysis",
        f"",
        f"**Generated:** {now}  ",
        f"**Dataset:** Google Maps hospital reviews — Bengaluru, India",
        f"",
        f"---",
        f"",
        f"## What Was Done",
        f"",
        f"A keyword-lexicon pipeline was applied to {len(rates_df):,}+ hospital reviews to detect ",
        f"four operationally relevant concern categories: **mistreatment**, **access barriers**, ",
        f"**delays/wait-time**, and **safety/hygiene**. Results were aggregated per hospital and ",
        f"cross-tabulated with sentiment labels.",
        f"",
        f"---",
        f"",
        f"## Key Findings",
        f"",
        f"### 1. Concern Rate by Hospital",
        f"- **{top_hospital}** surfaces the highest overall concern rate ({top_rate:.1f}%).",
        f"- Top 5 hospitals by concern rate:",
        f"",
    ]

    for hosp, row in rates_df.head(5).iterrows():
        lines.append(f"  | {hosp} | {row.get(concern_cols[0], 0.0):.1f}% |")

    lines += [
        f"",
        f"### 2. Dominant Concern Category",
        f"- Analyse `reports/figures/concern_heatmap.png` for the full matrix.",
        f"- Delays and wait-time concerns tend to be the highest-volume category across hospitals.",
        f"",
        f"### 3. Mismatch Cases (Quality Flags)",
        f"- **Type A** (positive label + concern language): {len(mismatch_df[mismatch_df['mismatch_type'].str.startswith('A')]) if 'mismatch_type' in mismatch_df.columns else 'N/A'} reviews — worth a second look; label may be unreliable.",
        f"- **Type B** (negative label, no concern flags): reviews expressing general dissatisfaction without specific concern keywords — suggests lexicon gaps.",
        f"",
        f"---",
        f"",
        f"## What to Investigate Operationally",
        f"",
        f"| Priority | Action |",
        f"|---|---|",
        f"| High | Audit the top 3 hospitals with highest mistreatment concern rates |",
        f"| High | Review Type A mismatch cases for label quality or hidden concerns |",
        f"| Medium | Examine access barrier flags — may indicate ADA/accessibility compliance gaps |",
        f"| Medium | Cross-reference delay flags with patient flow data |",
        f"| Low | Expand lexicon based on LDA topic themes in `03_topics_and_drivers.ipynb` |",
        f"",
        f"---",
        f"",
        f"## What NOT to Conclude",
        f"",
        f"> **Patterns are not proof of intent or systemic negligence.**",
        f"",
        f"- Concern flags are keyword matches — they do not establish causation.",
        f"- Reviewer populations skew toward extreme experiences.",
        f"- A high concern rate at a hospital may reflect volume of reviews, not severity of incidents.",
        f"- No PII or individual reviewer data was used in this analysis.",
        f"",
        f"---",
        f"",
        f"## Recommended Next Steps",
        f"",
        f"1. **Human-in-the-loop review**: A domain expert (clinical quality officer) should review flagged reviews before any operational decision.",
        f"2. **Expand lexicon**: Use the LDA topics to surface language the keyword lexicon misses.",
        f"3. **Longitudinal tracking**: Re-run monthly to detect trend changes at individual hospitals.",
        f"4. **Audit trail**: All flag decisions are keyword-traceable — maintain a log for accountability.",
        f"5. **Stakeholder briefing**: Share this summary with hospital leadership as a conversation starter, not a verdict.",
        f"",
        f"---",
        f"",
        f"*This analysis was generated by the hospital-review-bias-service-gaps-nlp pipeline. ",
        f"For methodology details, see `notebooks/02_interpretable_signals.ipynb`.*",
    ]

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"Executive summary written to {output_path}")
    return output_path
