"""
generate_poster.py
------------------
Generates a crisp, high-resolution project poster using matplotlib and real data.
Theme: Light / Compact
Output: reports/poster.png  (200 DPI, print-ready)

Run from repo root:
    python generate_poster.py
"""

import sys
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent

# ── Load & prep data ──────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "hospital.csv")
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
df = df.reset_index(drop=True)

CONCERN_LEXICON = {
    "Mistreatment":     ["rude","rudely","ignored","dismissive","disrespectful","insulted","arrogant","negligent","unprofessional"],
    "Access Barriers":  ["wheelchair","ramp","accessible","lift","elevator","disabled","handicap","parking"],
    "Delays / Wait":    ["waiting","wait","waited","hours","queue","delayed","slow","late","forever","long time"],
    "Safety / Hygiene": ["dirty","unclean","unhygienic","hygiene","infection","infested","smell","smelly","contaminated"],
}

CAT_KEYS = ["mistreatment", "access_barriers", "delays_wait", "safety_hygiene"]
CAT_LABELS_SHORT = ["Mistreat-\nment", "Access\nBarriers", "Delays /\nWait", "Safety /\nHygiene"]

for cat_key, (cat, kws) in zip(CAT_KEYS, CONCERN_LEXICON.items()):
    df[f"flag_{cat_key}"] = df["review_text"].apply(lambda x: int(any(kw in str(x).lower() for kw in kws)))

flag_cols = [f"flag_{k}" for k in CAT_KEYS]
df["flag_any"] = (df[flag_cols].sum(axis=1) > 0).astype(int)

# ── Compute stats ─────────────────────────────────────────────────────────────
n_total   = len(df)
n_pos     = (df["sentiment_label"]==1).sum()
n_neg     = (df["sentiment_label"]==0).sum()
n_flagged = df["flag_any"].sum()
type_a    = ((df["sentiment_label"]==1) & (df["flag_any"]==1)).sum()
type_b    = ((df["sentiment_label"]==0) & (df["flag_any"]==0)).sum()

flag_rate_neg = df[df["sentiment_label"]==0][flag_cols].mean()*100
flag_rate_pos = df[df["sentiment_label"]==1][flag_cols].mean()*100
lift = flag_rate_neg - flag_rate_pos

# ── Light Theme Palette ────────────────────────────────────────────────────────
BG        = "#F9FAFB"  # Very light gray
PANEL_BG  = "#FFFFFF"  # Pure white
TEXT      = "#1F2937"  # Dark gray/black
TEXT_SEC  = "#4B5563"  # Medium gray
CYAN      = "#0284C7"  # Deep blue-cyan
ORANGE    = "#EA580C"  # Darker orange
GREEN     = "#16A34A"  # Forest green
RED       = "#DC2626"  # Strong red
PURPLE    = "#7C3AED"  # Darker purple
AMBER     = "#D97706"  # Golden amber
CAT_CLRS  = [RED, ORANGE, CYAN, PURPLE]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "text.color":       TEXT,
    "axes.facecolor":   PANEL_BG,
    "figure.facecolor": BG,
    "axes.edgecolor":   "#D1D5DB",
    "axes.labelcolor":  TEXT_SEC,
    "xtick.color":      TEXT_SEC,
    "ytick.color":      TEXT_SEC,
    "grid.color":       "#E5E7EB",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
})

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22), facecolor=BG)
fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.03, hspace=0.6, wspace=0.35)

gs = gridspec.GridSpec(
    5, 3,
    height_ratios=[0.08, 0.12, 0.25, 0.25, 0.25],
    figure=fig,
)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
ax_title = fig.add_subplot(gs[0, :])
ax_title.set_facecolor(BG)
ax_title.axis("off")

ax_title.text(0.5, 0.9, "Detecting Service Gaps & Bias Signals in Hospital Reviews",
              fontsize=28, fontweight="bold", color=TEXT,
              ha="center", va="top", transform=ax_title.transAxes)
ax_title.text(0.5, 0.4, "Interpretable NLP Pipeline  ·  Hospital Sentiment Reliability Analysis",
              fontsize=14, color=CYAN, ha="center", va="top", transform=ax_title.transAxes)
ax_title.text(0.5, 0.05, "Divya Dhole  |  M.S. Data Science  |  github.com/Divyadhole",
              fontsize=11, color=TEXT_SEC, ha="center", va="top", transform=ax_title.transAxes)
ax_title.axhline(y=-0.1, color=CYAN, linewidth=2, alpha=0.3)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1  ─  KPI Metrics (Compact)
# ═══════════════════════════════════════════════════════════════════════════════
kpi_keys   = ["Total", "Positive", "Negative", "Flagged", "Type A Bias", "Type B Bias"]
kpi_vals   = [f"{n_total:,}", f"{n_pos:,}", f"{n_neg:,}",
              f"{n_flagged:,}", f"{type_a:,}", f"{type_b:,}"]
kpi_colors = [CYAN, GREEN, RED, AMBER, ORANGE, PURPLE]

ax_kpi = fig.add_subplot(gs[1, :])
ax_kpi.set_facecolor(BG)
ax_kpi.axis("off")

for i, (key, val, col) in enumerate(zip(kpi_keys, kpi_vals, kpi_colors)):
    x0 = i / 6 + 0.005
    rect = FancyBboxPatch((x0, 0.05), 0.155, 0.9,
                          boxstyle="round,pad=0.01", linewidth=1.5,
                          edgecolor="#E5E7EB", facecolor=PANEL_BG,
                          transform=ax_kpi.transAxes, zorder=1)
    ax_kpi.add_patch(rect)
    xc = x0 + 0.077
    ax_kpi.text(xc, 0.65, val, fontsize=24, fontweight="bold",
                color=col, ha="center", va="center", transform=ax_kpi.transAxes)
    ax_kpi.text(xc, 0.3, key, fontsize=10, color=TEXT_SEC, fontweight="bold",
                ha="center", va="center", transform=ax_kpi.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2  ─  Technical Lexicon & General Flags
# ═══════════════════════════════════════════════════════════════════════════════

# ── Panel 2.1: technical Lexicon ─────────────────────────────────────────────
ax_lex = fig.add_subplot(gs[2, 0:2])
ax_lex.set_facecolor(PANEL_BG)
ax_lex.axis("off")

ax_lex.text(0.01, 0.95, "TECHNICAL LEXICON & SIGNAL DETECTION", fontsize=12, color=CYAN,
            fontweight="bold", transform=ax_lex.transAxes)
ax_lex.text(0.01, 0.88, "Extraction of high-confidence service signals using categorized keyword filters.",
            fontsize=9, color=TEXT_SEC, transform=ax_lex.transAxes)

y_start = 0.74
for i, (cat, keywords) in enumerate(CONCERN_LEXICON.items()):
    y = y_start - i * 0.17
    ax_lex.text(0.01, y, f" {cat}", fontsize=11, color=CAT_CLRS[i], fontweight="bold", transform=ax_lex.transAxes)
    ax_lex.text(0.24, y, ", ".join(keywords[:6]) + "...", fontsize=9, color=TEXT, transform=ax_lex.transAxes)
    
    eg = {
        "Mistreatment": "\"The staff was incredibly rude...\"",
        "Access Barriers": "\"No wheelchair ramp at entrance.\"",
        "Delays / Wait": "\"We waited for over 3 hours...\"",
        "Safety / Hygiene": "\"The ward was very unhygienic.\"",
    }
    ax_lex.text(0.24, y-0.06, f"Example: {eg.get(cat, '')}", fontsize=8, color=AMBER, style="italic", transform=ax_lex.transAxes)

# ── Panel 2.2: Concern Rate Chart ─────────────────────────────────────────────
ax_a = fig.add_subplot(gs[2, 2])
rates_overall = df[flag_cols].mean() * 100
ax_a.bar(CAT_LABELS_SHORT, rates_overall.values, color=CAT_CLRS, alpha=0.8, edgecolor="#D1D5DB", linewidth=1, width=0.6)
ax_a.set_title("Overall Concern Rates (%)", fontsize=11, color=TEXT, fontweight="bold", pad=8)
ax_a.yaxis.grid(True)
for bar, val in zip(ax_a.patches, rates_overall.values):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3  ─  Bias Metrics & Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# ── Panel 3.1: Sentiment x Concern Comparison ─────────────────────────────────
ax_b = fig.add_subplot(gs[3, 0])
x = np.arange(len(CAT_KEYS))
w = 0.35
ax_b.bar(x - w/2, flag_rate_pos.values, w, label="Pos", color=GREEN, alpha=0.7)
ax_b.bar(x + w/2, flag_rate_neg.values, w, label="Neg", color=RED,   alpha=0.7)
ax_b.set_title("Concern Rate × Sentiment", fontsize=11, fontweight="bold")
ax_b.set_xticks(x)
ax_b.set_xticklabels(CAT_LABELS_SHORT, fontsize=8)
ax_b.legend(fontsize=8, loc="upper right")
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# ── Panel 3.2: Bias Mismatch Definitions ──────────────────────────────────────
ax_bias = fig.add_subplot(gs[3, 1:3])
ax_bias.set_facecolor(PANEL_BG)
ax_bias.axis("off")

ax_bias.text(0.01, 0.95, "SENTIMENT BIAS & LABEL MISMATCH", fontsize=12, color=ORANGE,
            fontweight="bold", transform=ax_bias.transAxes)

ax_bias.text(0.01, 0.78, "Type A: HIDDEN DISSATISFACTION", fontsize=10, color=AMBER, fontweight="bold", transform=ax_bias.transAxes)
ax_bias.text(0.01, 0.70, "Reviews with POSITIVE labels but MAJOR service gap flags. Indicates patients being over-polite despite failures.", fontsize=9, color=TEXT_SEC, transform=ax_bias.transAxes)

ax_bias.text(0.01, 0.48, "Type B: UNEXPLAINED NEGATIVES", fontsize=10, color=PURPLE, fontweight="bold", transform=ax_bias.transAxes)
ax_bias.text(0.01, 0.40, "Reviews with NEGATIVE labels but NO flags triggered. Points to lexicon gaps (e.g., pricing or specific facilities).", fontsize=9, color=TEXT_SEC, transform=ax_bias.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4  ─  Methodology & Consistency
# ═══════════════════════════════════════════════════════════════════════════════

# ── Panel 4.1: Mismatch Pie ───────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[4, 0])
consistent = n_total - type_a - type_b
ax_c.pie([consistent, type_a, type_b], labels=["Ready", "Type A", "Type B"], 
         colors=["#F3F4F6", AMBER, PURPLE], autopct="%1.1f%%", startangle=90, 
         textprops={'fontsize': 8, 'color': TEXT}, wedgeprops={'edgecolor': '#D1D5DB'})
ax_c.set_title("Label Reliability", fontsize=11, fontweight="bold")

# ── Panel 4.2: Methodology ───────────────────────────────────────────────────
ax_proc = fig.add_subplot(gs[4, 1:3])
ax_proc.set_facecolor(PANEL_BG)
ax_proc.axis("off")

ax_proc.text(0.01, 0.95, "METHODOLOGY & FINDINGS", fontsize=12, color=GREEN,
            fontweight="bold", transform=ax_proc.transAxes)

proc_steps = [
    "1. Data: Scraped 977 Bengaluru hospital reviews.",
    "2. Cleaning: Regex normalization and patient text preprocessing.",
    "3. Signals: Boolean flagging based on categorical lexicons.",
    "4. Modeling: Logistic Regression Tone Detection (Acc: 87%, AUC: 0.91).",
    "5. Insight: 6.1% of 'Positive' reviews hide severe operational gaps (Type A)."
]

for i, step in enumerate(proc_steps):
    ax_proc.text(0.01, 0.8 - i*0.13, f" \u2192  {step}", fontsize=9, color=TEXT_SEC, transform=ax_proc.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
fig.text(0.5, 0.02, "Tech: Python/Pandas/Scikit-Learn/Matplotlib/Streamlit  |  github.com/Divyadhole",
         fontsize=10, color=TEXT_SEC, ha="center")

# ── Save ───────────────────────────────────────────────────────────────────────
out = ROOT / "reports" / "poster.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"✅ Light & Compact Poster saved → {out}")
plt.close(fig)
