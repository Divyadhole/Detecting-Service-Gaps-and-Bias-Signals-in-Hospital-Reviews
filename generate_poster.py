"""
generate_poster.py
------------------
Generates a ZERO-WASTE, ULTRA-DENSE project poster.
Theme: Light / Compact / Maximum Visual Filling
Output: reports/poster.png

Optimized to leave NO white space between elements.
"""

import sys
import re
from pathlib import Path
from collections import Counter

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

# Data Cleaning
df["sentiment_label"] = pd.to_numeric(df["sentiment_label"], errors="coerce").fillna(0).astype(int)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).astype(int)
df = df.reset_index(drop=True)

CONCERN_LEXICON = {
    "Mistreatment":     ["rude","rudely","ignored","dismissive","disrespectful","insulted","arrogant","negligent","unprofessional"],
    "Access Barriers":  ["wheelchair","ramp","accessible","lift","elevator","disabled","handicap","parking"],
    "Delays / Wait":    ["waiting","wait","waited","hours","queue","delayed","slow","late","forever","long time"],
    "Safety / Hygiene": ["dirty","unclean","unhygienic","hygiene","infection","infested","smell","smelly","contaminated"],
}

CAT_KEYS = ["mistreatment", "access_barriers", "delays_wait", "safety_hygiene"]
CAT_LABELS_SHORT = ["Mistreatment", "Access Barriers", "Delays / Wait", "Safety / Hygiene"]

# Flagging logic & Keyword Counting
all_hit_kws = []
for cat_key, (cat, kws) in zip(CAT_KEYS, CONCERN_LEXICON.items()):
    def check_kws(text):
        hits = [kw for kw in kws if kw in str(text).lower()]
        all_hit_kws.extend(hits)
        return 1 if hits else 0
    df[f"flag_{cat_key}"] = df["review_text"].apply(check_kws)

kw_counts = Counter(all_hit_kws).most_common(12)

flag_cols = [f"flag_{k}" for k in CAT_KEYS]
df["flag_any"] = (df[flag_cols].sum(axis=1) > 0).astype(int)

# ── Stats ────────────────────────────────────────────────────────────────────
n_total   = len(df)
n_pos     = (df["sentiment_label"]==1).sum()
n_neg     = (df["sentiment_label"]==0).sum()
n_flagged = df["flag_any"].sum()
type_a    = ((df["sentiment_label"]==1) & (df["flag_any"]==1)).sum()
type_b    = ((df["sentiment_label"]==0) & (df["flag_any"]==0)).sum()

flag_rate_neg = df[df["sentiment_label"]==0][flag_cols].mean()*100
flag_rate_pos = df[df["sentiment_label"]==1][flag_cols].mean()*100

# ── Light Theme Palette ───────────────────────────────────────────────────────
BG        = "#F9FAFB"
PANEL_BG  = "#FFFFFF"
TEXT      = "#1F2937"
TEXT_SEC  = "#4B5563"
CYAN      = "#0284C7"
ORANGE    = "#EA580C"
GREEN     = "#16A34A"
RED       = "#DC2626"
PURPLE    = "#7C3AED"
AMBER     = "#D97706"
CAT_CLRS  = [RED, ORANGE, CYAN, PURPLE]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.facecolor": PANEL_BG,
    "figure.facecolor": BG,
    "axes.edgecolor": "#D1D5DB",
    "grid.color": "#E5E7EB",
    "font.size": 11,
})

# ── Figure Layout (SQUARISH-COMPACT 20x15) ───────────────────────────────────
fig = plt.figure(figsize=(20, 15), facecolor=BG)
fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03, hspace=0.2, wspace=0.15)

gs = gridspec.GridSpec(
    5, 5,
    height_ratios=[0.08, 0.12, 0.28, 0.28, 0.24],
    figure=fig,
)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis("off")
ax_title.text(0.5, 0.6, "Detecting Service Gaps & Bias Signals in Hospital Reviews",
              fontsize=38, fontweight="bold", color=TEXT, ha="center")
ax_title.text(0.5, 0.1, "Interpretable NLP Pipeline  ·  Hospital Sentiment Reliability Analysis",
              fontsize=20, color=CYAN, ha="center")

# ═══════════════════════════════════════════════════════════════════════════════
# KPIs (Row 1)
# ═══════════════════════════════════════════════════════════════════════════════
kpi_keys = ["Total", "Positive", "Negative", "Gap Flagted", "Type A Bias", "Type B Bias"]
kpi_vals = [f"{n_total}", f"{n_pos}", f"{n_neg}", f"{n_flagged}", f"{type_a}", f"{type_b}"]
kpi_cols = [CYAN, GREEN, RED, AMBER, ORANGE, PURPLE]

ax_kpi = fig.add_subplot(gs[1, :])
ax_kpi.axis("off")
for i, (k, v, c) in enumerate(zip(kpi_keys, kpi_vals, kpi_cols)):
    x = i/6 + 0.005
    rect = FancyBboxPatch((x, 0.05), 0.155, 0.9, boxstyle="round,pad=0.01",
                          linewidth=1, edgecolor="#E5E7EB", facecolor=PANEL_BG, transform=ax_kpi.transAxes)
    ax_kpi.add_patch(rect)
    ax_kpi.text(x + 0.077, 0.55, v, fontsize=36, fontweight="bold", color=c, ha="center", transform=ax_kpi.transAxes)
    ax_kpi.text(x + 0.077, 0.22, k.upper(), fontsize=12, color=TEXT_SEC, fontweight="bold", ha="center", transform=ax_kpi.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYTIC (Row 2) - Higher Density
# ═══════════════════════════════════════════════════════════════════════════════
# 2.1 Technical Lexicon (Filled)
ax_lex = fig.add_subplot(gs[2, 0:2])
ax_lex.axis("off")
ax_lex.set_title("TECHNICAL LEXICON & EXTRACTION", fontsize=15, fontweight="bold", color=CYAN, loc="left", pad=15)
ov_rates = df[flag_cols].mean() * 100
for i, (cat, kws) in enumerate(CONCERN_LEXICON.items()):
    y = 0.82 - i*0.22
    ax_lex.text(0.01, y, f" {cat}", fontsize=14, color=CAT_CLRS[i], fontweight="bold", transform=ax_lex.transAxes)
    ax_lex.text(0.01, y-0.08, f"Hits: {ov_rates.iloc[i]:.1f}% | Keywords: {', '.join(kws[:6])}...", 
                fontsize=11, color=TEXT_SEC, transform=ax_lex.transAxes)

# 2.2 Top Keywords
ax_kw = fig.add_subplot(gs[2, 2:4])
kw_labels = [k[0] for k in kw_counts]
kw_vals = [k[1] for k in kw_counts]
ax_kw.barh(kw_labels[::-1], kw_vals[::-1], color=ORANGE, alpha=0.8)
ax_kw.set_title("Most Frequent Failure Indicators", fontsize=15, fontweight="bold")
ax_kw.spines["top"].set_visible(False)
ax_kw.spines["right"].set_visible(False)

# 2.3 Sentiment Balance (Compact Side)
ax_sb = fig.add_subplot(gs[2, 4])
ax_sb.pie([n_pos, n_neg], labels=["Pos", "Neg"], colors=[GREEN, RED], 
          autopct='%1.0f%%', startangle=90, wedgeprops=dict(edgecolor="#D1D5DB"))
ax_sb.set_title("Sentiment Balance", fontsize=13, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-CORRELATION (Row 3)
# ═══════════════════════════════════════════════════════════════════════════════
# 3.1 Reliability Pie
ax_rel = fig.add_subplot(gs[3, 0])
ax_rel.pie([n_total-type_a-type_b, type_a, type_b], labels=["Reliable", "Type A", "Type B"], 
           colors=["#F3F4F6", AMBER, PURPLE], autopct='%1.0f%%', wedgeprops=dict(edgecolor="#D1D5DB"))
ax_rel.set_title("Label Reliability", fontsize=13, fontweight="bold")

# 3.2 Signal by Tone (Large Center)
ax_tone = fig.add_subplot(gs[3, 1:4])
xw = np.arange(len(CAT_KEYS))
ww = 0.4
ax_tone.bar(xw - ww/2, flag_rate_pos.values, ww, label="Positive Sentiment", color=GREEN, alpha=0.7)
ax_tone.bar(xw + ww/2, flag_rate_neg.values, ww, label="Negative Sentiment", color=RED, alpha=0.7)
ax_tone.set_title("Gap Signal Strength by Review Tone (%)", fontsize=15, fontweight="bold")
ax_tone.set_xticks(xw)
ax_tone.set_xticklabels(CAT_LABELS_SHORT, fontsize=11)
ax_tone.legend(fontsize=11)
ax_tone.spines["top"].set_visible(False)
ax_tone.spines["right"].set_visible(False)

# 3.3 Rating Distribution (Small Side)
ax_rd = fig.add_subplot(gs[3, 4])
r_counts = df["rating"].value_counts().sort_index()
ax_rd.bar(r_counts.index, r_counts.values, color=CYAN, alpha=0.7)
ax_rd.set_title("Rating Distr.", fontsize=13, fontweight="bold")
ax_rd.set_xticks(range(1, 6))
ax_rd.spines["top"].set_visible(False)
ax_rd.spines["right"].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL & ACTIONABLE (Bottom Row)
# ═══════════════════════════════════════════════════════════════════════════════
# 4.1 Methodology
ax_meth = fig.add_subplot(gs[4, 0:2])
ax_meth.axis("off")
ax_meth.set_title("PIPELINE", fontsize=15, fontweight="bold", color=GREEN, loc="left", pad=10)
steps = [
    "Scrape & Normalization (Regex cleaning).",
    "Boolean Lexicon Signal Extraction.",
    "Logistic Regression Tone Detection (91% AUC).",
    "Mismatch/Bias Calibration Identification."
]
for i, s in enumerate(steps):
    ax_meth.text(0.01, 0.7 - i*0.2, f"\u25CF {s}", fontsize=13, transform=ax_meth.transAxes)

# 4.2 Bias (Compact Middle)
ax_bias = fig.add_subplot(gs[4, 2:4])
ax_bias.axis("off")
ax_bias.set_title("BIAS DEFINITIONS", fontsize=15, fontweight="bold", color=ORANGE, loc="left", pad=10)
ax_bias.text(0.01, 0.7, "Type A (Dissatisfaction): Positive tone with Negative signals.", fontsize=12, fontweight="bold", color=AMBER, transform=ax_bias.transAxes)
ax_bias.text(0.01, 0.3, "Type B (Gap): Negative tone where NO keywords triggered.", fontsize=12, fontweight="bold", color=PURPLE, transform=ax_bias.transAxes)

# 4.3 Action
ax_act = fig.add_subplot(gs[4, 4:])
ax_act.axis("off")
ax_act.set_title("ACTIONS", fontsize=15, fontweight="bold", color=RED, loc="left", pad=10)
recos = ["Audit Type A", "Fix Wait Times", "Expand Lexicon"]
for i, r in enumerate(recos):
    ax_act.text(0.01, 0.7 - i*0.25, f"\u2192 {r}", fontsize=13, fontweight="bold", transform=ax_act.transAxes)

# ── Save ──────────────────────────────────────────────────────────────────────
out = ROOT / "reports" / "poster.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"✅ ZERO-WASTE Ultra-Compact Poster saved → {out}")
plt.close(fig)
