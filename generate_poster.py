"""
generate_poster.py
------------------
Generates a ULTRA-COMPACT, high-density project poster.
Theme: Light / Compact / No Empty Space
Output: reports/poster.png

Optimized to be high-impact with minimal white space.
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

kw_counts = Counter(all_hit_kws).most_common(8)

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
    "font.size": 10,
})

# ── Figure Layout (ULTRA-COMPACT) ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 20), facecolor=BG)
fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.03, hspace=0.3, wspace=0.25)

gs = gridspec.GridSpec(
    5, 3,
    height_ratios=[0.06, 0.1, 0.28, 0.28, 0.28],
    figure=fig,
)

# HELPER: Rounded Box for groups
def draw_panel(ax, title, color=CYAN):
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    ax.text(0.01, 0.95, title.upper(), fontsize=14, fontweight="bold", color=color, transform=ax.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis("off")
ax_title.text(0.5, 0.7, "Detecting Service Gaps & Bias Signals in Hospital Reviews",
              fontsize=34, fontweight="bold", color=TEXT, ha="center")
ax_title.text(0.5, 0.2, "Interpretable NLP Pipeline  ·  Hospital Sentiment Reliability Analysis",
              fontsize=18, color=CYAN, ha="center")

# ═══════════════════════════════════════════════════════════════════════════════
# KPIs (Row 1)
# ═══════════════════════════════════════════════════════════════════════════════
kpi_keys = ["Total", "Positive", "Negative", "Gap Flagged", "Type A Bias", "Type B Bias"]
kpi_vals = [f"{n_total}", f"{n_pos}", f"{n_neg}", f"{n_flagged}", f"{type_a}", f"{type_b}"]
kpi_cols = [CYAN, GREEN, RED, AMBER, ORANGE, PURPLE]

ax_kpi = fig.add_subplot(gs[1, :])
ax_kpi.axis("off")
for i, (k, v, c) in enumerate(zip(kpi_keys, kpi_vals, kpi_cols)):
    x = i/6 + 0.005
    rect = FancyBboxPatch((x, 0.05), 0.155, 0.9, boxstyle="round,pad=0.02",
                          linewidth=1, edgecolor="#E5E7EB", facecolor=PANEL_BG, transform=ax_kpi.transAxes)
    ax_kpi.add_patch(rect)
    ax_kpi.text(x + 0.077, 0.55, v, fontsize=32, fontweight="bold", color=c, ha="center", transform=ax_kpi.transAxes)
    ax_kpi.text(x + 0.077, 0.25, k.upper(), fontsize=10, color=TEXT_SEC, fontweight="bold", ha="center", transform=ax_kpi.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL & ANALYTIC (Row 2)
# ═══════════════════════════════════════════════════════════════════════════════
# 2.1 Lexicon & Frequency
ax_lex = fig.add_subplot(gs[2, 0:2])
draw_panel(ax_lex, "Technical Lexicon & Frequency", CYAN)
ov_rates = df[flag_cols].mean() * 100
for i, (cat, kws) in enumerate(CONCERN_LEXICON.items()):
    y = 0.75 - i*0.18
    ax_lex.text(0.02, y, f" {cat}", fontsize=13, color=CAT_CLRS[i], fontweight="bold", transform=ax_lex.transAxes)
    ax_lex.text(0.24, y, f"Hits: {ov_rates.iloc[i]:.1f}%", fontsize=11, fontweight="bold", color=TEXT, transform=ax_lex.transAxes)
    ax_lex.text(0.24, y-0.07, ", ".join(kws[:7]) + "...", fontsize=10, color=TEXT_SEC, transform=ax_lex.transAxes)

# 2.2 Top Keywords (Bar)
ax_kw = fig.add_subplot(gs[2, 2])
kw_labels = [k[0] for k in kw_counts]
kw_vals = [k[1] for k in kw_counts]
ax_kw.barh(kw_labels[::-1], kw_vals[::-1], color=ORANGE, alpha=0.8)
ax_kw.set_title("Top Service Gap Keywords", fontsize=12, fontweight="bold", pad=15)
ax_kw.spines["top"].set_visible(False)
ax_kw.spines["right"].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# BIAS & SIGNALS (Row 3)
# ═══════════════════════════════════════════════════════════════════════════════
# 3.1 Label Reliability (Pie)
ax_pi = fig.add_subplot(gs[3, 0])
ax_pi.pie([n_total-type_a-type_b, type_a, type_b], labels=["Reliable", "Type A", "Type B"], 
          colors=["#F3F4F6", AMBER, PURPLE], autopct='%1.1f%%', wedgeprops=dict(edgecolor="#D1D5DB"))
ax_pi.set_title("Label Reliability Analysis", fontsize=12, fontweight="bold")

# 3.2 Signal vs Tone (Bar)
ax_tone = fig.add_subplot(gs[3, 1:3])
xw = np.arange(len(CAT_KEYS))
ww = 0.35
ax_tone.bar(xw - ww/2, flag_rate_pos.values, ww, label="Positive Sentiment", color=GREEN, alpha=0.7)
ax_tone.bar(xw + ww/2, flag_rate_neg.values, ww, label="Negative Sentiment", color=RED, alpha=0.7)
ax_tone.set_title("Gap Signal Presence by Tone (%)", fontsize=12, fontweight="bold")
ax_tone.set_xticks(xw)
ax_tone.set_xticklabels(CAT_LABELS_SHORT, fontsize=9)
ax_tone.legend(fontsize=9, loc="upper right")
ax_tone.spines["top"].set_visible(False)
ax_tone.spines["right"].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY & ACTIONS (Row 4)
# ═══════════════════════════════════════════════════════════════════════════════
# 4.1 Methodology
ax_meth = fig.add_subplot(gs[4, 0:1])
draw_panel(ax_meth, "Technical Pipeline", GREEN)
steps = [
    "1. Data: 977 Patient Reviews.",
    "2. Cleaning: RegEx Normalization.",
    "3. Signals: Boolean Lexicon Flags.",
    "4. Modeling: Tone Det. (91% AUC).",
    "5. Calibration: Bias Mismatch ID."
]
for i, s in enumerate(steps):
    ax_meth.text(0.02, 0.78 - i*0.15, f"\u25CF {s}", fontsize=11, transform=ax_meth.transAxes)

# 4.2 Bias Definitions (Compact)
ax_def = fig.add_subplot(gs[4, 1:2])
draw_panel(ax_def, "Bias Definitions", ORANGE)
ax_def.text(0.02, 0.78, "Type A: HIDDEN DISSATISFACTION", fontsize=11, fontweight="bold", color=AMBER, transform=ax_def.transAxes)
ax_def.text(0.02, 0.68, "Positive tone with negative gaps.\nIndicates patients being over-polite.", fontsize=10, transform=ax_def.transAxes)
ax_def.text(0.02, 0.45, "Type B: UNEXPLAINED NEGATIVES", fontsize=11, fontweight="bold", color=PURPLE, transform=ax_def.transAxes)
ax_def.text(0.02, 0.35, "Negative tone with NO keywords.\nPoints to pricing or facility gaps.", fontsize=10, transform=ax_def.transAxes)

# 4.3 Actionable Insights
ax_act = fig.add_subplot(gs[4, 2:3])
draw_panel(ax_act, "Actionable Insights", RED)
recos = [
    "Mitigate 'Wait Time': #1 Operational concern.",
    "Audit 'Type A' cases for service failure.",
    "Expand Lexicon to capture Type B root causes.",
    "Prioritize accessibility in facilities."
]
for i, r in enumerate(recos):
    ax_act.text(0.02, 0.78 - i*0.15, f"\u2192 {r}", fontsize=11, transform=ax_act.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
fig.text(0.5, 0.015, "Full Analysis & Code: github.com/Divyadhole  |  Built with Python, Scikit-Learn, Matplotlib, Streamlit", 
         fontsize=12, color="#6B7280", ha="center")

# ── Save ──────────────────────────────────────────────────────────────────────
out = ROOT / "reports" / "poster.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"✅ ULTRA-COMPACT High-Density Poster saved → {out}")
plt.close(fig)
