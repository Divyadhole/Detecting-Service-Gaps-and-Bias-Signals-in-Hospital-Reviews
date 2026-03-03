"""
generate_poster.py
------------------
Generates a crisp, high-resolution project poster using matplotlib and real data.
Output: reports/poster.png  (300 DPI, print-ready)

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

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#0B1929"
PANEL_BG  = "#122035"
WHITE     = "#FFFFFF"
LIGHT     = "#B0C4DE"
CYAN      = "#00D4FF"
ORANGE    = "#FF8C42"
GREEN     = "#2ECC71"
RED       = "#E74C3C"
PURPLE    = "#9B59B6"
AMBER     = "#F1C40F"
CAT_CLRS  = [RED, ORANGE, "#3498DB", PURPLE]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "text.color":       WHITE,
    "axes.facecolor":   PANEL_BG,
    "figure.facecolor": BG,
    "axes.edgecolor":   "#1E3A5F",
    "axes.labelcolor":  LIGHT,
    "xtick.color":      LIGHT,
    "ytick.color":      LIGHT,
    "grid.color":       "#1E3A5F",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
})

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 28), facecolor=BG)
fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.02, hspace=0.55, wspace=0.35)

gs = gridspec.GridSpec(
    4, 3,
    height_ratios=[0.12, 0.28, 0.28, 0.28],
    figure=fig,
)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
ax_title = fig.add_subplot(gs[0, :])
ax_title.set_facecolor(BG)
ax_title.axis("off")

ax_title.text(0.5, 0.85, "Detecting Service Gaps & Bias Signals in Hospital Reviews",
              fontsize=28, fontweight="bold", color=WHITE,
              ha="center", va="top", transform=ax_title.transAxes)
ax_title.text(0.5, 0.45, "Interpretable NLP Pipeline  ·  Google Maps Reviews  ·  Bengaluru, India",
              fontsize=14, color=CYAN, ha="center", va="top", transform=ax_title.transAxes)
ax_title.text(0.5, 0.10, "Divya Dhole  |  M.S. Data Science  |  github.com/Divyadhole",
              fontsize=11, color=LIGHT, ha="center", va="top", transform=ax_title.transAxes)
ax_title.axhline(y=0.0, color=CYAN, linewidth=1.5, alpha=0.6)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1  ─  KPI Metrics  (3 cells merged as one row of annotations)
# ═══════════════════════════════════════════════════════════════════════════════
kpi_keys   = ["Total Reviews", "Positive", "Negative", "Flagged (≥1 concern)", "Type A Mismatches", "Type B Mismatches"]
kpi_vals   = [f"{n_total:,}", f"{n_pos:,}\n72.9%", f"{n_neg:,}\n27.1%",
              f"{n_flagged:,}\n{n_flagged/n_total*100:.1f}%",
              f"{type_a:,}\n{type_a/n_total*100:.1f}%",
              f"{type_b:,}\n{type_b/n_total*100:.1f}%"]
kpi_colors = [CYAN, GREEN, RED, AMBER, ORANGE, PURPLE]

ax_kpi = fig.add_subplot(gs[1, :])
ax_kpi.set_facecolor(BG)
ax_kpi.axis("off")

# Section label
ax_kpi.text(0.0, 1.02, "  DATASET AT A GLANCE", fontsize=9, color=CYAN,
            fontweight="bold", transform=ax_kpi.transAxes, va="bottom")

for i, (key, val, col) in enumerate(zip(kpi_keys, kpi_vals, kpi_colors)):
    x0 = i / 6 + 0.005
    rect = FancyBboxPatch((x0, 0.02), 0.15, 0.9,
                          boxstyle="round,pad=0.01", linewidth=2,
                          edgecolor=col, facecolor=PANEL_BG,
                          transform=ax_kpi.transAxes, zorder=1)
    ax_kpi.add_patch(rect)
    xc = x0 + 0.075
    ax_kpi.text(xc, 0.72, val.split("\n")[0], fontsize=22, fontweight="bold",
                color=col, ha="center", va="center", transform=ax_kpi.transAxes)
    if "\n" in val:
        ax_kpi.text(xc, 0.42, val.split("\n")[1], fontsize=12, color=LIGHT,
                    ha="center", va="center", transform=ax_kpi.transAxes)
    ax_kpi.text(xc, 0.14, key, fontsize=9, color=LIGHT,
                ha="center", va="center", transform=ax_kpi.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2  ─  3 charts
# ═══════════════════════════════════════════════════════════════════════════════

cat_labels = list(CONCERN_LEXICON.keys())
cat_labels_short = ["Mistreat-\nment", "Access\nBarriers", "Delays /\nWait", "Safety /\nHygiene"]

# ── Chart A: Concern flags by category (overall) ──────────────────────────────
ax_a = fig.add_subplot(gs[2, 0])
rates_overall = df[flag_cols].mean() * 100
ax_a.bar(cat_labels_short, rates_overall.values, color=CAT_CLRS, edgecolor=BG, linewidth=0.8, width=0.6)
ax_a.set_title("Concern Flag Rate by Category", fontsize=11, color=WHITE, fontweight="bold", pad=8)
ax_a.set_ylabel("% of Reviews Flagged", fontsize=9, color=LIGHT)
ax_a.yaxis.grid(True)
ax_a.set_axisbelow(True)
for bar, val in zip(ax_a.patches, rates_overall.values):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              f"{val:.1f}%", ha="center", fontsize=9, color=WHITE, fontweight="bold")
ax_a.tick_params(axis="x", labelsize=8)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# ── Chart B: Concern rate × sentiment (grouped bars) ─────────────────────────
ax_b = fig.add_subplot(gs[2, 1])
x = np.arange(len(cat_labels))
w = 0.35
ax_b.bar(x - w/2, flag_rate_pos.values, w, label="Positive", color=GREEN, edgecolor=BG, linewidth=0.8)
ax_b.bar(x + w/2, flag_rate_neg.values, w, label="Negative", color=RED,   edgecolor=BG, linewidth=0.8)
ax_b.set_title("Concern Rate × Sentiment", fontsize=11, color=WHITE, fontweight="bold", pad=8)
ax_b.set_ylabel("% of Reviews Flagged", fontsize=9, color=LIGHT)
ax_b.set_xticks(x)
ax_b.set_xticklabels(cat_labels_short, fontsize=8)
ax_b.legend(fontsize=9, framealpha=0.2, labelcolor=WHITE, facecolor=PANEL_BG)
ax_b.yaxis.grid(True)
ax_b.set_axisbelow(True)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# ── Chart C: Mismatch breakdown ───────────────────────────────────────────────
ax_c = fig.add_subplot(gs[2, 2])
consistent = n_total - type_a - type_b
labels_m = ["Consistent", "Type A\n(Positive + Flags)", "Type B\n(Negative, No Flags)"]
sizes_m  = [consistent, type_a, type_b]
cols_m   = ["#2C4A6E", AMBER, PURPLE]
wedges, texts, autotexts = ax_c.pie(
    sizes_m, labels=labels_m, colors=cols_m,
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": BG, "linewidth": 2},
    textprops={"color": WHITE, "fontsize": 9},
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
ax_c.set_title("Mismatch Analysis", fontsize=11, color=WHITE, fontweight="bold", pad=8)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3  ─  3 more panels
# ═══════════════════════════════════════════════════════════════════════════════

# ── Chart D: Concern lift (neg − pos) ─────────────────────────────────────────
ax_d = fig.add_subplot(gs[3, 0])
lift_vals = lift.values
lift_cols = [RED if v >= 0 else GREEN for v in lift_vals]
bars = ax_d.barh(cat_labels_short, lift_vals, color=lift_cols, edgecolor=BG, linewidth=0.8, height=0.5)
ax_d.axvline(0, color=WHITE, linewidth=0.8, alpha=0.5)
ax_d.set_title("Concern Lift in Negative Reviews\n(Neg rate − Pos rate, %pts)", fontsize=10, color=WHITE, fontweight="bold", pad=8)
ax_d.set_xlabel("Percentage-point difference", fontsize=9, color=LIGHT)
for bar, val in zip(bars, lift_vals):
    xpos = val + 0.05 if val >= 0 else val - 0.05
    ha   = "left" if val >= 0 else "right"
    ax_d.text(xpos, bar.get_y() + bar.get_height()/2, f"{val:+.1f}%",
              va="center", ha=ha, fontsize=9, color=WHITE, fontweight="bold")
ax_d.spines["top"].set_visible(False)
ax_d.spines["right"].set_visible(False)
ax_d.xaxis.grid(True)
ax_d.set_axisbelow(True)

# ── Chart E: Sentiment distribution bar ──────────────────────────────────────
ax_e = fig.add_subplot(gs[3, 1])
ax_e.bar(["Positive", "Negative"], [n_pos, n_neg], color=[GREEN, RED],
         edgecolor=BG, linewidth=0.8, width=0.5)
ax_e.set_title("Sentiment Label Distribution", fontsize=11, color=WHITE, fontweight="bold", pad=8)
ax_e.set_ylabel("Number of Reviews", fontsize=9, color=LIGHT)
for bar, val in zip(ax_e.patches, [n_pos, n_neg]):
    ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
              f"{val:,}\n({val/n_total*100:.1f}%)", ha="center", fontsize=10,
              color=WHITE, fontweight="bold")
ax_e.spines["top"].set_visible(False)
ax_e.spines["right"].set_visible(False)
ax_e.yaxis.grid(True)
ax_e.set_axisbelow(True)

# ── Panel F: Methodology & Ethics text panel ──────────────────────────────────
ax_f = fig.add_subplot(gs[3, 2])
ax_f.set_facecolor(PANEL_BG)
ax_f.axis("off")

steps = [
    ("1 ─ Ingest",      "Google Maps reviews (text + sentiment)"),
    ("2 ─ Clean",       "Lowercase, normalise, tokenise"),
    ("3 ─ Flag",        "4-category lexicon → binary concern flags"),
    ("4 ─ Analyse",     "Flag rates, mismatch cases, lift analysis"),
    ("5 ─ Topics",      "LDA on negative reviews (6 themes)"),
    ("6 ─ Baseline",    "TF-IDF + LR  →  Acc 87%  |  AUC 0.91"),
]

ax_f.text(0.05, 0.97, "Pipeline Steps", fontsize=11, color=CYAN,
          fontweight="bold", va="top", transform=ax_f.transAxes)

for i, (step, desc) in enumerate(steps):
    y = 0.87 - i * 0.135
    ax_f.text(0.05, y, step, fontsize=9.5, color=AMBER, fontweight="bold",
              va="top", transform=ax_f.transAxes)
    ax_f.text(0.38, y, desc, fontsize=9, color=LIGHT,
              va="top", transform=ax_f.transAxes)

ax_f.plot([0.02, 0.98], [0.06, 0.06], color=RED, linewidth=0.8, alpha=0.5,
          transform=ax_f.transAxes, clip_on=False)
ax_f.text(0.5, 0.03, "⚠  Patterns ≠ Proof of Intent  |  Human review required",
          fontsize=8.5, color=RED, ha="center", va="bottom",
          transform=ax_f.transAxes, style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER — tech stack
# ═══════════════════════════════════════════════════════════════════════════════
stack = ["Python 3.9+", "Pandas", "NumPy", "scikit-learn", "Matplotlib", "Streamlit", "Jupyter"]
fig.text(0.5, 0.005,
         "Tech Stack:   " + "  ·  ".join(stack),
         fontsize=9, color=LIGHT, ha="center", va="bottom")

# ── Save ───────────────────────────────────────────────────────────────────────
out = ROOT / "reports" / "poster.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"✅  Poster saved → {out}")
plt.close(fig)
