"""
config.py
---------
Central configuration: file paths, column name mappings, and lexicon definitions.
Edit this file to match your actual dataset schema.
"""

import os
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Input files ───────────────────────────────────────────────────────────────
RAW_CSV = DATA_RAW_DIR / "reviews.csv"

# ── Column name mapping (adjust to match your dataset) ───────────────────────
COL_TEXT = "review_text"       # review body
COL_SENTIMENT = "sentiment"    # e.g. "positive" / "negative" or 1 / 0
COL_HOSPITAL = "hospital_name"
COL_RATING = "rating"          # optional star rating 1-5
COL_DATE = "review_date"       # optional

# Positive label value(s) — used to binarise sentiment
POSITIVE_LABEL_VALUES = {"positive", "pos", 1, "1"}

# ── Concern lexicon ───────────────────────────────────────────────────────────
# Each key is a concern category; each value is a list of trigger keywords.
# All matching is case-insensitive and substring-based.
CONCERN_LEXICON = {
    "mistreatment": [
        "rude", "rudely", "ignored", "dismissive", "disrespectful",
        "insulted", "humiliated", "yelled", "arrogant", "negligent",
        "misbehaved", "unprofessional", "attitude", "shouted", "refused",
    ],
    "access_barriers": [
        "wheelchair", "ramp", "accessible", "accessibility", "lift",
        "elevator", "stairs", "disabled", "handicap", "parking",
        "entrance", "navigation", "blind", "mobility",
    ],
    "delays_wait": [
        "waiting", "wait", "waited", "hours", "queue", "queued",
        "delayed", "delay", "slow", "late", "forever", "long time",
        "appointment", "scheduled", "overdue", "hours long",
    ],
    "safety_hygiene": [
        "dirty", "unclean", "unhygienic", "hygiene", "infection",
        "infested", "cockroach", "rats", "smell", "smelly", "stench",
        "sterile", "contaminated", "unsanitary", "blood stained",
        "broken equipment",
    ],
}

# ── Model parameters ──────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 10_000
LR_MAX_ITER = 500
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Topic modelling ───────────────────────────────────────────────────────────
LDA_NUM_TOPICS = 8
LDA_PASSES = 10
LDA_RANDOM_STATE = 42
