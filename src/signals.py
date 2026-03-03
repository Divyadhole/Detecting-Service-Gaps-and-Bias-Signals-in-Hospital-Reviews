"""
signals.py
----------
Configurable lexicon-based concern flag computation.

Each review is checked against the CONCERN_LEXICON in config.py.
Results are binary columns (0/1) — fully traceable to specific keywords.
"""

import re
import logging
from typing import Dict, List

import pandas as pd

from .config import CONCERN_LEXICON, COL_TEXT

logger = logging.getLogger(__name__)


# ── Core flag logic ───────────────────────────────────────────────────────────

def _contains_keyword(text: str, keywords: List[str]) -> bool:
    """
    Return True if any keyword appears as a word/substring in text.
    Matching is case-insensitive.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _matched_keywords(text: str, keywords: List[str]) -> List[str]:
    """
    Return the list of keywords that matched in text (for explainability).
    """
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower]


def flag_review(text: str, lexicon: Dict[str, List[str]] = None) -> Dict[str, int]:
    """
    Compute concern flags for a single review string.

    Parameters
    ----------
    text : str
        Cleaned review text.
    lexicon : dict, optional
        Category → keyword list. Defaults to CONCERN_LEXICON from config.

    Returns
    -------
    dict
        {category: 0 or 1}
    """
    if lexicon is None:
        lexicon = CONCERN_LEXICON
    return {category: int(_contains_keyword(text, kws)) for category, kws in lexicon.items()}


def explain_flags(text: str, lexicon: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
    """
    Return the specific trigger keywords for each category in a review.
    Used for interview demos and spot-checks.

    Parameters
    ----------
    text : str
    lexicon : dict, optional

    Returns
    -------
    dict
        {category: [matched_keywords]}
    """
    if lexicon is None:
        lexicon = CONCERN_LEXICON
    return {cat: _matched_keywords(text, kws) for cat, kws in lexicon.items()}


# ── DataFrame-level helpers ───────────────────────────────────────────────────

def compute_concern_flags(df: pd.DataFrame,
                           text_col: str = None,
                           lexicon: Dict[str, List[str]] = None) -> pd.DataFrame:
    """
    Add one binary flag column per concern category to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
        Column containing cleaned review text. Defaults to COL_TEXT.
    lexicon : dict, optional

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new flag columns appended.
    """
    if text_col is None:
        text_col = COL_TEXT
    if lexicon is None:
        lexicon = CONCERN_LEXICON

    logger.info(f"Computing concern flags for {len(df):,} reviews...")
    flags = df[text_col].fillna("").apply(lambda t: pd.Series(flag_review(t, lexicon)))
    df = df.copy()
    for col in flags.columns:
        df[f"flag_{col}"] = flags[col].values

    # Composite: any concern flagged
    flag_cols = [f"flag_{c}" for c in lexicon]
    df["flag_any_concern"] = (df[flag_cols].sum(axis=1) > 0).astype(int)

    total_flagged = df["flag_any_concern"].sum()
    logger.info(f"Flagged {total_flagged:,} reviews ({total_flagged/len(df)*100:.1f}%) with ≥1 concern.")
    return df


def mismatch_analysis(df: pd.DataFrame,
                       sentiment_col: str = "sentiment_binary") -> pd.DataFrame:
    """
    Identify mismatch cases — the most valuable for quality assurance:

    Type A: positive sentiment label but ≥1 concern flag
    Type B: negative sentiment label but 0 concern flags

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'sentiment_binary' and 'flag_any_concern'.

    Returns
    -------
    pd.DataFrame
        Mismatch subset with a 'mismatch_type' column.
    """
    type_a = df[(df[sentiment_col] == 1) & (df["flag_any_concern"] == 1)].copy()
    type_a["mismatch_type"] = "A: positive_label+concern_flags"

    type_b = df[(df[sentiment_col] == 0) & (df["flag_any_concern"] == 0)].copy()
    type_b["mismatch_type"] = "B: negative_label+no_flags"

    mismatches = pd.concat([type_a, type_b], ignore_index=True)
    logger.info(
        f"Mismatch analysis: Type A={len(type_a):,} | Type B={len(type_b):,} | "
        f"Total={len(mismatches):,}"
    )
    return mismatches


def concern_rate_by_hospital(df: pd.DataFrame,
                              hospital_col: str = None,
                              min_reviews: int = 5) -> pd.DataFrame:
    """
    Compute per-category concern rate (%) for each hospital.

    Parameters
    ----------
    df : pd.DataFrame
    hospital_col : str
        Defaults to config.COL_HOSPITAL.
    min_reviews : int
        Hospitals with fewer reviews are excluded.

    Returns
    -------
    pd.DataFrame
        hospital × concern_category concern rate table.
    """
    from .config import COL_HOSPITAL
    if hospital_col is None:
        hospital_col = COL_HOSPITAL

    flag_cols = [c for c in df.columns if c.startswith("flag_") and c != "flag_any_concern"]
    grouped = df.groupby(hospital_col)
    counts = grouped[flag_cols].sum()
    totals = grouped.size().rename("total_reviews")
    rates = counts.div(totals, axis=0) * 100
    rates["total_reviews"] = totals

    # Filter out low-sample hospitals
    rates = rates[rates["total_reviews"] >= min_reviews]
    rates.columns = [c.replace("flag_", "") for c in rates.columns]
    return rates.sort_values("any_concern" if "any_concern" in rates.columns else rates.columns[0],
                             ascending=False)
