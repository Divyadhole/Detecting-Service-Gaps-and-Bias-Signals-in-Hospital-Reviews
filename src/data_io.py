"""
data_io.py
----------
Load raw CSV / JSON data and perform basic schema validation.
"""

import pandas as pd
import logging
from pathlib import Path

from .config import (
    COL_TEXT, COL_SENTIMENT, COL_HOSPITAL, COL_RATING,
    POSITIVE_LABEL_VALUES,
)

logger = logging.getLogger(__name__)


def load_reviews(filepath: str | Path, filetype: str = "csv") -> pd.DataFrame:
    """
    Load reviews from a CSV or JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw data file.
    filetype : str
        'csv' or 'json'.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with at minimum: review_text, sentiment, hospital_name.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Place your dataset in data/raw/ — see data/README_DATA.md"
        )

    logger.info(f"Loading data from {filepath}")
    if filetype == "csv":
        df = pd.read_csv(filepath)
    elif filetype == "json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported filetype: {filetype}. Use 'csv' or 'json'.")

    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assert required columns exist and perform basic coercions.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Validated (and lightly cleaned) DataFrame.
    """
    required = [COL_TEXT, COL_SENTIMENT, COL_HOSPITAL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Update src/config.py to match your dataset's column names."
        )

    # Drop rows with null review text or sentiment
    before = len(df)
    df = df.dropna(subset=[COL_TEXT, COL_SENTIMENT]).copy()
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with null text or sentiment.")

    # Normalise sentiment to binary integer (1 = positive, 0 = negative)
    df["sentiment_binary"] = df[COL_SENTIMENT].apply(
        lambda v: 1 if str(v).strip().lower() in {str(x).lower() for x in POSITIVE_LABEL_VALUES} else 0
    )

    # Optional: coerce rating to float if present
    if COL_RATING in df.columns:
        df[COL_RATING] = pd.to_numeric(df[COL_RATING], errors="coerce")

    logger.info(
        f"Schema validated. Positive: {df['sentiment_binary'].sum():,} | "
        f"Negative: {(df['sentiment_binary'] == 0).sum():,}"
    )
    return df


def load_and_validate(filepath: str | Path, filetype: str = "csv") -> pd.DataFrame:
    """Convenience wrapper: load + validate in one call."""
    df = load_reviews(filepath, filetype)
    df = validate_schema(df)
    return df
