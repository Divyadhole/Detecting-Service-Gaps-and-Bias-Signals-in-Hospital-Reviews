"""
preprocess.py
-------------
Text cleaning, normalisation, and tokenisation utilities.
All functions are pure (input → output) for easy unit testing.
"""

import re
import string
import logging
from typing import List

import nltk

logger = logging.getLogger(__name__)

# Download stopwords on first use (cached after that)
try:
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words("english"))


# ── Low-level helpers ─────────────────────────────────────────────────────────

def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Strip http/https URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags (defensive — reviews sometimes contain them)."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters."""
    return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))


def remove_numbers(text: str) -> str:
    """Remove standalone digit sequences."""
    return re.sub(r"\b\d+\b", " ", text)


def normalise_whitespace(text: str) -> str:
    """Collapse multiple spaces / newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Drop common English stopwords from a token list."""
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ── Pipeline ──────────────────────────────────────────────────────────────────

def clean_text(text: str, remove_stops: bool = False) -> str:
    """
    Full cleaning pipeline for a single review string.

    Steps: lowercase → strip URLs → strip HTML → remove punctuation
           → remove standalone numbers → normalise whitespace
           → (optional) remove stopwords

    Parameters
    ----------
    text : str
    remove_stops : bool
        If True, stopwords are removed (useful for topic modelling).

    Returns
    -------
    str
        Cleaned text (space-joined tokens if stopwords removed).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = normalise_whitespace(text)

    if remove_stops:
        tokens = text.split()
        tokens = remove_stopwords(tokens)
        text = " ".join(tokens)

    return text


def tokenise(text: str) -> List[str]:
    """
    Simple whitespace tokeniser.
    Returns list of lowercased, non-empty tokens.
    """
    return [t for t in text.lower().split() if t]


def apply_cleaning(series, remove_stops: bool = False):
    """
    Apply clean_text to a pandas Series.

    Parameters
    ----------
    series : pd.Series
    remove_stops : bool

    Returns
    -------
    pd.Series
        Cleaned text series.
    """
    return series.fillna("").apply(lambda t: clean_text(t, remove_stops=remove_stops))
