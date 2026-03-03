"""
modeling.py
-----------
TF-IDF + Logistic Regression baseline for sentiment prediction.
Provides a simple, interpretable ML baseline to compare against the lexicon heuristic.
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    TFIDF_MAX_FEATURES, LR_MAX_ITER, RANDOM_STATE, TEST_SIZE,
    DATA_PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


def build_pipeline(max_features: int = TFIDF_MAX_FEATURES,
                   max_iter: int = LR_MAX_ITER,
                   random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Construct a TF-IDF → Logistic Regression sklearn Pipeline.

    Parameters
    ----------
    max_features : int
    max_iter : int
    random_state : int

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("lr", LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            class_weight="balanced",   # handles imbalanced sentiment splits
        )),
    ])
    return pipeline


def split_data(df: pd.DataFrame,
               text_col: str,
               label_col: str = "sentiment_binary",
               test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE) -> Tuple:
    """
    Train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df[text_col].fillna("")
    y = df[label_col]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def train_model(X_train, y_train, pipeline: Optional[Pipeline] = None) -> Pipeline:
    """
    Fit the pipeline on training data.

    Parameters
    ----------
    X_train : array-like of str
    y_train : array-like of int
    pipeline : Pipeline, optional. Built fresh if not provided.

    Returns
    -------
    Fitted Pipeline
    """
    if pipeline is None:
        pipeline = build_pipeline()
    logger.info("Training TF-IDF + Logistic Regression model...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")
    return pipeline


def save_model(pipeline: Pipeline, path: Optional[Path] = None) -> Path:
    """Persist trained pipeline to disk (pickle)."""
    if path is None:
        path = DATA_PROCESSED_DIR / "tfidf_lr_model.pkl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved to {path}")
    return path


def load_model(path: Optional[Path] = None) -> Pipeline:
    """Load a previously saved pipeline."""
    if path is None:
        path = DATA_PROCESSED_DIR / "tfidf_lr_model.pkl"
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def get_top_features(pipeline: Pipeline, n: int = 20) -> pd.DataFrame:
    """
    Extract top positive/negative TF-IDF features from the logistic regression.
    Useful for interpretability in interviews.

    Parameters
    ----------
    pipeline : fitted Pipeline
    n : int
        Number of top features per class.

    Returns
    -------
    pd.DataFrame
        Columns: ['feature', 'coefficient', 'direction']
    """
    tfidf = pipeline.named_steps["tfidf"]
    lr = pipeline.named_steps["lr"]
    features = np.array(tfidf.get_feature_names_out())
    coefs = lr.coef_[0]

    idx_top_pos = np.argsort(coefs)[-n:][::-1]
    idx_top_neg = np.argsort(coefs)[:n]

    rows = []
    for i in idx_top_pos:
        rows.append({"feature": features[i], "coefficient": coefs[i], "direction": "positive"})
    for i in idx_top_neg:
        rows.append({"feature": features[i], "coefficient": coefs[i], "direction": "negative"})

    return pd.DataFrame(rows)


def lexicon_heuristic_predict(df: pd.DataFrame,
                               flag_col: str = "flag_any_concern") -> np.ndarray:
    """
    Simple lexicon heuristic: predict negative (0) if any concern flag is raised,
    positive (1) otherwise. Used as a baseline to compare against the ML model.

    Parameters
    ----------
    df : pd.DataFrame
    flag_col : str

    Returns
    -------
    np.ndarray of 0/1 predictions
    """
    return (df[flag_col] == 0).astype(int).values
