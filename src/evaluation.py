"""
evaluation.py
-------------
Model evaluation: metrics, confusion matrix, ROC, and error analysis helpers.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)

from .config import FIGURES_DIR

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    y_prob : array-like, optional — predicted probabilities for ROC-AUC

    Returns
    -------
    dict
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def print_report(y_true, y_pred, title: str = "Classification Report") -> None:
    """Print sklearn classification report."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))


def plot_confusion_matrix(y_true, y_pred,
                           title: str = "Confusion Matrix",
                           save: bool = True,
                           filename: str = "confusion_matrix.png") -> plt.Figure:
    """
    Plot and optionally save a confusion matrix heatmap.

    Parameters
    ----------
    save : bool
    filename : str

    Returns
    -------
    matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
        xlabel="Predicted", ylabel="Actual",
        title=title,
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    return fig


def plot_roc_curve(y_true, y_prob,
                   title: str = "ROC Curve",
                   save: bool = True,
                   filename: str = "roc_curve.png") -> plt.Figure:
    """
    Plot ROC curve for the binary classifier.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1, linestyle="--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    return fig


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a comparison table for multiple model variants.

    Parameters
    ----------
    results : dict
        {model_name: metrics_dict}

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = metrics.
    """
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def error_analysis(df: pd.DataFrame,
                   y_true_col: str = "sentiment_binary",
                   y_pred_col: str = "pred_ml",
                   text_col: str = "review_text",
                   n: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Sample false positives and false negatives for qualitative error analysis.

    Parameters
    ----------
    df : pd.DataFrame  (must have y_true_col and y_pred_col columns)
    n : int   samples per error type

    Returns
    -------
    dict with 'false_positives' and 'false_negatives' DataFrames
    """
    fp = df[(df[y_true_col] == 0) & (df[y_pred_col] == 1)]
    fn = df[(df[y_true_col] == 1) & (df[y_pred_col] == 0)]
    return {
        "false_positives": fp[[text_col, y_true_col, y_pred_col]].head(n),
        "false_negatives": fn[[text_col, y_true_col, y_pred_col]].head(n),
    }
