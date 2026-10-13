"""Metric calculations and evaluation plots.

Accuracy is reported beside the positive rate so a majority-class score is not
mistaken for ranking quality. ROC-AUC and PR-AUC do not use the decision threshold.
Precision, recall, and F1 use a threshold chosen on validation data.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve


def positive_probability(pipeline, features: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class, even if the estimator saw one class."""
    model = pipeline.named_steps["model"]
    probabilities = pipeline.predict_proba(features)
    classes = [int(value) for value in model.classes_]
    if 1 not in classes:
        return np.zeros(len(features), dtype=float)
    return probabilities[:, classes.index(1)]


def metric_row(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        raise ValueError("Cannot score an empty label vector.")
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (int(value) for value in matrix.ravel())
    row = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))),
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
    }
    if len(np.unique(y_true)) > 1:
        row["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        row["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        row["roc_auc"] = None
        row["pr_auc"] = None
    return row


def threshold_for_f1(y_true, y_prob) -> float:
    """Pick a threshold on validation labels. Do not call this with test labels."""
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.round(np.linspace(0.05, 0.95, 91), 2):
        score = f1_score(y_true, np.asarray(y_prob) >= threshold, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def save_evaluation_plots(
    y_true,
    y_prob,
    threshold: float,
    output_dir: Path,
    stem: str,
) -> dict[str, str]:
    """Save confusion, ROC, precision-recall, and calibration figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    paths: dict[str, str] = {}

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=False,
        xticklabels=["Predicted no purchase", "Predicted purchase"],
        yticklabels=["No purchase", "Purchase"],
        ax=axis,
    )
    axis.set_title(f"Test confusion matrix at threshold {threshold:.2f}")
    fig.tight_layout()
    confusion_path = output_dir / f"{stem}_confusion_matrix.png"
    fig.savefig(confusion_path, dpi=120)
    plt.close(fig)
    paths["confusion_matrix"] = str(confusion_path)

    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig, axis = plt.subplots(figsize=(6, 4.2))
        axis.plot(recall, precision, color="#0F6E6E", linewidth=2)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Test precision-recall curve")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        fig.tight_layout()
        pr_path = output_dir / f"{stem}_precision_recall.png"
        fig.savefig(pr_path, dpi=120)
        plt.close(fig)
        paths["precision_recall"] = str(pr_path)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, axis = plt.subplots(figsize=(6, 4.2))
        axis.plot(fpr, tpr, color="#0F6E6E", linewidth=2)
        axis.plot([0, 1], [0, 1], color="#A8A29E", linestyle="--", linewidth=1)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title("Test ROC curve")
        fig.tight_layout()
        roc_path = output_dir / f"{stem}_roc.png"
        fig.savefig(roc_path, dpi=120)
        plt.close(fig)
        paths["roc"] = str(roc_path)

        fraction, mean_predicted = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
        fig, axis = plt.subplots(figsize=(6, 4.2))
        axis.plot(mean_predicted, fraction, marker="o", color="#C46B3A", linewidth=2)
        axis.plot([0, 1], [0, 1], color="#A8A29E", linestyle="--", linewidth=1)
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed purchase rate")
        axis.set_title("Test calibration")
        fig.tight_layout()
        calibration_path = output_dir / f"{stem}_calibration.png"
        fig.savefig(calibration_path, dpi=120)
        plt.close(fig)
        paths["calibration"] = str(calibration_path)
        pd.DataFrame({"mean_predicted": mean_predicted, "observed_rate": fraction}).to_csv(
            output_dir / f"{stem}_calibration_bins.csv",
            index=False,
        )
    return paths
