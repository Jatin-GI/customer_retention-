"""Train the baseline and the candidate models on a temporal split.

Hyperparameters are chosen on the last training snapshot. Models are compared
on the validation snapshot by PR-AUC. The test snapshot is scored after that
choice and is not used to pick a model or a threshold.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluate import metric_row, positive_probability, threshold_for_f1
from src.preprocessing import build_preprocessor

logger = logging.getLogger("retention.train")


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Modeling table is missing features: {missing}")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    labels = frame[TARGET_COLUMN].astype(int).to_numpy()
    return features, labels


def _fit_pipeline(estimator, features: pd.DataFrame, labels: np.ndarray, scale: bool) -> Pipeline:
    model = clone(estimator)
    if isinstance(model, XGBClassifier):
        positives = max(int((labels == 1).sum()), 1)
        negatives = int((labels == 0).sum())
        model.set_params(scale_pos_weight=negatives / positives)
    pipeline = Pipeline(
        [
            ("prep", build_preprocessor(scale=scale)),
            ("model", model),
        ]
    )
    pipeline.fit(features, labels)
    return pipeline


def _safe_pr_auc(y_true, y_prob) -> float:
    row = metric_row(y_true, y_prob, threshold=0.5)
    if row["pr_auc"] is None:
        return -1.0
    return float(row["pr_auc"])


def model_catalog(seed: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "dummy_prior",
            "scale": False,
            "estimator": DummyClassifier(strategy="prior", random_state=seed),
            "grid": [{}],
        },
        {
            "name": "logistic_regression",
            "scale": True,
            "estimator": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            ),
            "grid": [{"C": 0.1}, {"C": 1.0}, {"C": 3.0}],
        },
        {
            "name": "random_forest",
            "scale": False,
            "estimator": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            ),
            "grid": [
                {"max_depth": 8, "min_samples_leaf": 5},
                {"max_depth": 8, "min_samples_leaf": 20},
                {"max_depth": 16, "min_samples_leaf": 5},
                {"max_depth": 16, "min_samples_leaf": 20},
            ],
        },
        {
            "name": "xgboost",
            "scale": False,
            "estimator": XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            ),
            "grid": [
                {"max_depth": 3, "learning_rate": 0.05},
                {"max_depth": 3, "learning_rate": 0.1},
                {"max_depth": 5, "learning_rate": 0.05},
                {"max_depth": 5, "learning_rate": 0.1},
            ],
        },
    ]


def train_models(modeling: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Fit every candidate and return pipelines, metrics, and test scores."""
    required = {"split", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required.difference(modeling.columns)
    if missing:
        raise ValueError(f"Modeling table is missing columns: {sorted(missing)}")
    frame = modeling.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test splits must all be non-empty.")
    if set(train["snapshot_date"]) & set(test["snapshot_date"]):
        raise ValueError("Training and test snapshots overlap.")
    if set(validation["snapshot_date"]) & set(test["snapshot_date"]):
        raise ValueError("Validation and test snapshots overlap.")

    tune_dates = sorted(train["snapshot_date"].unique())
    if len(tune_dates) >= 2:
        holdout_date = tune_dates[-1]
        tune_train = train[train["snapshot_date"] < holdout_date]
        tune_valid = train[train["snapshot_date"] == holdout_date]
    else:
        holdout_date = None
        tune_train = train
        tune_valid = None
    if tune_valid is not None and (tune_train.empty or tune_valid[TARGET_COLUMN].nunique() < 2):
        logger.info("Inner time split cannot score hyperparameters. Using default parameters.")
        tune_valid = None

    x_train, y_train = _xy(train)
    x_validation, y_validation = _xy(validation)
    x_test, y_test = _xy(test)
    comparison = []
    fitted = {}

    for spec in model_catalog(seed):
        best_params: dict[str, Any] = spec["grid"][0]
        best_inner = None
        if tune_valid is not None and spec["grid"] != [{}]:
            x_inner, y_inner = _xy(tune_train)
            x_holdout, y_holdout = _xy(tune_valid)
            best_inner = -1.0
            for params in spec["grid"]:
                candidate = clone(spec["estimator"]).set_params(**params)
                pipeline = _fit_pipeline(candidate, x_inner, y_inner, spec["scale"])
                score = _safe_pr_auc(y_holdout, positive_probability(pipeline, x_holdout))
                logger.info("Inner PR-AUC %s %s = %.4f", spec["name"], params, score)
                if score > best_inner:
                    best_inner = score
                    best_params = params
        estimator = clone(spec["estimator"]).set_params(**best_params)
        pipeline = _fit_pipeline(estimator, x_train, y_train, spec["scale"])
        validation_prob = positive_probability(pipeline, x_validation)
        threshold = threshold_for_f1(y_validation, validation_prob)
        validation_metrics = metric_row(y_validation, validation_prob, threshold)
        test_prob = positive_probability(pipeline, x_test)
        test_metrics = metric_row(y_test, test_prob, threshold)
        fitted[spec["name"]] = {
            "pipeline": pipeline,
            "threshold": threshold,
            "params": best_params,
            "inner_pr_auc": best_inner,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_probability": test_prob,
        }
        for split_name, metrics in (("validation", validation_metrics), ("test", test_metrics)):
            comparison.append({"model": spec["name"], "split": split_name, "params": best_params, **metrics})
        logger.info(
            "%s validation PR-AUC=%s F1=%s",
            spec["name"],
            validation_metrics["pr_auc"],
            validation_metrics["f1"],
        )

    def sort_key(name: str) -> tuple:
        metrics = fitted[name]["validation_metrics"]
        pr_auc = -1.0 if metrics["pr_auc"] is None else metrics["pr_auc"]
        return (pr_auc, metrics["f1"], name)

    selected = max(fitted, key=sort_key)
    selected_bundle = fitted[selected]
    test_scores = test.copy()
    test_scores["y_true"] = y_test
    test_scores["y_prob"] = selected_bundle["test_probability"]
    test_scores["y_pred"] = (test_scores["y_prob"] >= selected_bundle["threshold"]).astype(int)

    dummy_threshold = fitted["dummy_prior"]["threshold"]
    dummy_accuracy = fitted["dummy_prior"]["test_metrics"]["accuracy"]
    selected_accuracy = selected_bundle["test_metrics"]["accuracy"]
    positive_rate = selected_bundle["test_metrics"]["positive_rate"]
    majority = max(positive_rate, 1 - positive_rate)
    accuracy_note = (
        f"On the test snapshot, {positive_rate:.1%} of eligible customers purchased again "
        f"within the horizon. A rule that predicts no purchase for every customer would be right "
        f"about {majority:.1%} of the time and would have recall 0. The dummy prior model scores "
        f"every customer at the training base rate. Its F1-maximizing validation threshold is "
        f"{dummy_threshold:.2f}, which labels every test customer as a purchaser, so its test "
        f"accuracy equals the positive rate ({dummy_accuracy:.3f}) and its recall is 1. "
        f"The selected model's test accuracy was {selected_accuracy:.3f}. That is a modest gap "
        f"versus the {majority:.1%} majority-class rule, which is why accuracy is a weak way to "
        "choose a model when the classes are uneven. Model choice uses validation PR-AUC."
    )
    return {
        "selected_model": selected,
        "selection_metric": "validation_pr_auc",
        "selection_rule": (
            "Highest validation average precision. Ties break toward higher validation F1 "
            "at the validation-chosen threshold, then model name."
        ),
        "threshold": selected_bundle["threshold"],
        "threshold_rule": "Threshold maximizes F1 on the validation snapshot and is frozen before test scoring.",
        "pipelines": {name: bundle["pipeline"] for name, bundle in fitted.items()},
        "selected_pipeline": selected_bundle["pipeline"],
        "comparison": comparison,
        "test_scores": test_scores,
        "accuracy_note": accuracy_note,
        "inner_holdout_snapshot": None if holdout_date is None else str(pd.Timestamp(holdout_date).date()),
        "fitted_details": {
            name: {
                "params": bundle["params"],
                "inner_pr_auc": bundle["inner_pr_auc"],
                "threshold": bundle["threshold"],
                "validation_metrics": bundle["validation_metrics"],
                "test_metrics": bundle["test_metrics"],
            }
            for name, bundle in fitted.items()
        },
    }
