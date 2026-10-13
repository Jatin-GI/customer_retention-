"""Validate a single customer feature row and score it with a saved pipeline."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, SPARSE_NUMERIC_FEATURES
from src.evaluate import positive_probability


class FeatureInputError(ValueError):
    """The dashboard or API caller sent a feature row that cannot be scored."""


def frame_from_user(payload: dict) -> pd.DataFrame:
    """Build a one-row feature frame. Blank sparse features become missing values."""
    errors: list[str] = []
    row: dict = {}
    for column in NUMERIC_FEATURES + SPARSE_NUMERIC_FEATURES:
        raw = payload.get(column, None)
        blank = raw is None or (isinstance(raw, str) and raw.strip() == "")
        if blank and column in SPARSE_NUMERIC_FEATURES:
            row[column] = np.nan
            continue
        if blank:
            errors.append(f"{column} is required.")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{column} must be a number.")
            continue
        if not math.isfinite(value):
            errors.append(f"{column} must be a finite number.")
            continue
        if value < 0:
            errors.append(f"{column} cannot be negative.")
        row[column] = value
    if "return_rate" in row and math.isfinite(row["return_rate"]) and not 0 <= row["return_rate"] <= 1:
        errors.append("return_rate must be between 0 and 1.")
    if "frequency" in row and row["frequency"] < 1:
        errors.append("frequency must be at least 1.")
    if "tenure_days" in row and "recency_days" in row and row["tenure_days"] < row["recency_days"]:
        errors.append("tenure_days cannot be shorter than recency_days.")
    country = payload.get("country")
    if not isinstance(country, str) or not country.strip():
        errors.append("country is required.")
    else:
        row["country"] = country.strip()
    if errors:
        raise FeatureInputError(" ".join(errors))
    return pd.DataFrame([row]).loc[:, FEATURE_COLUMNS]


def score_row(pipeline, payload: dict, threshold: float) -> dict:
    features = frame_from_user(payload)
    probability = float(positive_probability(pipeline, features)[0])
    return {
        "probability": probability,
        "threshold": float(threshold),
        "predicted_repeat_purchase": int(probability >= threshold),
        "features": features.iloc[0].to_dict(),
        "numeric_features": NUMERIC_FEATURES,
        "sparse_features": SPARSE_NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
