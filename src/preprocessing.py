"""Sklearn preprocessing used inside the saved training pipeline.

The rare-country grouper is fit on training rows only. Joblib loads this class
from ``src.preprocessing``, so inference must be run with the project root on
the Python path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SPARSE_NUMERIC_FEATURES


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Replace infrequent categories with a single other label."""

    def __init__(self, min_count: int = 50, other_label: str = "Other"):
        self.min_count = min_count
        self.other_label = other_label

    def fit(self, X, y=None):  # noqa: N803 - sklearn convention
        series = self._series(X).fillna("Missing").astype(str)
        counts = series.value_counts()
        self.keep_ = set(counts[counts >= self.min_count].index.astype(str))
        return self

    def transform(self, X):  # noqa: N803 - sklearn convention
        series = self._series(X).fillna("Missing").astype(str)
        grouped = series.where(series.isin(self.keep_), self.other_label)
        return grouped.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(["country"])
        return np.asarray(input_features)

    @staticmethod
    def _series(X) -> pd.Series:  # noqa: N803 - sklearn convention
        array = np.asarray(X, dtype=object)
        if array.ndim == 2:
            array = array[:, 0]
        return pd.Series(array)


def build_preprocessor(scale: bool, min_category_count: int = 50) -> ColumnTransformer:
    numeric_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    sparse_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
        sparse_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        [
            ("group", RareCategoryGrouper(min_count=min_category_count)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("sparse", Pipeline(sparse_steps), SPARSE_NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
