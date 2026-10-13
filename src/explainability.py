"""Global and individual explanations for the selected pipeline.

Permutation importance is computed on the original features and is the
plain-language ranking. SHAP is added when the installed library can explain
the selected estimator. High importance is a predictive association, not a
causal effect.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import FEATURE_COLUMNS, NUMERIC_FEATURES, SPARSE_NUMERIC_FEATURES

logger = logging.getLogger("retention.explain")

FEATURE_TEXT = {
    "recency_days": "days since the customer's last purchase, measured at the cutoff",
    "frequency": "count of historical purchase invoices",
    "monetary": "historical purchase revenue before the cutoff",
    "avg_order_value": "average revenue of historical purchase invoices",
    "avg_quantity": "average quantity on historical purchase invoices",
    "n_distinct_products": "count of distinct merchandise codes purchased",
    "tenure_days": "days since the first purchase invoice",
    "spend_30d": "purchase revenue in the 30 days ending on the cutoff",
    "spend_90d": "purchase revenue in the 90 days ending on the cutoff",
    "invoices_30d": "purchase invoices in the 30 days ending on the cutoff",
    "invoices_90d": "purchase invoices in the 90 days ending on the cutoff",
    "avg_unit_price": "historical revenue divided by quantity",
    "return_rate": "share of historical lines that are cancellations or returns",
    "spend_per_tenure_day": "historical revenue divided by tenure plus one day",
    "avg_interpurchase_days": "average days between purchase invoices",
    "spend_trend_ratio": "recent 90-day revenue divided by the previous 90 days",
    "country": "country on the most recent historical invoice, with rare countries grouped",
}


def original_feature(transformed_name: str) -> str:
    """Map a transformed column name back to a modeling feature."""
    rest = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    if rest.startswith("missingindicator_"):
        return rest.replace("missingindicator_", "") + " missingness"
    if rest.startswith("country_"):
        return "country"
    return rest


def _transformed_matrix(pipeline, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    prep = pipeline.named_steps["prep"]
    matrix = prep.transform(features)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    names = [str(name) for name in prep.get_feature_names_out()]
    return np.asarray(matrix), names


def _positive_shap_values(raw) -> np.ndarray:
    if hasattr(raw, "values") and not isinstance(raw, np.ndarray):
        raw = raw.values
    if isinstance(raw, list):
        return np.asarray(raw[1 if len(raw) > 1 else 0])
    array = np.asarray(raw)
    if array.ndim == 3:
        return array[:, :, 1]
    return array


def aggregate_shap(shap_matrix: np.ndarray, transformed_names: list[str]) -> pd.DataFrame:
    """Sum SHAP values that belong to the same original feature."""
    buckets: dict[str, np.ndarray] = {}
    for index, name in enumerate(transformed_names):
        key = original_feature(name)
        column = shap_matrix[:, index]
        buckets[key] = column if key not in buckets else buckets[key] + column
    return pd.DataFrame(buckets)


def _direction_from_shap(features: pd.DataFrame, aggregated: pd.DataFrame) -> dict[str, str]:
    directions = {}
    for column in aggregated.columns:
        if column not in features.columns:
            directions[column] = (
                "This grouped effect combines several encoded columns, so a single direction is not reported."
            )
            continue
        values = pd.to_numeric(features[column], errors="coerce")
        paired = pd.DataFrame({"x": values, "s": aggregated[column]}).dropna()
        if len(paired) < 20 or paired["x"].nunique() < 2:
            directions[column] = "There was not enough variation in the SHAP sample to describe a direction."
            continue
        correlation = float(np.corrcoef(paired["x"], paired["s"])[0, 1])
        if correlation > 0.05:
            directions[column] = (
                "In this SHAP sample, higher values are associated with a higher predicted probability."
            )
        elif correlation < -0.05:
            directions[column] = (
                "In this SHAP sample, higher values are associated with a lower predicted probability."
            )
        else:
            directions[column] = "In this SHAP sample, the association does not run consistently in one direction."
    return directions


def plain_language(importance: pd.DataFrame, directions: dict[str, str], limit: int = 5) -> list[str]:
    """Sentences tied to computed permutation scores. They do not claim causation."""
    lines = []
    ordered = importance.sort_values("importance_mean", ascending=False).head(limit)
    for row in ordered.itertuples(index=False):
        description = FEATURE_TEXT.get(row.feature, row.feature)
        direction = directions.get(
            row.feature,
            "A direction was not estimated for this feature.",
        )
        lines.append(
            f"{row.feature} ({description}). Permuting it changed validation average precision "
            f"by {row.importance_mean:.4f} on average "
            f"(standard deviation {row.importance_std:.4f} across repeats). {direction} "
            "This is a predictive association in the fitted model, not evidence that the feature "
            "causes a repeat purchase."
        )
    return lines


def _shap_matrix(pipeline, features: pd.DataFrame) -> tuple[np.ndarray | None, list[str], str | None]:
    model = pipeline.named_steps["model"]
    try:
        matrix, names = _transformed_matrix(pipeline, features)
    except Exception as exc:  # noqa: BLE001 - explanation should degrade, not stop the pipeline
        return None, [], f"The preprocessor could not name transformed columns: {exc}"
    try:
        import shap
    except ImportError:
        return None, names, "The shap package is not installed. Permutation importance is the fallback."
    try:
        if isinstance(model, (XGBClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(model)
            values = _positive_shap_values(explainer.shap_values(matrix))
            return values, names, None
        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, matrix)
            values = _positive_shap_values(explainer.shap_values(matrix))
            return values, names, None
    except Exception as exc:  # noqa: BLE001 - keep a documented fallback
        logger.warning("SHAP failed: %s", exc)
        return None, names, f"SHAP could not explain this estimator ({exc}). Permutation importance is the fallback."
    return None, names, "SHAP is not used for this estimator. Permutation importance is the fallback."


def explain_pipeline(
    pipeline,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    sample_size: int = 400,
) -> dict[str, Any]:
    """Permutation importance, optional SHAP, and example-customer contributions."""
    from sklearn.dummy import DummyClassifier

    model = pipeline.named_steps["model"]
    result: dict[str, Any] = {
        "method": [],
        "limitation": (
            "Feature attributions describe the fitted model. They do not show that changing a "
            "feature would cause a customer to buy again."
        ),
    }
    if isinstance(model, DummyClassifier):
        result["method"].append("none")
        result["shap_note"] = "The dummy prior model ignores features, so it has no feature attributions."
        result["importance"] = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        result["sentences"] = [
            "The selected model is the dummy prior baseline. It predicts the training base rate "
            "for every customer and does not rank behavioral features."
        ]
        result["examples"] = []
        result["shap_summary"] = pd.DataFrame(columns=["feature", "mean_abs_shap"])
        return result

    valid = validation.sample(n=min(sample_size, len(validation)), random_state=seed)
    try:
        permutation = permutation_importance(
            pipeline,
            valid[FEATURE_COLUMNS],
            valid["repeat_purchase"].astype(int),
            n_repeats=5,
            random_state=seed,
            scoring="average_precision",
            n_jobs=1,
        )
        importance = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance_mean": permutation.importances_mean,
                "importance_std": permutation.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        result["method"].append("permutation_importance")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Permutation importance failed: %s", exc)
        importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_mean": np.nan, "importance_std": np.nan})
        result["shap_note"] = f"Permutation importance failed: {exc}"
    result["importance"] = importance

    explain_rows = test.sample(n=min(sample_size, len(test)), random_state=seed)
    shap_values, names, shap_note = _shap_matrix(pipeline, explain_rows[FEATURE_COLUMNS])
    result["shap_note"] = shap_note
    directions: dict[str, str] = {}
    if shap_values is not None:
        result["method"].append("shap")
        aggregated = aggregate_shap(shap_values, names)
        summary = (
            aggregated.abs()
            .mean()
            .rename("mean_abs_shap")
            .reset_index()
            .rename(columns={"index": "feature"})
            .sort_values("mean_abs_shap", ascending=False)
        )
        result["shap_summary"] = summary
        directions = _direction_from_shap(explain_rows.reset_index(drop=True), aggregated.reset_index(drop=True))
        result["examples"] = _example_contributions(
            explain_rows.reset_index(drop=True),
            aggregated.reset_index(drop=True),
            pipeline,
        )
    else:
        result["shap_summary"] = pd.DataFrame(columns=["feature", "mean_abs_shap"])
        result["examples"] = []
        if isinstance(model, LogisticRegression) and hasattr(model, "coef_"):
            result["method"].append("logistic_coefficients")
            try:
                _, names = _transformed_matrix(pipeline, explain_rows[FEATURE_COLUMNS])
                coefs = pd.DataFrame({"transformed_feature": names, "coefficient": model.coef_.ravel()})
                coefs["feature"] = coefs["transformed_feature"].map(original_feature)
                result["coefficients"] = coefs
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read logistic coefficients: %s", exc)
    result["sentences"] = plain_language(importance.dropna(subset=["importance_mean"]), directions)
    result["directions"] = directions
    return result


def _example_contributions(rows: pd.DataFrame, aggregated: pd.DataFrame, pipeline, limit: int = 6) -> list[dict]:
    from src.evaluate import positive_probability

    probabilities = positive_probability(pipeline, rows[FEATURE_COLUMNS])
    order = np.linspace(0, len(rows) - 1, min(limit, len(rows))).astype(int)
    examples = []
    for position in order:
        contributions = []
        for feature, shap_value in aggregated.iloc[position].items():
            if feature not in rows.columns:
                continue
            value = rows.iloc[position][feature]
            contributions.append(
                {
                    "feature": feature,
                    "value": None if pd.isna(value) else value,
                    "shap": float(shap_value),
                }
            )
        contributions.sort(key=lambda item: abs(item["shap"]), reverse=True)
        examples.append(
            {
                "customer_id": str(rows.iloc[position]["customer_id"]),
                "snapshot_date": str(pd.Timestamp(rows.iloc[position]["snapshot_date"]).date()),
                "probability": float(probabilities[position]),
                "actual_repeat_purchase": int(rows.iloc[position]["repeat_purchase"]),
                "contributions": contributions[:5],
                "features": _json_features(rows.iloc[position]),
            }
        )
    return examples


def _json_features(row: pd.Series) -> dict[str, Any]:
    payload = {}
    for column in FEATURE_COLUMNS:
        value = row[column]
        if pd.isna(value):
            payload[column] = None
        elif column in NUMERIC_FEATURES or column in SPARSE_NUMERIC_FEATURES:
            payload[column] = float(value)
        else:
            payload[column] = str(value)
    return payload


def contributions_for_row(pipeline, row: pd.DataFrame) -> list[dict[str, Any]]:
    """Explain one customer. Returns an empty list when SHAP is unavailable."""
    values, names, note = _shap_matrix(pipeline, row[FEATURE_COLUMNS])
    if values is None:
        return []
    aggregated = aggregate_shap(values, names)
    items = []
    for feature, shap_value in aggregated.iloc[0].items():
        if feature not in row.columns:
            continue
        raw_value = row.iloc[0][feature]
        items.append(
            {
                "feature": feature,
                "value": None if pd.isna(raw_value) else raw_value,
                "shap": float(shap_value),
                "note": note,
            }
        )
    items.sort(key=lambda item: abs(item["shap"]), reverse=True)
    return items[:5]
