"""Training loop, temporal separation, and inference on a tiny labeled table."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.inference import score_row
from src.train import model_catalog, train_models


def _modeling_frame() -> pd.DataFrame:
    rows = []
    plan = {
        "train": ["2021-01-01", "2021-02-01"],
        "validation": ["2021-03-01"],
        "test": ["2021-04-01"],
    }
    for split, dates in plan.items():
        for date in dates:
            for index in range(24):
                label = int(index % 2 == 0)
                rows.append(
                    {
                        "customer_id": f"{split[0]}{date[-2:]}{index}",
                        "snapshot_date": date,
                        "split": split,
                        TARGET_COLUMN: label,
                        "recency_days": 40 if label == 0 else 5,
                        "frequency": 1 if label == 0 else 4,
                        "monetary": 20.0 if label == 0 else 80.0,
                        "avg_order_value": 20.0 if label == 0 else 20.0,
                        "avg_quantity": 1.0,
                        "n_distinct_products": 2 + label,
                        "tenure_days": 50,
                        "spend_30d": 0.0 if label == 0 else 20.0,
                        "spend_90d": 20.0 if label == 0 else 80.0,
                        "invoices_30d": 0 if label == 0 else 1,
                        "invoices_90d": 1 if label == 0 else 3,
                        "avg_unit_price": 10.0,
                        "return_rate": 0.0,
                        "spend_per_tenure_day": 0.4,
                        "avg_interpurchase_days": np.nan if label == 0 else 15.0,
                        "spend_trend_ratio": np.nan if label == 0 else 1.5,
                        "country": "United Kingdom" if index % 4 else "Germany",
                    }
                )
    return pd.DataFrame(rows)


def test_catalog_includes_the_required_models():
    names = [spec["name"] for spec in model_catalog(42)]
    assert names == ["dummy_prior", "logistic_regression", "random_forest", "xgboost"]
    assert "customer_id" not in FEATURE_COLUMNS
    assert "invoice_no" not in FEATURE_COLUMNS


def test_training_keeps_the_test_snapshot_out_of_selection(monkeypatch):
    def tiny_catalog(seed: int):
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
                "estimator": LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed),
                "grid": [{"C": 1.0}],
            },
        ]

    monkeypatch.setattr("src.train.model_catalog", tiny_catalog)
    trained = train_models(_modeling_frame(), seed=42)
    assert trained["selected_model"] in {"dummy_prior", "logistic_regression"}
    assert trained["selection_metric"] == "validation_pr_auc"
    test_rows = trained["test_scores"]
    assert set(pd.to_datetime(test_rows["snapshot_date"]).dt.strftime("%Y-%m-%d")) == {"2021-04-01"}
    assert "2021-04-01" not in set(pd.to_datetime(_modeling_frame().loc[lambda df: df["split"] == "train", "snapshot_date"]).dt.strftime("%Y-%m-%d"))
    scored = score_row(
        trained["selected_pipeline"],
        test_rows.iloc[0][FEATURE_COLUMNS].to_dict(),
        threshold=trained["threshold"],
    )
    assert 0 <= scored["probability"] <= 1
    logistic = next(row for row in trained["comparison"] if row["model"] == "logistic_regression" and row["split"] == "test")
    dummy = next(row for row in trained["comparison"] if row["model"] == "dummy_prior" and row["split"] == "test")
    assert logistic["pr_auc"] is not None
    assert dummy["pr_auc"] is not None
    assert dummy["accuracy"] >= dummy["positive_rate"] - 1e-9
