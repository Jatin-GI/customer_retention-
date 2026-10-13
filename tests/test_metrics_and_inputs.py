"""Metric math and input checks that do not need the full training run."""

import numpy as np
import pytest

from src.evaluate import metric_row, threshold_for_f1
from src.explainability import original_feature, plain_language
from src.inference import FeatureInputError, frame_from_user
from src.insights import build_insights
import pandas as pd


def test_metric_row_perfect_ranking():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    row = metric_row(labels, scores, threshold=0.5)
    assert row["precision"] == 1
    assert row["recall"] == 1
    assert row["f1"] == 1
    assert row["tp"] == 2 and row["tn"] == 2 and row["fp"] == 0 and row["fn"] == 0
    assert row["roc_auc"] == 1
    assert row["accuracy"] == 1
    assert row["positive_rate"] == 0.5


def test_threshold_for_f1_uses_only_the_supplied_labels():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.45, 0.9])
    threshold = threshold_for_f1(labels, scores)
    row = metric_row(labels, scores, threshold)
    assert row["f1"] >= metric_row(labels, scores, 0.5)["f1"]


def test_original_feature_names():
    assert original_feature("num__recency_days") == "recency_days"
    assert original_feature("sparse__missingindicator_spend_trend_ratio") == "spend_trend_ratio missingness"
    assert original_feature("cat__country_United Kingdom") == "country"


def test_plain_language_quotes_the_computed_importance():
    importance = pd.DataFrame(
        {
            "feature": ["recency_days"],
            "importance_mean": [0.1234],
            "importance_std": [0.01],
        }
    )
    sentences = plain_language(importance, {"recency_days": "Higher values are associated with a lower predicted probability in this SHAP sample."})
    assert "0.1234" in sentences[0]
    assert "not evidence that the feature causes" in sentences[0]


def test_user_input_rejects_negative_recency_and_accepts_a_blank_gap():
    with pytest.raises(FeatureInputError):
        frame_from_user({"recency_days": -1, "country": "United Kingdom"})
    payload = {
        "recency_days": 10,
        "frequency": 2,
        "monetary": 30,
        "avg_order_value": 15,
        "avg_quantity": 2,
        "n_distinct_products": 3,
        "tenure_days": 40,
        "spend_30d": 10,
        "spend_90d": 30,
        "invoices_30d": 1,
        "invoices_90d": 2,
        "avg_unit_price": 5,
        "return_rate": 0,
        "spend_per_tenure_day": 0.5,
        "avg_interpurchase_days": "",
        "spend_trend_ratio": "",
        "country": "United Kingdom",
    }
    frame = frame_from_user(payload)
    assert frame.iloc[0]["country"] == "United Kingdom"
    assert np.isnan(frame.iloc[0]["avg_interpurchase_days"])


def test_insights_use_supplied_statistics_and_label_hypotheses():
    insights = build_insights(
        {
            "repeat_purchase_rate": 0.25,
            "purchase_customers": 100,
            "purchase_revenue": 1234.5,
            "purchase_invoices": 80,
        },
        [
            {
                "segment": "Inactive",
                "customers": 40,
                "purchase_revenue": 100,
                "revenue_share": 0.1,
                "customer_share": 0.4,
            }
        ],
        {"comparison": [], "selected_model": None, "accuracy_note": ""},
        {"median_gap_days": 18.5, "peak_month": None},
    )
    text = " ".join(insights["measured"])
    assert "25.0%" in text
    assert "£1,234.50" in text
    assert "18.5" in text
    assert all("Hypothesis" in item for item in insights["hypotheses"])
