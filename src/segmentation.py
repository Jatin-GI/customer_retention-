"""Rule-based RFM segments. These are descriptive groups, not model labels."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import SEGMENT_RULES


def quintile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Map a measure to scores 1 through 5. Score 5 is the favorable end."""
    if series.empty:
        return series.astype(int)
    ranks = series.rank(method="first", ascending=higher_is_better)
    unique_ranks = int(ranks.nunique())
    bins = min(5, unique_ranks)
    if bins < 2:
        return pd.Series(3, index=series.index, dtype=int)
    labels = list(range(1, bins + 1))
    # When fewer than five bins exist, stretch labels so the top bin is still 5.
    if bins < 5:
        labels = [int(round(1 + (index * 4 / (bins - 1)))) for index in range(bins)]
    scored = pd.qcut(ranks, bins, labels=labels)
    return scored.astype(int)


def _in_range(value: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= int(value) <= bounds[1]


def assign_segment(row: pd.Series, rules: list[dict[str, Any]] | None = None) -> str:
    rules = SEGMENT_RULES if rules is None else rules
    for rule in rules:
        matched = True
        for key in ("r_score", "f_score", "m_score"):
            if key in rule and not _in_range(row[key], rule[key]):
                matched = False
                break
        if matched:
            return str(rule["name"])
    return "Regular"


def segment_customers(
    customers: pd.DataFrame,
    rules: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Score recency, frequency, and monetary value, then apply ordered rules.

    ``customers`` needs customer_id, recency_days, frequency, and monetary.
    Lower recency_days is better. Higher frequency and monetary value are better.
    """
    required = {"customer_id", "recency_days", "frequency", "monetary"}
    missing = required.difference(customers.columns)
    if missing:
        raise ValueError(f"Segmentation frame is missing columns: {sorted(missing)}")
    frame = customers.copy()
    frame["r_score"] = quintile_score(frame["recency_days"], higher_is_better=False)
    frame["f_score"] = quintile_score(frame["frequency"], higher_is_better=True)
    frame["m_score"] = quintile_score(frame["monetary"], higher_is_better=True)
    frame["segment"] = frame.apply(lambda row: assign_segment(row, rules), axis=1)
    return frame


def segment_summary(segmented: pd.DataFrame) -> pd.DataFrame:
    """Customer counts and revenue contribution by segment."""
    total_customers = len(segmented)
    total_revenue = float(segmented["monetary"].sum())
    summary = (
        segmented.groupby("segment", as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            purchase_revenue=("monetary", "sum"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .sort_values("purchase_revenue", ascending=False)
    )
    summary["customer_share"] = summary["customers"] / total_customers if total_customers else 0
    summary["revenue_share"] = summary["purchase_revenue"] / total_revenue if total_revenue else 0
    return summary
