"""Customer-level features computed only from transactions at or before a snapshot."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.target_creation import (
    assign_splits,
    choose_snapshots,
    history_cutoff,
    label_repeat_purchasers,
    label_window,
)

logger = logging.getLogger("retention.features")


def _invoice_table(purchase_lines: pd.DataFrame) -> pd.DataFrame:
    return (
        purchase_lines.groupby(["customer_id", "invoice_no"], as_index=False)
        .agg(
            invoice_date=("invoice_date", "min"),
            revenue=("line_revenue", "sum"),
            quantity=("quantity", "sum"),
        )
        .sort_values(["customer_id", "invoice_date", "invoice_no"], kind="mergesort")
    )


def features_for_snapshot(
    transactions: pd.DataFrame,
    snapshot: pd.Timestamp,
    horizon_days: int,
    min_history_invoices: int,
) -> pd.DataFrame:
    """Build one modeling row per eligible customer at ``snapshot``.

    Eligible customers have at least ``min_history_invoices`` purchase invoices
    on or before the snapshot. Features do not use lines on or after the next
    midnight. The label uses purchase invoices in the following horizon only.
    """
    if min_history_invoices < 1:
        raise ValueError("min_history_invoices must be at least 1.")
    frame = transactions.copy()
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    snapshot = pd.Timestamp(snapshot).normalize()
    cutoff = history_cutoff(snapshot)
    window_start, window_end = label_window(snapshot, horizon_days)

    purchase_lines = frame.loc[frame["is_purchase"] == 1].copy()
    hist_lines = purchase_lines.loc[purchase_lines["invoice_date"] < cutoff]
    if hist_lines.empty:
        return pd.DataFrame(columns=["customer_id", TARGET_COLUMN, *FEATURE_COLUMNS])

    invoices = _invoice_table(hist_lines)
    grouped = invoices.groupby("customer_id", sort=False)
    features = grouped.agg(
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
        avg_quantity=("quantity", "mean"),
        total_quantity=("quantity", "sum"),
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
    )
    features = features.loc[features["frequency"] >= min_history_invoices].copy()
    if features.empty:
        return pd.DataFrame(columns=["customer_id", TARGET_COLUMN, *FEATURE_COLUMNS])

    features["avg_unit_price"] = features["monetary"] / features["total_quantity"].replace(0, np.nan)
    features["recency_days"] = (snapshot - features["last_purchase"].dt.normalize()).dt.days.astype(int)
    features["tenure_days"] = (snapshot - features["first_purchase"].dt.normalize()).dt.days.astype(int)
    features["spend_per_tenure_day"] = features["monetary"] / (features["tenure_days"] + 1)

    def _window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        subset = invoices[(invoices["invoice_date"] >= start) & (invoices["invoice_date"] < end)]
        return subset.groupby("customer_id").agg(spend=("revenue", "sum"), invoices=("invoice_no", "nunique"))

    recent_30 = _window(cutoff - pd.Timedelta(days=30), cutoff)
    recent_90 = _window(cutoff - pd.Timedelta(days=90), cutoff)
    prior_90 = _window(cutoff - pd.Timedelta(days=180), cutoff - pd.Timedelta(days=90))
    features["spend_30d"] = recent_30["spend"].reindex(features.index).fillna(0.0)
    features["invoices_30d"] = recent_30["invoices"].reindex(features.index).fillna(0).astype(int)
    features["spend_90d"] = recent_90["spend"].reindex(features.index).fillna(0.0)
    features["invoices_90d"] = recent_90["invoices"].reindex(features.index).fillna(0).astype(int)
    prior_spend = prior_90["spend"].reindex(features.index)
    features["spend_trend_ratio"] = np.where(prior_spend.fillna(0) > 0, features["spend_90d"] / prior_spend, np.nan)

    gaps = invoices.copy()
    gaps["previous_date"] = gaps.groupby("customer_id")["invoice_date"].shift(1)
    gaps["gap_days"] = (gaps["invoice_date"] - gaps["previous_date"]).dt.total_seconds() / 86400.0
    features["avg_interpurchase_days"] = gaps.groupby("customer_id")["gap_days"].mean().reindex(features.index)

    features["n_distinct_products"] = (
        hist_lines.groupby("customer_id")["stock_code"].nunique().reindex(features.index).fillna(0).astype(int)
    )
    latest_country = (
        hist_lines.sort_values(["invoice_date", "invoice_no"], kind="mergesort")
        .groupby("customer_id")["country"]
        .last()
        .reindex(features.index)
    )
    features["country"] = latest_country.fillna("Missing").astype(str)

    return_lines = frame[
        frame["customer_id"].notna()
        & frame["is_exact_duplicate"].eq(0)
        & frame["invoice_date"].lt(cutoff)
        & (frame["is_cancellation"].eq(1) | frame["is_return"].eq(1))
    ]
    purchase_line_counts = hist_lines.groupby("customer_id").size()
    return_line_counts = return_lines.groupby("customer_id").size()
    denominator = purchase_line_counts.add(return_line_counts, fill_value=0).replace(0, np.nan)
    features["return_rate"] = (
        return_line_counts.reindex(features.index).fillna(0) / denominator.reindex(features.index)
    ).fillna(0.0)

    all_invoices = _invoice_table(purchase_lines)
    features = features.reset_index()
    features[TARGET_COLUMN] = label_repeat_purchasers(
        all_invoices,
        features["customer_id"],
        snapshot,
        horizon_days,
    )
    features["snapshot_date"] = snapshot
    features["label_window_start"] = window_start
    features["label_window_end"] = window_end
    features["monetary"] = features["monetary"].astype(float)
    features["avg_order_value"] = features["avg_order_value"].astype(float)
    features["avg_quantity"] = features["avg_quantity"].astype(float)
    return features


def build_modeling_table(
    transactions: pd.DataFrame,
    horizon_days: int,
    min_history_invoices: int,
    warmup_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stack snapshot tables and attach a time-based split."""
    purchases = transactions.loc[transactions["is_purchase"] == 1].copy()
    if purchases.empty:
        raise ValueError(
            "No rows meet the purchase definition, so a repeat-purchase label cannot be built."
        )
    purchases["invoice_date"] = pd.to_datetime(purchases["invoice_date"])
    min_date = purchases["invoice_date"].min()
    max_date = purchases["invoice_date"].max()
    snapshots = choose_snapshots(min_date, max_date, horizon_days, warmup_days)
    if len(snapshots) < 3:
        raise ValueError(
            "The purchase history does not cover three snapshot dates with a fully "
            f"observed {horizon_days}-day horizon after a {warmup_days}-day warmup. "
            f"Observed purchases run from {min_date} to {max_date}."
        )
    splits = assign_splits(snapshots)
    split_of = {date: name for name, dates in splits.items() for date in dates}
    frames = []
    for snapshot in snapshots:
        logger.info("Building features at snapshot %s", snapshot.date())
        snapshot_frame = features_for_snapshot(
            transactions,
            snapshot,
            horizon_days,
            min_history_invoices,
        )
        if snapshot_frame.empty:
            logger.info("Snapshot %s has no eligible customers", snapshot.date())
            continue
        snapshot_frame["split"] = split_of[snapshot]
        frames.append(snapshot_frame)
    if not frames:
        raise ValueError("Feature generation produced no customer rows.")
    table = pd.concat(frames, ignore_index=True)
    meta = {
        "min_purchase_date": str(min_date),
        "max_purchase_date": str(max_date),
        "horizon_days": horizon_days,
        "min_history_invoices": min_history_invoices,
        "warmup_days": warmup_days,
        "snapshots": {name: [str(pd.Timestamp(value).date()) for value in dates] for name, dates in splits.items()},
        "rows": int(len(table)),
        "positive_rate_by_split": {
            name: float(table.loc[table["split"] == name, TARGET_COLUMN].mean())
            for name in ("train", "validation", "test")
            if (table["split"] == name).any()
        },
        "target_definition": (
            f"repeat_purchase = 1 when the customer has at least one purchase invoice "
            f"in the {horizon_days} days after the snapshot date. Features use only "
            "purchases on or before the snapshot date."
        ),
        "feature_columns": FEATURE_COLUMNS,
    }
    logger.info(
        "Modeling table rows=%s positive rates=%s",
        len(table),
        meta["positive_rate_by_split"],
    )
    return table, meta


def audit_temporal_integrity(
    transactions: pd.DataFrame,
    modeling: pd.DataFrame,
    horizon_days: int,
    sample_size: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    """Recompute monetary value and labels for a sample and fail on mismatch."""
    if modeling.empty:
        raise ValueError("Cannot audit an empty modeling table.")
    sample = modeling.sample(n=min(sample_size, len(modeling)), random_state=seed)
    purchases = transactions.loc[transactions["is_purchase"] == 1].copy()
    purchases["invoice_date"] = pd.to_datetime(purchases["invoice_date"])
    invoices = _invoice_table(purchases)
    checked = 0
    for row in sample.itertuples(index=False):
        snapshot = pd.Timestamp(row.snapshot_date).normalize()
        cutoff = history_cutoff(snapshot)
        history = invoices[
            (invoices["customer_id"] == row.customer_id) & (invoices["invoice_date"] < cutoff)
        ]
        monetary = float(history["revenue"].sum())
        frequency = int(history["invoice_no"].nunique())
        if abs(monetary - float(row.monetary)) > 0.05:
            raise AssertionError(
                f"Monetary leakage or mismatch for customer {row.customer_id} at {snapshot.date()}: "
                f"feature={row.monetary} recomputed={monetary}"
            )
        if frequency != int(row.frequency):
            raise AssertionError(
                f"Frequency mismatch for customer {row.customer_id} at {snapshot.date()}: "
                f"feature={row.frequency} recomputed={frequency}"
            )
        expected = int(
            label_repeat_purchasers(
                invoices,
                pd.Series([row.customer_id]),
                snapshot,
                horizon_days,
            ).iloc[0]
        )
        if expected != int(row.repeat_purchase):
            raise AssertionError(
                f"Label mismatch for customer {row.customer_id} at {snapshot.date()}: "
                f"stored={row.repeat_purchase} recomputed={expected}"
            )
        future_in_features = history[history["invoice_date"] >= cutoff]
        if not future_in_features.empty:
            raise AssertionError("History filter included a post-cutoff invoice.")
        checked += 1
    return {"rows_checked": checked, "passed": True}
