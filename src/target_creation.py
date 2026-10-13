"""Observation dates, labels, and the temporal train/validation/test split.

A snapshot date S is inclusive history: purchases on S are features.
The label window is the next ``horizon_days`` calendar days, starting the day
after S and ending on S + horizon_days, inclusive.

A snapshot is eligible only when that whole window ends before the last
observed calendar date. The final day of this dataset is a partial day, so the
window must finish at least one day earlier. Customers who simply do not buy
inside a fully observed window are negative examples, not dropped rows.
"""

from __future__ import annotations

import pandas as pd


def choose_snapshots(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    horizon_days: int,
    warmup_days: int,
) -> list[pd.Timestamp]:
    """Month-start snapshots with enough history and a fully observed horizon."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")
    if warmup_days < 0:
        raise ValueError("warmup_days cannot be negative.")
    first_history = pd.Timestamp(min_date).normalize() + pd.Timedelta(days=warmup_days)
    last_eligible = pd.Timestamp(max_date).normalize() - pd.Timedelta(days=horizon_days + 1)
    if first_history > last_eligible:
        return []
    start = pd.offsets.MonthBegin().rollforward(first_history)
    if start > last_eligible:
        return []
    months = pd.date_range(start, last_eligible, freq="MS")
    return [pd.Timestamp(month) for month in months if pd.Timestamp(month) <= last_eligible]


def assign_splits(snapshots: list[pd.Timestamp]) -> dict[str, list[pd.Timestamp]]:
    """Hold out the last snapshot as test and the one before it as validation."""
    if len(snapshots) < 3:
        raise ValueError(
            "Need at least three snapshot dates so training, validation, and test "
            f"do not share a cutoff. Found {len(snapshots)}."
        )
    ordered = sorted(pd.Timestamp(value) for value in snapshots)
    return {
        "train": ordered[:-2],
        "validation": [ordered[-2]],
        "test": [ordered[-1]],
    }


def label_window(snapshot: pd.Timestamp, horizon_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the half-open label window [start, end)."""
    start = pd.Timestamp(snapshot).normalize() + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=horizon_days)
    return start, end


def history_cutoff(snapshot: pd.Timestamp) -> pd.Timestamp:
    """Purchases strictly before this timestamp are historical."""
    return pd.Timestamp(snapshot).normalize() + pd.Timedelta(days=1)


def label_repeat_purchasers(
    invoices: pd.DataFrame,
    customer_ids: pd.Series,
    snapshot: pd.Timestamp,
    horizon_days: int,
) -> pd.Series:
    """1 when the customer has a purchase invoice inside the future window.

    ``invoices`` must contain customer_id and invoice_date. Rows on or before
    the snapshot are ignored, as are rows on or after the end of the horizon.
    """
    start, end = label_window(snapshot, horizon_days)
    future = invoices[(invoices["invoice_date"] >= start) & (invoices["invoice_date"] < end)]
    buyers = set(future["customer_id"].unique())
    return customer_ids.map(lambda customer_id: int(customer_id in buyers)).astype(int)
