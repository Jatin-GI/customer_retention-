"""Target windows and the ban on using future purchases as features."""

import pandas as pd

from src.feature_engineering import audit_temporal_integrity, features_for_snapshot
from src.target_creation import assign_splits, choose_snapshots, history_cutoff, label_window
from tests.helpers import purchase_line


def _history() -> pd.DataFrame:
    rows = []
    line_id = 1
    for month in range(1, 9):
        rows.append(
            purchase_line(
                line_id,
                f"A{month}",
                "A",
                f"2020-{month:02d}-15 00:00:00",
                quantity=1,
                unit_price=10,
            )
        )
        line_id += 1
    rows.append(purchase_line(line_id, "E1", "E", "2020-06-01 00:00:00", quantity=1, unit_price=50))
    line_id += 1
    rows.append(purchase_line(line_id, "E2", "E", "2020-08-10 00:00:00", quantity=1, unit_price=80))
    line_id += 1
    rows.append(purchase_line(line_id, "B1", "B", "2020-01-15 00:00:00", quantity=1, unit_price=10))
    return pd.DataFrame(rows)


def test_snapshot_dates_leave_a_complete_horizon():
    snapshots = choose_snapshots(
        pd.Timestamp("2010-12-01"),
        pd.Timestamp("2011-12-09 12:50:00"),
        horizon_days=30,
        warmup_days=90,
    )
    assert snapshots[0] == pd.Timestamp("2011-03-01")
    assert snapshots[-1] == pd.Timestamp("2011-11-01")
    assert pd.Timestamp("2011-12-01") not in snapshots
    splits = assign_splits(snapshots)
    assert splits["test"] == [pd.Timestamp("2011-11-01")]
    assert splits["validation"] == [pd.Timestamp("2011-10-01")]
    assert splits["test"][0] not in splits["train"]


def test_label_window_starts_the_day_after_the_snapshot():
    start, end = label_window(pd.Timestamp("2011-11-01"), 30)
    assert start == pd.Timestamp("2011-11-02")
    assert end == pd.Timestamp("2011-12-02")
    assert history_cutoff(pd.Timestamp("2011-11-01")) == start


def test_future_purchases_do_not_change_features_or_leak_into_monetary_value():
    frame = _history()
    features = features_for_snapshot(frame, pd.Timestamp("2020-07-01"), horizon_days=30, min_history_invoices=1)
    features = features.set_index("customer_id")
    # July 15 is inside the label window. August 15 is outside both history and the label window.
    assert features.loc["A", "frequency"] == 6
    assert features.loc["A", "monetary"] == 60
    assert int(features.loc["A", "repeat_purchase"]) == 1
    # August 10 must not create a positive July label and must not enter monetary value.
    assert features.loc["E", "monetary"] == 50
    assert features.loc["E", "frequency"] == 1
    assert int(features.loc["E", "repeat_purchase"]) == 0
    assert features.loc["B", "monetary"] == 10
    assert int(features.loc["B", "repeat_purchase"]) == 0

    leaked = frame.copy()
    leaked.loc[leaked["invoice_no"] == "E2", "unit_price"] = 500
    leaked.loc[leaked["invoice_no"] == "E2", "line_revenue"] = 500
    recomputed = features_for_snapshot(leaked, pd.Timestamp("2020-07-01"), 30, 1).set_index("customer_id")
    assert recomputed.loc["E", "monetary"] == 50
    assert int(recomputed.loc["E", "repeat_purchase"]) == 0
    assert recomputed.loc["A", "monetary"] == features.loc["A", "monetary"]


def test_audit_accepts_a_table_built_by_the_feature_function():
    frame = _history()
    features = features_for_snapshot(frame, pd.Timestamp("2020-07-01"), 30, 1)
    features["snapshot_date"] = pd.Timestamp("2020-07-01")
    audit = audit_temporal_integrity(frame, features, horizon_days=30, sample_size=3, seed=1)
    assert audit["passed"] is True
    assert audit["rows_checked"] == 3
