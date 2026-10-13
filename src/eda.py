"""Exploratory plots and written interpretations driven by computed statistics."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.artifacts import write_json


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def _invoice_table(transactions: pd.DataFrame) -> pd.DataFrame:
    purchases = transactions.loc[transactions["is_purchase"] == 1].copy()
    purchases["invoice_date"] = pd.to_datetime(purchases["invoice_date"])
    return (
        purchases.groupby(["customer_id", "invoice_no"], as_index=False)
        .agg(invoice_date=("invoice_date", "min"), revenue=("line_revenue", "sum"))
        .sort_values(["customer_id", "invoice_date"])
    )


def compute_stats(
    transactions: pd.DataFrame,
    modeling: pd.DataFrame,
    quality: dict,
    segments: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> dict:
    """Numbers that the written interpretations are allowed to cite."""
    invoices = _invoice_table(transactions)
    invoices["year_month"] = invoices["invoice_date"].dt.strftime("%Y-%m")
    monthly = invoices.groupby("year_month", as_index=False)["revenue"].sum()
    peak = monthly.loc[monthly["revenue"].idxmax()]
    low = monthly.loc[monthly["revenue"].idxmin()]
    max_date = pd.to_datetime(transactions["invoice_date"]).max()
    last_month = max_date.strftime("%Y-%m")
    month_end = (max_date + pd.offsets.MonthEnd(0)).normalize()
    counts = invoices.groupby("customer_id").size()
    one_time = counts[counts == 1].index
    repeat = counts[counts >= 2].index
    revenue_by_customer = invoices.groupby("customer_id")["revenue"].sum()
    total_revenue = float(revenue_by_customer.sum())
    gaps = invoices.copy()
    gaps["previous"] = gaps.groupby("customer_id")["invoice_date"].shift(1)
    gaps["gap_days"] = (gaps["invoice_date"] - gaps["previous"]).dt.total_seconds() / 86400.0
    gap_means = gaps.groupby("customer_id")["gap_days"].mean().dropna()
    order_values = invoices["revenue"]
    month1 = cohorts[cohorts["month_offset"] == 1] if not cohorts.empty and "month_offset" in cohorts.columns else cohorts
    test = modeling[modeling["split"] == "test"] if "split" in modeling.columns else modeling.iloc[0:0]
    train = modeling[modeling["split"] == "train"] if "split" in modeling.columns else modeling.iloc[0:0]
    if not test.empty:
        pos_recency = float(test.loc[test["repeat_purchase"] == 1, "recency_days"].median())
        neg_recency = float(test.loc[test["repeat_purchase"] == 0, "recency_days"].median())
        test_rate = float(test["repeat_purchase"].mean())
    else:
        pos_recency = None
        neg_recency = None
        test_rate = None
    if segments.empty:
        largest = None
        richest = None
    else:
        largest = segments.sort_values("customers", ascending=False).iloc[0]
        richest = segments.sort_values("purchase_revenue", ascending=False).iloc[0]
    return {
        "source_rows": int(quality.get("source_rows", len(transactions))),
        "rows_retained": int(quality.get("rows_retained", len(transactions))),
        "missing_customer_rows": int(quality.get("missing_customer_rows", 0)),
        "exact_duplicate_rows": int(quality.get("exact_duplicate_rows", 0)),
        "cancellation_rows": int(quality.get("cancellation_rows", 0)),
        "purchase_rows": int(quality.get("purchase_rows", int((transactions["is_purchase"] == 1).sum()))),
        "purchase_revenue": float(quality.get("purchase_revenue", total_revenue)),
        "purchase_customers": int(quality.get("purchase_customers", invoices["customer_id"].nunique())),
        "min_invoice_date": str(quality.get("min_invoice_date", transactions["invoice_date"].min())),
        "max_invoice_date": str(quality.get("max_invoice_date", max_date)),
        "peak_month": str(peak["year_month"]),
        "peak_revenue": float(peak["revenue"]),
        "low_month": str(low["year_month"]),
        "low_revenue": float(low["revenue"]),
        "last_month": last_month,
        "last_month_is_partial": bool(max_date.normalize() < month_end),
        "median_order_value": float(order_values.median()),
        "p95_order_value": float(order_values.quantile(0.95)),
        "one_time_customers": int(len(one_time)),
        "repeat_customers": int(len(repeat)),
        "repeat_purchase_rate": float(len(repeat) / len(counts)) if len(counts) else None,
        "one_time_revenue_share": float(revenue_by_customer.loc[one_time].sum() / total_revenue) if total_revenue else None,
        "repeat_revenue_share": float(revenue_by_customer.loc[repeat].sum() / total_revenue) if total_revenue else None,
        "median_gap_days": None if gap_means.empty else float(gap_means.median()),
        "gap_customers": int(gap_means.shape[0]),
        "largest_segment": None if largest is None else str(largest["segment"]),
        "largest_segment_customers": None if largest is None else int(largest["customers"]),
        "largest_segment_customer_share": None if largest is None else float(largest["customer_share"]),
        "highest_revenue_segment": None if richest is None else str(richest["segment"]),
        "highest_revenue_share": None if richest is None else float(richest["revenue_share"]),
        "mean_month1_retention": None if month1.empty else float(month1["retention_rate"].mean()),
        "cohorts_with_month1": int(len(month1)),
        "test_positive_rate": test_rate,
        "train_positive_rate": None if train.empty else float(train["repeat_purchase"].mean()),
        "test_positive_median_recency": pos_recency,
        "test_negative_median_recency": neg_recency,
        "monthly": monthly,
        "invoices": invoices,
        "counts": counts,
        "gap_means": gap_means,
    }


def interpretations(stats: dict) -> list[dict]:
    """Four-part notes. Every figure uses only values present in ``stats``."""
    missing_share = stats["missing_customer_rows"] / stats["source_rows"] if stats["source_rows"] else 0
    notes = [
        {
            "chart": "Data quality flags",
            "shows": "How many source rows were retained, and how many were flagged as missing a customer, cancelled, or duplicated.",
            "pattern": (
                f"The file has {stats['source_rows']:,} source rows and {stats['rows_retained']:,} retained rows. "
                f"{stats['missing_customer_rows']:,} retained rows ({missing_share:.1%}) have no customer ID. "
                f"{stats['cancellation_rows']:,} rows are cancellations and {stats['exact_duplicate_rows']:,} are exact duplicate copies."
            ),
            "why": "Customer-level revenue and the repeat-purchase label cannot use anonymous or duplicate lines without inventing an identity or double-counting an order.",
            "further": "A merchant would still need to learn whether guest checkouts can be matched back to accounts before treating the excluded revenue as lost demand.",
        },
        {
            "chart": "Monthly purchase revenue",
            "shows": "Merchandise purchase revenue by calendar month under the documented purchase definition.",
            "pattern": (
                f"The highest month is {stats['peak_month']} at £{stats['peak_revenue']:,.2f}. "
                f"The lowest month is {stats['low_month']} at £{stats['low_revenue']:,.2f}. "
                + (
                    f"The latest invoice is {stats['max_invoice_date']}, so {stats['last_month']} does not contain a full month."
                    if stats["last_month_is_partial"]
                    else f"The latest invoice is {stats['max_invoice_date']}."
                )
            ),
            "why": "A single peak can be seasonality, wholesale timing, or a partial month. The series is the denominator for any claim about growth.",
            "further": "More than one year of invoices would be needed before treating the peak month as a recurring seasonal effect.",
        },
        {
            "chart": "Order-value distribution",
            "shows": "The distribution of purchase-invoice revenue.",
            "pattern": (
                f"The median purchase invoice is £{stats['median_order_value']:,.2f} and the 95th percentile is "
                f"£{stats['p95_order_value']:,.2f}."
                + (
                    " The upper tail is more than five times the median, so a small number of invoices can move total revenue."
                    if stats["median_order_value"] and stats["p95_order_value"] > 5 * stats["median_order_value"]
                    else " The upper tail is not more than five times the median in this cut of the data."
                )
            ),
            "why": "Means and totals will follow the tail. A retention metric based on revenue is not the same as one based on customer counts.",
            "further": "Separating wholesale and consumer accounts, which this file does not label, would show whether the tail is a few business buyers.",
        },
        {
            "chart": "One-time and repeat customers",
            "shows": "Customers with one purchase invoice versus customers with two or more, and the revenue each group contributes.",
            "pattern": (
                f"{stats['one_time_customers']:,} customers have one purchase invoice and {stats['repeat_customers']:,} have more than one. "
                f"The repeat-purchase rate is {stats['repeat_purchase_rate']:.1%}. "
                f"Repeat customers contribute {stats['repeat_revenue_share']:.1%} of purchase revenue, "
                f"while one-time customers contribute {stats['one_time_revenue_share']:.1%}."
            ),
            "why": "A high repeat rate means the prediction problem is not the same as finding a rare second purchase, and revenue concentration changes who is costly to lose.",
            "further": "This rate uses the whole history. It is not the 30-day future label used by the model.",
        },
        {
            "chart": "Time between purchases",
            "shows": "The average gap between consecutive purchase invoices, for customers who have at least two.",
            "pattern": (
                f"{stats['gap_customers']:,} customers have a measurable gap. "
                f"The median of those customer-level average gaps is {stats['median_gap_days']:.1f} days."
                if stats["median_gap_days"] is not None
                else "No customer has two purchase invoices, so an inter-purchase gap cannot be computed."
            ),
            "why": "The gap is the descriptive context for a 30-day prediction horizon. A horizon much shorter than the typical gap will have a low positive rate.",
            "further": "Gaps can mix consumers and wholesale buyers. A business decision about reminder timing would need that split, which this dataset does not provide.",
        },
    ]
    if stats["largest_segment"] is not None:
        notes.append(
            {
                "chart": "RFM segments",
                "shows": "Rule-based segments from recency, frequency, and monetary quintiles on the full observed purchase history.",
                "pattern": (
                    f"The largest segment by customers is {stats['largest_segment']} "
                    f"({stats['largest_segment_customers']:,} customers, {stats['largest_segment_customer_share']:.1%} of the profiled base). "
                    f"The largest segment by revenue is {stats['highest_revenue_segment']} "
                    f"at {stats['highest_revenue_share']:.1%} of profiled revenue."
                ),
                "why": "Segment shares show where customers and revenue sit. They are not proof that a message sent to that segment would change behavior.",
                "further": "The SQL NTILE segments can place boundary customers differently from these pandas quintiles. Compare both before using either as an operational list.",
            }
        )
    if stats["mean_month1_retention"] is not None:
        notes.append(
            {
                "chart": "Monthly cohorts",
                "shows": "For each acquisition month, the share of customers who purchase again in a later month offset.",
                "pattern": (
                    f"Across {stats['cohorts_with_month1']} cohorts that have a month-1 observation, "
                    f"the mean month-1 retention rate is {stats['mean_month1_retention']:.1%}."
                ),
                "why": "Cohort retention separates customers who were acquired recently from customers who have had more time to return.",
                "further": "Later offsets have fewer cohorts, and the last calendar month is partial, so the right edge of the heatmap is thinner than the left.",
            }
        )
    if stats["test_positive_rate"] is not None:
        recency_sentence = (
            f"On the test snapshot, the median recency is {stats['test_positive_median_recency']:.0f} days "
            f"for customers who do purchase in the next horizon and {stats['test_negative_median_recency']:.0f} days "
            f"for customers who do not."
        )
        if stats["test_positive_median_recency"] < stats["test_negative_median_recency"]:
            recency_sentence += " Buyers in the horizon are more recent at the cutoff than non-buyers."
        elif stats["test_positive_median_recency"] > stats["test_negative_median_recency"]:
            recency_sentence += " Buyers in the horizon are less recent at the cutoff than non-buyers."
        else:
            recency_sentence += " The two medians are equal in this snapshot."
        notes.append(
            {
                "chart": "Features and the future label",
                "shows": "Recency at the test cutoff, split by whether the customer purchases in the following horizon.",
                "pattern": (
                    f"The test positive rate is {stats['test_positive_rate']:.1%}. "
                    f"The training positive rate is {stats['train_positive_rate']:.1%}. {recency_sentence}"
                ),
                "why": "A difference in recency is an association with the label. It can also dominate a model and hide other useful signals. Accuracy will look high if the positive rate is far from one half, even for a constant prediction.",
                "further": "Check the held-out precision and recall before using a low score as a contact list. Recency is available at the cutoff; using a later purchase to build it would leak the label.",
            }
        )
    return notes


def run_eda(
    transactions: pd.DataFrame,
    modeling: pd.DataFrame,
    quality: dict,
    segments: pd.DataFrame,
    cohorts: pd.DataFrame,
    figures_dir: Path,
) -> dict:
    """Create figures and a summary JSON. Interpretations are generated from the summary."""
    _style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_stats(transactions, modeling, quality, segments, cohorts)
    plots = []

    quality_counts = pd.Series(
        {
            "Missing customer": stats["missing_customer_rows"],
            "Cancellations": stats["cancellation_rows"],
            "Exact duplicates": stats["exact_duplicate_rows"],
            "Purchase lines": stats["purchase_rows"],
        }
    )
    fig, axis = plt.subplots(figsize=(8, 4.2))
    quality_counts.plot(kind="bar", ax=axis, color=["#A8A29E", "#C46B3A", "#78716C", "#0F6E6E"])
    axis.set_title("Retained rows by data-quality flag")
    axis.set_ylabel("Rows")
    axis.tick_params(axis="x", rotation=0)
    plots.append(_save(fig, figures_dir / "01_data_quality.png"))

    monthly = stats["monthly"]
    fig, axis = plt.subplots(figsize=(9, 4.2))
    axis.plot(monthly["year_month"], monthly["revenue"], marker="o", color="#0F6E6E")
    axis.set_title("Monthly merchandise purchase revenue")
    axis.set_xlabel("Month")
    axis.set_ylabel("Revenue (GBP)")
    axis.tick_params(axis="x", rotation=45)
    plots.append(_save(fig, figures_dir / "02_monthly_revenue.png"))

    fig, axis = plt.subplots(figsize=(8, 4.2))
    positive_orders = stats["invoices"].loc[stats["invoices"]["revenue"] > 0, "revenue"]
    axis.hist(np.log10(positive_orders), bins=40, color="#0F6E6E", edgecolor="white")
    axis.set_title("Purchase-invoice revenue, log10 scale")
    axis.set_xlabel("log10(invoice revenue)")
    axis.set_ylabel("Invoices")
    plots.append(_save(fig, figures_dir / "03_order_value.png"))

    fig, axis = plt.subplots(figsize=(8, 4.2))
    clipped = stats["counts"].clip(upper=20)
    axis.hist(clipped, bins=np.arange(0.5, 21.5, 1), color="#0F6E6E", edgecolor="white")
    axis.set_title("Purchase invoices per customer (20 or more shown as 20)")
    axis.set_xlabel("Purchase invoices")
    axis.set_ylabel("Customers")
    plots.append(_save(fig, figures_dir / "04_frequency.png"))

    if not stats["gap_means"].empty:
        fig, axis = plt.subplots(figsize=(8, 4.2))
        axis.hist(stats["gap_means"].clip(upper=180), bins=30, color="#C46B3A", edgecolor="white")
        axis.set_title("Average days between purchases (capped at 180 for display)")
        axis.set_xlabel("Average gap in days")
        axis.set_ylabel("Customers")
        plots.append(_save(fig, figures_dir / "05_interpurchase_gap.png"))

    fig, axis = plt.subplots(figsize=(7, 4.2))
    share = pd.Series(
        {
            "One-time customers": stats["one_time_revenue_share"],
            "Repeat customers": stats["repeat_revenue_share"],
        }
    )
    share.plot(kind="bar", ax=axis, color=["#A8A29E", "#0F6E6E"])
    axis.set_ylim(0, 1)
    axis.set_title("Share of purchase revenue")
    axis.set_ylabel("Revenue share")
    axis.tick_params(axis="x", rotation=0)
    plots.append(_save(fig, figures_dir / "06_repeat_revenue_share.png"))

    if not segments.empty:
        fig, axis = plt.subplots(figsize=(9, 4.4))
        ordered = segments.sort_values("purchase_revenue", ascending=False)
        axis.bar(ordered["segment"], ordered["purchase_revenue"], color="#0F6E6E")
        axis.set_title("Purchase revenue by RFM segment")
        axis.set_ylabel("Revenue (GBP)")
        axis.tick_params(axis="x", rotation=30)
        plots.append(_save(fig, figures_dir / "07_segment_revenue.png"))

    if not cohorts.empty:
        pivot = cohorts.pivot_table(index="cohort_month", columns="month_offset", values="retention_rate")
        fig, axis = plt.subplots(figsize=(10, 5.5))
        sns.heatmap(pivot, cmap="YlGnBu", vmin=0, vmax=1, ax=axis)
        axis.set_title("Cohort retention by month offset")
        axis.set_xlabel("Months since first purchase")
        axis.set_ylabel("Cohort")
        plots.append(_save(fig, figures_dir / "08_cohort_retention.png"))

    test = modeling[modeling["split"] == "test"] if "split" in modeling.columns else modeling.iloc[0:0]
    if not test.empty and test["repeat_purchase"].nunique() > 1:
        fig, axis = plt.subplots(figsize=(7, 4.2))
        plot_frame = test.copy()
        plot_frame["label"] = np.where(plot_frame["repeat_purchase"] == 1, "Purchased in horizon", "Did not purchase")
        sns.boxplot(data=plot_frame, x="label", y="recency_days", ax=axis, color="#8FB9B9")
        axis.set_title("Test-snapshot recency by future purchase")
        axis.set_xlabel("")
        axis.set_ylabel("Days since last purchase")
        plots.append(_save(fig, figures_dir / "09_recency_by_label.png"))

        balance = (
            modeling.groupby(["snapshot_date", "split"], as_index=False)["repeat_purchase"].mean()
        )
        balance["snapshot_date"] = pd.to_datetime(balance["snapshot_date"]).dt.strftime("%Y-%m-%d")
        fig, axis = plt.subplots(figsize=(9, 4.2))
        sns.barplot(data=balance, x="snapshot_date", y="repeat_purchase", hue="split", ax=axis, palette="Set2")
        axis.set_title("Positive rate by snapshot")
        axis.set_ylabel("Share who purchase in the next horizon")
        axis.set_xlabel("Snapshot date")
        axis.tick_params(axis="x", rotation=45)
        plots.append(_save(fig, figures_dir / "10_class_balance.png"))

    serializable = {key: value for key, value in stats.items() if key not in {"monthly", "invoices", "counts", "gap_means"}}
    serializable["plots"] = plots
    serializable["interpretations"] = interpretations(serializable)
    write_json(figures_dir.parent / "eda_summary.json", serializable)
    return serializable
