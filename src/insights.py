"""Business-facing text built only from calculated tables.

Proposed actions are labeled as hypotheses. Nothing here claims that a campaign
caused retention, and nothing here is a finding about a company other than the
public retail dataset that was actually loaded.
"""

from __future__ import annotations


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _money(value: float) -> str:
    return f"£{value:,.2f}"


def build_insights(kpis: dict, segments: list[dict], metrics: dict, eda: dict) -> dict:
    """Assemble measured statements and separate experiment hypotheses."""
    repeat_rate = float(kpis["repeat_purchase_rate"])
    customers = int(kpis["purchase_customers"])
    revenue = float(kpis["purchase_revenue"])
    invoices = int(kpis["purchase_invoices"])
    measured = [
        (
            f"The purchase definition covers {customers:,} customers, {invoices:,} purchase invoices, "
            f"and {_money(revenue)} of merchandise revenue."
        ),
        (
            f"The repeat-purchase rate is {_pct(repeat_rate)}: that is the share of purchasing customers "
            "with at least two purchase invoices anywhere in the observed history."
        ),
    ]
    if eda.get("median_gap_days") is not None:
        measured.append(
            f"Among customers with at least two purchase invoices, the median average gap between "
            f"invoices is {eda['median_gap_days']:.1f} days."
        )
    if eda.get("peak_month"):
        partial = ""
        if eda.get("last_month_is_partial"):
            partial = (
                f" The latest invoice date is {eda['max_invoice_date']}, so {eda['last_month']} "
                "is an incomplete month."
            )
        measured.append(
            f"The highest observed purchase-revenue month is {eda['peak_month']} "
            f"at {_money(float(eda['peak_revenue']))}. The lowest is {eda['low_month']} "
            f"at {_money(float(eda['low_revenue']))}.{partial}"
        )
    if segments:
        top_revenue = max(segments, key=lambda row: float(row["purchase_revenue"]))
        measured.append(
            f"Under the Python RFM rules, the {top_revenue['segment']} segment contributes "
            f"{_pct(float(top_revenue['revenue_share']))} of profiled purchase revenue and "
            f"{_pct(float(top_revenue['customer_share']))} of profiled customers. "
            "These segments are rules, not ground-truth labels."
        )
    test_metrics = {}
    for row in metrics.get("comparison", []):
        if row.get("model") == metrics.get("selected_model") and row.get("split") == "test":
            test_metrics = row
            break
    if test_metrics:
        measured.append(
            f"The selected model is {metrics['selected_model']}. On the held-out test snapshot its "
            f"PR-AUC is {_fmt_metric(test_metrics.get('pr_auc'))}, ROC-AUC is "
            f"{_fmt_metric(test_metrics.get('roc_auc'))}, and F1 at the validation threshold "
            f"{metrics.get('threshold')} is {_fmt_metric(test_metrics.get('f1'))}. "
            f"Test precision is {_fmt_metric(test_metrics.get('precision'))} and recall is "
            f"{_fmt_metric(test_metrics.get('recall'))}."
        )
    if metrics.get("accuracy_note"):
        measured.append(metrics["accuracy_note"])

    hypotheses = []
    reengagement = [row for row in segments if row["segment"] in {"Inactive", "Needs re-engagement"}]
    if reengagement:
        focus = max(reengagement, key=lambda row: int(row["customers"]))
        hypotheses.append(
            "Hypothesis, not a measured outcome: a retention offer aimed at the "
            f"{focus['segment']} segment ({int(focus['customers']):,} customers, "
            f"{_pct(float(focus['revenue_share']))} of profiled revenue) could be tested with a "
            "randomized holdout. The success metric would be the 30-day purchase rate versus the "
            "holdout, not the model's historical accuracy."
        )
    if test_metrics and test_metrics.get("pr_auc") is not None:
        hypotheses.append(
            "Hypothesis, not a measured outcome: ranking customers by the model's predicted "
            "probability and contacting those below the validation threshold may concentrate "
            "outreach. That hypothesis is only supported if a later experiment shows incremental "
            "purchases. The score is an association with past repurchase, not a causal effect of intervening."
        )
    hypotheses.append(
        "Hypothesis, not a measured outcome: because one-time and repeat customers can differ in "
        "recency and order cadence, a test of reminder timing around the median inter-purchase gap "
        "is more informative than a blanket discount. The historical gap is a description of this "
        "dataset, not a recommended wait time for another business."
    )
    limitations = [
        "The source is the UCI Online Retail dataset, a UK gift retailer with many wholesale buyers, "
        "from December 2010 through December 2011. It is not AstroTalk customer data and it is not a "
        "current consumer-app event stream.",
        "Anonymous invoices are excluded from customer metrics because they have no ID to follow. "
        "That removes revenue from the repeat-purchase calculations.",
        "Cancellations and non-merchandise lines are excluded from the purchase definition. "
        "A different treatment of returns would change revenue and labels.",
        "The final calendar day is partial. Months and horizons that include it are not a complete day of trade.",
        "RFM segment names are configurable rules. They are not the supervised label and they are not causal clusters.",
        "Model explanations are predictive associations. They do not say that changing recency or spend would cause a purchase.",
    ]
    return {"measured": measured, "hypotheses": hypotheses, "limitations": limitations}


def _fmt_metric(value) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.3f}"
