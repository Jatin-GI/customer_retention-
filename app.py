"""Overview page for the retention dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard_data import ArtifactMissing, load_eda, load_kpis, load_metrics, monthly_revenue
from src.ui import configure_page, missing_artifact, show_figure, show_frame

configure_page("Customer Retention Intelligence")
st.title("Customer Retention Intelligence")
st.markdown(
    "Behavioral analytics and repeat-purchase prediction on the **UCI Online Retail** dataset "
    "(CC BY 4.0). This is a portfolio simulation. It does not use AstroTalk data and it does not "
    "describe AstroTalk customers."
)

try:
    kpis = load_kpis()
    metrics = load_metrics()
    eda = load_eda()
except ArtifactMissing as exc:
    missing_artifact(exc)

columns = st.columns(4)
columns[0].metric("Customers", f"{int(kpis['purchase_customers']):,}")
columns[1].metric("Purchase invoices", f"{int(kpis['purchase_invoices']):,}")
columns[2].metric("Purchase revenue", f"£{float(kpis['purchase_revenue']):,.0f}")
columns[3].metric("Repeat-purchase rate", f"{float(kpis['repeat_purchase_rate']):.1%}")

st.caption(
    "Purchase revenue sums merchandise lines with a customer ID, a positive quantity, a positive price, "
    "and an invoice that is not a cancellation. Exact duplicates are counted once. "
    f"Observed invoices run from {eda['min_invoice_date']} to {eda['max_invoice_date']}."
)

st.subheader("Monthly purchase revenue")
try:
    monthly = monthly_revenue("2010-01", "2012-12")
except Exception as exc:  # noqa: BLE001 - show the saved query if the database file is unavailable
    st.warning(f"Live SQL filter failed ({exc}). Showing the saved monthly query.")
    from src.dashboard_data import load_query_result

    monthly = load_query_result("03_monthly_trends")
if not monthly.empty:
    chart = monthly.copy()
    chart["year_month"] = pd.to_datetime(chart["year_month"])
    st.line_chart(chart.set_index("year_month")["purchase_revenue"])
    show_frame(monthly)

st.subheader("What the series shows")
for note in eda.get("interpretations", []):
    if note["chart"] == "Monthly purchase revenue":
        st.markdown(f"**Pattern.** {note['pattern']}")
        st.markdown(f"**Why it matters.** {note['why']}")
        st.markdown(f"**Before a business decision.** {note['further']}")

selected = metrics.get("selected_model", "unknown")
test_metrics = metrics.get("test_metrics", {})
st.subheader("Model snapshot")
st.markdown(
    f"Selected model: `{selected}`, chosen by validation PR-AUC. "
    f"Test PR-AUC {test_metrics.get('pr_auc')}, test ROC-AUC {test_metrics.get('roc_auc')}, "
    f"test F1 {test_metrics.get('f1')} at threshold {metrics.get('threshold')}."
)
st.caption(metrics.get("accuracy_note", ""))
show_figure((metrics.get("plot_paths") or {}).get("precision_recall"))
