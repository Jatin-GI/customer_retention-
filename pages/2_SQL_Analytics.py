"""SQL analytics loaded from the database results."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard_data import ArtifactMissing, load_query_result, monthly_revenue
from src.ui import configure_page, missing_artifact, show_frame

configure_page("SQL analytics")
st.title("SQL analytics")
st.markdown(
    "These tables were produced by the SQL files in `sql/sqlite` (or `sql/postgres` if that dialect "
    "is configured). PostgreSQL scripts are included. This environment executed the SQLite scripts "
    "because PostgreSQL was not installed."
)

QUERIES = {
    "Overview KPIs": "02_overview_kpis",
    "Monthly trends": "03_monthly_trends",
    "Repeat-purchase bands": "04_repeat_purchase_rate",
    "Customer frequency": "05_customer_frequency",
    "Average order value by country": "06_average_order_value",
    "RFM scores": "07_rfm",
    "Top products": "08_top_products",
    "Monthly cohorts": "09_monthly_cohorts",
    "Revenue by segment": "10_revenue_by_segment",
    "Declining activity": "11_declining_activity",
    "Customer features at the test cutoff": "12_customer_features",
}

choice = st.selectbox("Query", list(QUERIES))
try:
    frame = load_query_result(QUERIES[choice])
except ArtifactMissing as exc:
    missing_artifact(exc)

if choice == "Monthly trends" and "year_month" in frame.columns:
    months = sorted(frame["year_month"].astype(str).unique())
    start, end = st.select_slider("Month range", options=months, value=(months[0], months[-1]))
    try:
        filtered = monthly_revenue(start, end)
        st.caption("This view was re-queried from the local database for the selected months.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not re-query the database ({exc}). Filtered the saved result instead.")
        filtered = frame[frame["year_month"].astype(str).between(start, end)]
    show_frame(filtered)
elif choice == "Revenue by segment" and "segment" in frame.columns:
    segments = ["All segments", *sorted(frame["segment"].astype(str).unique())]
    picked = st.selectbox("Segment", segments)
    filtered = frame if picked == "All segments" else frame[frame["segment"] == picked]
    show_frame(filtered)
elif choice == "Customer frequency":
    st.caption("Showing the 50 highest-revenue customers from the full SQL result.")
    show_frame(frame.head(50))
elif choice == "RFM scores":
    st.caption("Showing 50 customers from the SQL RFM score query. Scores use NTILE quintiles.")
    show_frame(frame.head(50))
elif choice == "Declining activity":
    st.caption(
        f"{len(frame):,} customers have fewer purchase invoices in the latest 90 days than in the prior 90 days, "
        "with at least two invoices in the prior window."
    )
    show_frame(frame.head(50))
elif choice == "Customer features at the test cutoff":
    st.caption("SQL feature mart at the test snapshot. The training table uses the same definitions at every snapshot.")
    show_frame(frame.head(50))
else:
    show_frame(frame)

if choice == "Top products":
    st.info(
        "The UCI file has stock codes and descriptions, not a product-category hierarchy. "
        "This query ranks merchandise. It does not invent categories."
    )
