"""Rule-based RFM segments."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.dashboard_data import ArtifactMissing, load_query_result
from src.ui import configure_page, missing_artifact, show_figure, show_frame

configure_page("Customer segmentation")
st.title("Customer segmentation")
st.markdown(
    "Segments use ordered rules on recency, frequency, and monetary quintiles. "
    "Score 5 is the favorable end. The first matching rule wins. "
    "These groups are not the machine-learning label."
)
st.markdown(
    """
- **Recent and frequent:** recency score 4–5 and frequency score 4–5.
- **High value:** monetary score 4–5 and frequency score 3–5.
- **Promising:** recency score 4–5 and frequency score 1–2.
- **Needs re-engagement:** recency score 1–2 and frequency score 3–5.
- **Inactive:** recency score 1–2 and frequency score 1–2.
- **Regular:** everyone else.
"""
)

settings = get_settings()
python_path = settings.reports_dir / "segment_summary.csv"
try:
    if not python_path.exists():
        raise ArtifactMissing(f"Missing {python_path.name}. Run python -m src.pipeline")
    python_segments = pd.read_csv(python_path)
    sql_segments = load_query_result("10_revenue_by_segment")
except ArtifactMissing as exc:
    missing_artifact(exc)

st.subheader("Python quintile rules")
st.caption(
    "Profiled on all observed purchase history, as of the last invoice date. "
    "The future-purchase label is not used."
)
show_frame(python_segments)
show_figure(str(settings.reports_dir / "figures" / "07_segment_revenue.png"))

st.subheader("SQL NTILE rules")
st.caption(
    "The SQL query uses NTILE(5). A customer on a boundary can land in a different segment "
    "than in the pandas quintiles. Both tables are descriptive."
)
show_frame(sql_segments)
