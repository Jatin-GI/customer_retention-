"""Measured results and hypotheses that are labeled as hypotheses."""

from __future__ import annotations

import streamlit as st

from src.dashboard_data import ArtifactMissing, load_eda, load_insights
from src.ui import configure_page, missing_artifact, show_figure

configure_page("Business insights")
st.title("Business insights")
st.markdown(
    "Measured statements below are calculated from this dataset. "
    "Actions are hypotheses for an experiment. They are not proven outcomes, and they are not findings about AstroTalk."
)

try:
    insights = load_insights()
    eda = load_eda()
except ArtifactMissing as exc:
    missing_artifact(exc)

st.subheader("Measured in this dataset")
for paragraph in insights["measured"]:
    st.markdown(f"- {paragraph}")

st.subheader("Hypotheses a business could test")
for paragraph in insights["hypotheses"]:
    st.markdown(f"- {paragraph}")

st.subheader("Limitations")
for paragraph in insights["limitations"]:
    st.markdown(f"- {paragraph}")

st.subheader("Analysis notes")
for note in eda.get("interpretations", []):
    with st.expander(note["chart"]):
        st.markdown(f"**What it shows.** {note['shows']}")
        st.markdown(f"**Pattern in this dataset.** {note['pattern']}")
        st.markdown(f"**Why it matters.** {note['why']}")
        st.markdown(f"**Further analysis before a decision.** {note['further']}")

figures = eda.get("plots") or []
if figures:
    st.subheader("Figures")
    for path in figures:
        show_figure(path)
