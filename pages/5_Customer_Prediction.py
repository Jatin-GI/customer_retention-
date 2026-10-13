"""Score one customer from an example or from entered features."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import FEATURE_COLUMNS, NUMERIC_FEATURES, SPARSE_NUMERIC_FEATURES
from src.dashboard_data import ArtifactMissing, load_examples, load_metrics, load_model
from src.explainability import contributions_for_row
from src.inference import FeatureInputError, score_row
from src.ui import configure_page, missing_artifact, show_frame

configure_page("Customer prediction")
st.title("Customer prediction")

try:
    metrics = load_metrics()
    examples_doc = load_examples()
    model = load_model()
except ArtifactMissing as exc:
    missing_artifact(exc)

st.warning(examples_doc.get("disclaimer") or "This output is a model estimate, not a certainty.")
st.caption(
    f"Horizon: {metrics['horizon_days']} days after the cutoff. "
    f"Decision threshold {metrics['threshold']} was chosen on validation to maximize F1. "
    "Customer IDs in the examples are the dataset's IDs, not personal identifiers from another company."
)

examples = examples_doc.get("examples") or []
labels = ["Enter features manually"] + [
    f"{item['customer_id']} @ {item['snapshot_date']} (stored probability {item['probability']:.3f})"
    for item in examples
]
choice = st.selectbox("Example customer", labels)
selected = None if choice == labels[0] else examples[labels.index(choice) - 1]
defaults = selected["features"] if selected else {column: 0 for column in FEATURE_COLUMNS}
if selected is None:
    defaults.update(
        {
            "frequency": 1,
            "country": "United Kingdom",
            "avg_interpurchase_days": None,
            "spend_trend_ratio": None,
        }
    )

with st.form("score"):
    entries = {}
    left, right = st.columns(2)
    numeric = NUMERIC_FEATURES + SPARSE_NUMERIC_FEATURES
    for index, column in enumerate(numeric):
        target = left if index % 2 == 0 else right
        current = defaults.get(column)
        entries[column] = target.text_input(
            column,
            value="" if current is None else str(current),
            help="Leave blank only for average interpurchase days or the spend-trend ratio.",
        )
    entries["country"] = st.text_input("country", value=str(defaults.get("country") or "United Kingdom"))
    submitted = st.form_submit_button("Estimate repeat-purchase probability")

if submitted or selected is not None:
    payload = dict(entries) if submitted else selected["features"]
    try:
        scored = score_row(model, payload, threshold=float(metrics["threshold"]))
    except FeatureInputError as exc:
        st.error(str(exc))
        st.stop()
    probability = scored["probability"]
    st.metric("Estimated probability of a purchase in the horizon", f"{probability:.1%}")
    decision = "at or above" if scored["predicted_repeat_purchase"] else "below"
    st.markdown(
        f"This estimate is **{decision}** the validation threshold of {metrics['threshold']}. "
        "A low probability is a candidate for a retention experiment, not a determination that the customer is gone."
    )
    if selected is not None and not submitted:
        st.caption(
            f"Stored pipeline probability for this example: {selected['probability']:.3f}. "
            f"Observed repeat purchase in that horizon: {selected['actual_repeat_purchase']}."
        )
    feature_frame = pd.DataFrame([payload])[FEATURE_COLUMNS]
    contributions = contributions_for_row(model, feature_frame)
    if contributions:
        st.subheader("Feature contributions for this row")
        st.caption("SHAP values are associations inside the model. They do not show the effect of changing the customer's behavior.")
        show_frame(pd.DataFrame(contributions))
    elif selected and selected.get("contributions"):
        st.subheader("Feature contributions saved for this example")
        show_frame(pd.DataFrame(selected["contributions"]))
    else:
        st.info("No row-level SHAP values are available. Use the global permutation ranking on the model page.")

with st.expander("Feature payload"):
    st.code(json.dumps(defaults if not submitted else entries, indent=2), language="json")
