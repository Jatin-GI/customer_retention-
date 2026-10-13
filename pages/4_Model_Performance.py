"""Model comparison, curves, and explanations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.dashboard_data import ArtifactMissing, load_metrics
from src.ui import configure_page, missing_artifact, show_figure, show_frame

configure_page("Model performance")
st.title("Model performance")

try:
    metrics = load_metrics()
except ArtifactMissing as exc:
    missing_artifact(exc)

settings = get_settings()
comparison_path = settings.reports_dir / "model_comparison.csv"
if not comparison_path.exists():
    missing_artifact(ArtifactMissing("Missing model_comparison.csv. Run python -m src.pipeline"))
comparison = pd.read_csv(comparison_path)

st.markdown(
    f"Model choice used **{metrics['selection_metric']}**. {metrics['selection_rule']} "
    f"{metrics['threshold_rule']}"
)
st.info(metrics.get("accuracy_note", ""))

split = st.radio("Split", ["validation", "test"], horizontal=True, index=1)
view = comparison[comparison["split"] == split].copy()
columns = [
    "model",
    "positive_rate",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "threshold",
    "tp",
    "fp",
    "tn",
    "fn",
]
show_frame(view[[column for column in columns if column in view.columns]])
st.caption(
    "The test split was not used to choose the model or the threshold. "
    "Accuracy sits next to the positive rate because a constant prediction can score well when one class is common."
)

st.subheader(f"Selected model: {metrics['selected_model']}")
plots = metrics.get("plot_paths") or {}
left, right = st.columns(2)
with left:
    show_figure(plots.get("confusion_matrix"))
    show_figure(plots.get("roc"))
with right:
    show_figure(plots.get("precision_recall"))
    show_figure(plots.get("calibration"))

st.subheader("What the model uses")
st.markdown(metrics.get("explanation_limitation", ""))
if metrics.get("shap_note"):
    st.caption(metrics["shap_note"])
for sentence in metrics.get("plain_language") or []:
    st.markdown(f"- {sentence}")

importance_path = settings.reports_dir / "permutation_importance.csv"
shap_path = settings.reports_dir / "shap_summary.csv"
if importance_path.exists():
    st.markdown("**Permutation importance on a validation sample**")
    show_frame(pd.read_csv(importance_path))
if shap_path.exists() and Path(shap_path).stat().st_size > 0:
    shap_frame = pd.read_csv(shap_path)
    if not shap_frame.empty:
        st.markdown("**Mean absolute SHAP by original feature**")
        show_frame(shap_frame)
