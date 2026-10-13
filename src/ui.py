"""Shared Streamlit display helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")


def show_frame(frame: pd.DataFrame, height: int | None = None) -> None:
    kwargs = {"hide_index": True}
    if height is not None:
        kwargs["height"] = height
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(frame, use_container_width=True, **kwargs)


def show_figure(path: str | None) -> None:
    if path and Path(path).exists():
        st.image(path)
        return
    st.caption("This figure has not been generated yet.")


def missing_artifact(exc: Exception) -> None:
    st.error(str(exc))
    st.markdown(
        "From the project root, with the virtual environment active, run `python -m src.pipeline` "
        "and then start the dashboard again."
    )
    st.stop()
