"""Load pipeline artifacts and run a small live SQL filter for the dashboard."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.artifacts import read_json
from src.config import Settings, get_settings
from src.database import make_engine, read_query


class ArtifactMissing(FileNotFoundError):
    """Raised when the dashboard is opened before the pipeline has been run."""


def _require(path: Path) -> Path:
    if not path.exists():
        raise ArtifactMissing(
            f"Missing {path.name}. From the project root, run: python -m src.pipeline"
        )
    return path


def project_settings() -> Settings:
    return get_settings()


def load_kpis(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    frame = pd.read_csv(_require(settings.reports_dir / "sql" / "02_overview_kpis.csv"))
    if frame.empty:
        raise ArtifactMissing("The overview SQL result is empty.")
    return frame.iloc[0].to_dict()


def load_query_result(name: str, settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    return pd.read_csv(_require(settings.reports_dir / "sql" / f"{name}.csv"))


def load_metrics(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    return read_json(_require(settings.reports_dir / "metrics.json"))


def load_model(settings: Settings | None = None):
    settings = settings or get_settings()
    return joblib.load(_require(settings.models_dir / "repeat_purchase_pipeline.joblib"))


def load_examples(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    path = settings.reports_dir / "example_customers.json"
    if not path.exists():
        return {"disclaimer": "Run the pipeline to score example customers.", "examples": []}
    return read_json(path)


def load_insights(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    return read_json(_require(settings.reports_dir / "insights.json"))


def load_eda(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    return read_json(_require(settings.reports_dir / "eda_summary.json"))


def monthly_revenue(start: str, end: str, settings: Settings | None = None) -> pd.DataFrame:
    """Re-query purchase revenue for an inclusive month range."""
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)
    if settings.sql_dialect == "sqlite":
        month_expr = "strftime('%Y-%m', invoice_date)"
    elif settings.sql_dialect == "postgres":
        month_expr = "to_char(invoice_date, 'YYYY-MM')"
    else:
        raise ValueError(f"Unsupported dialect {settings.sql_dialect}")
    sql = f"""
        SELECT
            {month_expr} AS year_month,
            COUNT(DISTINCT invoice_no) AS purchase_invoices,
            COUNT(DISTINCT customer_id) AS active_customers,
            ROUND(SUM(line_revenue), 2) AS purchase_revenue
        FROM transactions
        WHERE is_purchase = 1
          AND {month_expr} >= :start_month
          AND {month_expr} <= :end_month
        GROUP BY {month_expr}
        ORDER BY year_month
    """
    if settings.sql_dialect == "postgres":
        sql = sql.replace("ROUND(SUM(line_revenue), 2)", "ROUND(SUM(line_revenue)::numeric, 2)")
    return read_query(engine, sql, {"start_month": start, "end_month": end})
