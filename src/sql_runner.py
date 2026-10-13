"""Execute the analytics SQL pack and save each result."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from src.database import load_sql_file, read_query

logger = logging.getLogger("retention.sql")

ANALYTICS_FILES = [
    "02_overview_kpis.sql",
    "03_monthly_trends.sql",
    "04_repeat_purchase_rate.sql",
    "05_customer_frequency.sql",
    "06_average_order_value.sql",
    "07_rfm.sql",
    "08_top_products.sql",
    "09_monthly_cohorts.sql",
    "10_revenue_by_segment.sql",
    "11_declining_activity.sql",
    "12_customer_features.sql",
]


def run_analytics(engine: Engine, sql_dir: Path, output_dir: Path, as_of: str) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    manifest = []
    for name in ANALYTICS_FILES:
        path = sql_dir / name
        sql = load_sql_file(path)
        params = {"as_of": as_of} if ":as_of" in sql else None
        logger.info("Running %s", name)
        frame = read_query(engine, sql, params)
        key = name.replace(".sql", "")
        results[key] = frame
        csv_path = output_dir / f"{key}.csv"
        frame.to_csv(csv_path, index=False)
        manifest.append({"query": name, "rows": int(len(frame)), "columns": list(frame.columns)})
        logger.info("%s returned %s rows", name, len(frame))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results
