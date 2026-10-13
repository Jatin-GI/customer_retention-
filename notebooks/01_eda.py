"""Regenerate EDA figures from the pipeline outputs.

Run ``python -m src.pipeline`` first. This script reads the cleaned transactions,
the modeling table, and the saved SQL cohort result. It does not retrain models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import get_settings
from src.eda import run_eda


def main() -> None:
    settings = get_settings()
    transactions_path = settings.processed_dir / "transactions_clean.csv"
    modeling_path = settings.processed_dir / "modeling_table.csv"
    quality_path = settings.reports_dir / "data_quality.json"
    segments_path = settings.reports_dir / "segment_summary.csv"
    cohorts_path = settings.reports_dir / "sql" / "09_monthly_cohorts.csv"
    missing = [path for path in (transactions_path, modeling_path, quality_path, segments_path) if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing pipeline outputs: "
            + ", ".join(str(path) for path in missing)
            + ". Run python -m src.pipeline first."
        )
    transactions = pd.read_csv(transactions_path, parse_dates=["invoice_date"])
    modeling = pd.read_csv(modeling_path, parse_dates=["snapshot_date"])
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    segments = pd.read_csv(segments_path)
    cohorts = pd.read_csv(cohorts_path) if cohorts_path.exists() else pd.DataFrame()
    summary = run_eda(
        transactions,
        modeling,
        quality,
        segments,
        cohorts,
        settings.reports_dir / "figures",
    )
    for note in summary["interpretations"]:
        print(f"\n{note['chart']}")
        print(f"Shows: {note['shows']}")
        print(f"Pattern: {note['pattern']}")
        print(f"Why: {note['why']}")
        print(f"Further: {note['further']}")


if __name__ == "__main__":
    main()
