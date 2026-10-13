"""End-to-end run: ingest, clean, load, query, model, explain, and write reports."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src import __version__
from src.artifacts import write_json
from src.config import get_settings
from src.data_cleaning import clean_transactions
from src.data_ingestion import acquire_raw_table, file_sha256
from src.database import load_transactions, make_engine
from src.eda import run_eda
from src.explainability import explain_pipeline
from src.feature_engineering import audit_temporal_integrity, build_modeling_table, features_for_snapshot
from src.insights import build_insights
from src.logging_utils import configure_logging
from src.segmentation import segment_customers, segment_summary
from src.sql_runner import run_analytics
from src.train import train_models
from src.evaluate import save_evaluation_plots

logger = logging.getLogger("retention.pipeline")


def _segment_profile(transactions: pd.DataFrame) -> pd.DataFrame:
    purchases = transactions.loc[transactions["is_purchase"] == 1].copy()
    purchases["invoice_date"] = pd.to_datetime(purchases["invoice_date"])
    as_of = purchases["invoice_date"].max().normalize()
    profile = features_for_snapshot(transactions, as_of, horizon_days=1, min_history_invoices=1)
    drop_columns = [column for column in ("repeat_purchase", "label_window_start", "label_window_end", "split") if column in profile.columns]
    profile = profile.drop(columns=drop_columns)
    profile["profile_as_of"] = as_of
    profile["profile_note"] = (
        "Descriptive RFM as of the last observed purchase date. "
        "The future-purchase label is not used."
    )
    return segment_customers(profile)


def run_pipeline(steps: set[str] | None = None) -> dict:
    settings = get_settings()
    wanted = steps or {
        "ingest",
        "clean",
        "database",
        "features",
        "sql",
        "train",
        "explain",
        "segment",
        "eda",
        "insights",
    }
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(settings.reports_dir / "pipeline.log")
    logger.info("Starting pipeline version %s dialect %s", __version__, settings.sql_dialect)

    raw, raw_path = acquire_raw_table(settings.raw_data_path, settings.raw_data_file, settings.uci_zip_url)
    source_hash = file_sha256(raw_path)
    logger.info("Raw file %s sha256 %s", raw_path, source_hash)

    transactions, quality = clean_transactions(raw)
    quality["source_sha256"] = source_hash
    quality["source_path"] = str(raw_path)
    quality["license"] = "CC BY 4.0"
    quality["citation"] = "Chen, D. (2015). Online Retail. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33"
    write_json(settings.reports_dir / "data_quality.json", quality)
    clean_path = settings.processed_dir / "transactions_clean.csv"
    transactions.to_csv(clean_path, index=False)
    logger.info("Wrote %s", clean_path)

    if "database" in wanted or "sql" in wanted:
        engine = make_engine(settings.database_url)
        load_transactions(engine, transactions, settings.sql_dir, settings.sql_dialect)
    else:
        engine = None

    modeling, meta = build_modeling_table(
        transactions,
        horizon_days=settings.horizon_days,
        min_history_invoices=settings.min_history_invoices,
        warmup_days=settings.history_warmup_days,
    )
    audit = audit_temporal_integrity(
        transactions,
        modeling,
        horizon_days=settings.horizon_days,
        seed=settings.random_seed,
    )
    meta["leakage_audit"] = audit
    meta["source_sha256"] = source_hash
    write_json(settings.reports_dir / "modeling_metadata.json", meta)
    modeling_path = settings.processed_dir / "modeling_table.csv"
    modeling.to_csv(modeling_path, index=False)
    logger.info("Leakage audit %s", audit)

    as_of = meta["snapshots"]["test"][0]
    if engine is not None and "sql" in wanted:
        sql_results = run_analytics(
            engine,
            settings.sql_dir,
            settings.reports_dir / "sql",
            as_of=as_of,
        )
    else:
        sql_results = {}

    trained = train_models(modeling, settings.random_seed)
    model_path = settings.models_dir / "repeat_purchase_pipeline.joblib"
    joblib.dump(trained["selected_pipeline"], model_path)
    comparison = pd.DataFrame(trained["comparison"])
    comparison.to_csv(settings.reports_dir / "model_comparison.csv", index=False)
    plot_paths = save_evaluation_plots(
        trained["test_scores"]["y_true"],
        trained["test_scores"]["y_prob"],
        trained["threshold"],
        settings.reports_dir / "figures",
        stem="test",
    )
    score_path = settings.processed_dir / "test_predictions.csv"
    keep_columns = [
        column
        for column in trained["test_scores"].columns
        if column not in {"label_window_start", "label_window_end"}
    ]
    trained["test_scores"].loc[:, keep_columns].to_csv(score_path, index=False)

    validation = modeling[modeling["split"] == "validation"]
    test = modeling[modeling["split"] == "test"]
    explanation = explain_pipeline(
        trained["selected_pipeline"],
        validation,
        test,
        seed=settings.random_seed,
    )
    if not explanation["importance"].empty:
        explanation["importance"].to_csv(settings.reports_dir / "permutation_importance.csv", index=False)
    if not explanation["shap_summary"].empty:
        explanation["shap_summary"].to_csv(settings.reports_dir / "shap_summary.csv", index=False)
    if explanation.get("coefficients") is not None:
        explanation["coefficients"].to_csv(settings.reports_dir / "logistic_coefficients.csv", index=False)

    profile = _segment_profile(transactions)
    summary = segment_summary(profile)
    summary.to_csv(settings.reports_dir / "segment_summary.csv", index=False)
    profile.to_csv(settings.processed_dir / "customer_segments.csv", index=False)

    cohorts = sql_results.get("09_monthly_cohorts", pd.DataFrame())
    if cohorts.empty:
        cohort_path = settings.reports_dir / "sql" / "09_monthly_cohorts.csv"
        if cohort_path.exists():
            cohorts = pd.read_csv(cohort_path)
    eda_summary = run_eda(
        transactions,
        modeling,
        quality,
        summary,
        cohorts,
        settings.reports_dir / "figures",
    )
    kpis = sql_results.get("02_overview_kpis")
    if kpis is None or kpis.empty:
        kpi_path = settings.reports_dir / "sql" / "02_overview_kpis.csv"
        kpis = pd.read_csv(kpi_path) if kpi_path.exists() else pd.DataFrame()
    kpi_dict = {} if kpis.empty else kpis.iloc[0].to_dict()
    insights = build_insights(
        kpi_dict,
        summary.to_dict(orient="records"),
        {
            "selected_model": trained["selected_model"],
            "threshold": trained["threshold"],
            "comparison": trained["comparison"],
            "accuracy_note": trained["accuracy_note"],
        },
        eda_summary,
    )
    write_json(settings.reports_dir / "insights.json", insights)
    write_json(
        settings.reports_dir / "example_customers.json",
        {
            "disclaimer": (
                "The probability is a model estimate from historical retail transactions. "
                "It is not a certainty and it is not a causal effect of contacting the customer."
            ),
            "examples": explanation["examples"],
        },
    )
    card = {
        "project_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": trained["selected_model"],
        "selection_metric": trained["selection_metric"],
        "selection_rule": trained["selection_rule"],
        "threshold": trained["threshold"],
        "threshold_rule": trained["threshold_rule"],
        "horizon_days": settings.horizon_days,
        "random_seed": settings.random_seed,
        "snapshots": meta["snapshots"],
        "inner_holdout_snapshot": trained["inner_holdout_snapshot"],
        "feature_columns": meta["feature_columns"],
        "target_definition": meta["target_definition"],
        "positive_rate_by_split": meta["positive_rate_by_split"],
        "leakage_audit": audit,
        "database_dialect": settings.sql_dialect,
        "source_sha256": source_hash,
        "model_path": str(model_path),
        "plot_paths": plot_paths,
        "explanation_methods": explanation["method"],
        "shap_note": explanation.get("shap_note"),
        "explanation_limitation": explanation["limitation"],
        "plain_language": explanation["sentences"],
        "accuracy_note": trained["accuracy_note"],
        "fitted_details": trained["fitted_details"],
        "test_metrics": next(
            row for row in trained["comparison"] if row["model"] == trained["selected_model"] and row["split"] == "test"
        ),
        "validation_metrics": next(
            row
            for row in trained["comparison"]
            if row["model"] == trained["selected_model"] and row["split"] == "validation"
        ),
    }
    write_json(settings.models_dir / "model_card.json", card)
    write_json(settings.reports_dir / "metrics.json", card)
    logger.info("Selected %s. Test metrics: %s", trained["selected_model"], card["test_metrics"])
    return {"model_card": card, "quality": quality, "insights": insights}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the customer retention pipeline.")
    parser.add_argument(
        "--steps",
        default="",
        help="Comma-separated subset. Default runs every step.",
    )
    args = parser.parse_args()
    steps = {step.strip() for step in args.steps.split(",") if step.strip()} or None
    run_pipeline(steps)


if __name__ == "__main__":
    main()
