"""Central configuration loaded from the environment and project defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Purchase-definition stock codes that are fees, postage, or manual adjustments
# rather than products. The no-digit rule in cleaning catches most of these.
# C2 is kept here because it contains a digit but is a carriage adjustment.
ADJUSTMENT_STOCK_CODES = frozenset(
    {
        "POST",
        "DOT",
        "M",
        "D",
        "S",
        "AMAZONFEE",
        "BANK CHARGES",
        "CRUK",
        "PADS",
        "B",
        "C2",
        "ADJUST",
        "ADJUST2",
    }
)

NUMERIC_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_quantity",
    "n_distinct_products",
    "tenure_days",
    "spend_30d",
    "spend_90d",
    "invoices_30d",
    "invoices_90d",
    "avg_unit_price",
    "return_rate",
    "spend_per_tenure_day",
]

SPARSE_NUMERIC_FEATURES = [
    "avg_interpurchase_days",
    "spend_trend_ratio",
]

CATEGORICAL_FEATURES = ["country"]

FEATURE_COLUMNS = NUMERIC_FEATURES + SPARSE_NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "repeat_purchase"

# First matching rule wins. Scores are quintiles from 1 (low) to 5 (high).
SEGMENT_RULES = [
    {"name": "Recent and frequent", "r_score": (4, 5), "f_score": (4, 5)},
    {"name": "High value", "m_score": (4, 5), "f_score": (3, 5)},
    {"name": "Promising", "r_score": (4, 5), "f_score": (1, 2)},
    {"name": "Needs re-engagement", "r_score": (1, 2), "f_score": (3, 5)},
    {"name": "Inactive", "r_score": (1, 2), "f_score": (1, 2)},
]


def _resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = ROOT / path
    return path


def resolve_database_url(url: str, root: Path = ROOT) -> str:
    """Turn a relative SQLite URL into an absolute one so cwd does not matter."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw = url[len(prefix) :]
    if raw.startswith("/") and not raw.startswith("//"):
        # Absolute POSIX path, or a Windows path that as_posix() already produced.
        candidate = Path(raw)
        if candidate.is_absolute():
            return prefix + candidate.as_posix()
    path = Path(raw)
    if not path.is_absolute():
        path = (root / path).resolve()
    return prefix + path.resolve().as_posix()


@dataclass
class Settings:
    root: Path
    raw_data_path: Path
    raw_data_file: Path | None
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    database_url: str
    random_seed: int
    horizon_days: int
    min_history_invoices: int
    history_warmup_days: int
    uci_zip_url: str = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

    @property
    def sql_dialect(self) -> str:
        if self.database_url.startswith("postgresql"):
            return "postgres"
        if self.database_url.startswith("sqlite"):
            return "sqlite"
        raise ValueError(
            "DATABASE_URL must start with sqlite:/// or postgresql+psycopg://. "
            f"Received: {self.database_url}"
        )

    @property
    def sql_dir(self) -> Path:
        return self.root / "sql" / self.sql_dialect


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    processed = _resolve_path(os.getenv("PROCESSED_DIR"), ROOT / "data" / "processed")
    raw_file = os.getenv("RAW_DATA_FILE")
    database_url = os.getenv("DATABASE_URL", "sqlite:///data/processed/retail.sqlite")
    return Settings(
        root=ROOT,
        raw_data_path=_resolve_path(os.getenv("RAW_DATA_PATH"), ROOT / "data" / "raw"),
        raw_data_file=_resolve_path(raw_file, Path(raw_file)) if raw_file else None,
        processed_dir=processed,
        models_dir=_resolve_path(os.getenv("MODELS_DIR"), ROOT / "models"),
        reports_dir=_resolve_path(os.getenv("REPORTS_DIR"), ROOT / "reports"),
        database_url=resolve_database_url(database_url),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        horizon_days=int(os.getenv("PREDICTION_HORIZON_DAYS", "30")),
        min_history_invoices=int(os.getenv("MIN_HISTORY_INVOICES", "1")),
        history_warmup_days=int(os.getenv("HISTORY_WARMUP_DAYS", "90")),
    )
