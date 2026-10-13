"""Load cleaned transactions into SQLite or PostgreSQL and build dimension tables."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger("retention.database")


def make_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite"):
        return create_engine(database_url, future=True)
    if database_url.startswith("postgresql"):
        return create_engine(database_url, future=True, pool_pre_ping=True)
    raise ValueError(
        "Unsupported DATABASE_URL. Use sqlite:///... or postgresql+psycopg://..."
    )


def _strip_sql_comments(sql: str) -> str:
    lines = [line for line in sql.splitlines() if not line.strip().startswith("--")]
    return "\n".join(lines)


def execute_script(engine: Engine, sql: str) -> None:
    """Execute a script of semicolon-separated statements."""
    cleaned = _strip_sql_comments(sql)
    statements = [part.strip() for part in cleaned.split(";") if part.strip()]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def load_sql_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")


def prepare_frame_for_database(transactions: pd.DataFrame, dialect: str) -> pd.DataFrame:
    frame = transactions.copy()
    if dialect == "sqlite":
        frame["invoice_date"] = pd.to_datetime(frame["invoice_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    for column in [
        "is_cancellation",
        "is_return",
        "invalid_price",
        "is_exact_duplicate",
        "is_non_merchandise",
        "missing_customer",
        "is_purchase",
    ]:
        frame[column] = frame[column].astype(int)
    frame["line_revenue"] = frame["line_revenue"].astype(float)
    for column in ("customer_id", "description", "country", "stock_code"):
        frame[column] = frame[column].where(frame[column].notna(), None)
    frame["unit_price"] = frame["unit_price"].astype(object).where(frame["unit_price"].notna(), None)
    return frame


def load_transactions(engine: Engine, transactions: pd.DataFrame, sql_dir: Path, dialect: str) -> None:
    schema = load_sql_file(sql_dir / "00_schema.sql")
    logger.info("Creating transaction table from %s", sql_dir / "00_schema.sql")
    execute_script(engine, schema)
    frame = prepare_frame_for_database(transactions, dialect)
    logger.info("Inserting %s transaction lines", len(frame))
    frame.to_sql(
        "transactions",
        engine,
        if_exists="append",
        index=False,
        chunksize=5000,
    )
    dimensions = load_sql_file(sql_dir / "01_dimensions.sql")
    logger.info("Building customer and product tables and indexes")
    execute_script(engine, dimensions)


def read_query(engine: Engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    # Comment lines are removed so a bind name written in a comment is not sent
    # to the driver as an extra placeholder.
    statement = text(_strip_sql_comments(sql))
    with engine.connect() as connection:
        return pd.read_sql_query(statement, connection, params=params or {})
