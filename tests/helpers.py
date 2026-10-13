"""Shared builders for unit tests. They do not read the UCI file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQLITE_SQL = ROOT / "sql" / "sqlite"
POSTGRES_SQL = ROOT / "sql" / "postgres"


def raw_row(**overrides) -> dict:
    row = {
        "InvoiceNo": "536365",
        "StockCode": "85123A",
        "Description": "WHITE MUG",
        "Quantity": 6,
        "InvoiceDate": "2021-01-10 08:26:00",
        "UnitPrice": 2.55,
        "CustomerID": 17850,
        "Country": "United Kingdom",
    }
    row.update(overrides)
    return row


def raw_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def purchase_line(
    line_id: int,
    invoice_no: str,
    customer_id: str,
    invoice_date: str,
    quantity: int = 2,
    unit_price: float = 5.0,
    stock_code: str = "11111",
    **overrides,
) -> dict:
    row = {
        "line_id": line_id,
        "invoice_no": invoice_no,
        "stock_code": stock_code,
        "description": "Mug",
        "quantity": quantity,
        "invoice_date": invoice_date,
        "unit_price": unit_price,
        "customer_id": customer_id,
        "country": "United Kingdom",
        "is_cancellation": 0,
        "is_return": 0,
        "invalid_price": 0,
        "is_exact_duplicate": 0,
        "is_non_merchandise": 0,
        "missing_customer": 0,
        "is_purchase": 1,
        "line_revenue": quantity * unit_price,
    }
    row.update(overrides)
    if "line_revenue" not in overrides and row["is_exact_duplicate"] == 0:
        row["line_revenue"] = row["quantity"] * (row["unit_price"] or 0)
    return row
