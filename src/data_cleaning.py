"""Clean UCI Online Retail lines and record every exclusion decision.

Rows are retained in the cleaned table whenever they can be identified.
Flags record cancellations, returns, invalid prices, duplicates, missing
customer IDs, and non-merchandise adjustments. Only rows that cannot be
placed in time or have no invoice number are removed, and those removals
are counted in the quality report.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import ADJUSTMENT_STOCK_CODES

logger = logging.getLogger("retention.cleaning")

OUTPUT_COLUMNS = [
    "line_id",
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
    "is_cancellation",
    "is_return",
    "invalid_price",
    "is_exact_duplicate",
    "is_non_merchandise",
    "missing_customer",
    "is_purchase",
    "line_revenue",
]


def _normalize_customer_id(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def _is_non_merchandise(stock_code: str) -> bool:
    code = stock_code.strip().upper()
    if not code:
        return True
    if code in ADJUSTMENT_STOCK_CODES:
        return True
    return not any(character.isdigit() for character in code)


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return cleaned transactions and a data-quality report.

    Purchase definition (``is_purchase = 1``), used for revenue, RFM, and the
    repeat-purchase label:

    - invoice number is present and does not start with ``C``
    - invoice date parsed
    - customer ID is present
    - quantity is strictly positive
    - unit price is strictly positive
    - stock code is merchandise (contains a digit and is not a known fee code)
    - the row is not an exact duplicate of an earlier row

    Cancellations, returns, anonymous checkouts, fee lines, and duplicate
    copies stay in the table with flags. ``line_revenue`` is
    ``quantity * unit_price`` for the first copy of a priced line, and 0 for
    exact duplicate copies so they cannot be summed twice.
    """
    if raw.empty:
        raise ValueError("The raw dataset is empty.")

    frame = raw.copy()
    source_rows = int(len(frame))
    frame.columns = [str(column).strip() for column in frame.columns]
    required = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot clean data. Missing columns: {missing}")

    report: dict[str, Any] = {
        "source_rows": source_rows,
        "source_columns": list(frame.columns),
        "decisions": [],
    }

    work = pd.DataFrame(index=frame.index)
    work["invoice_no"] = frame["InvoiceNo"].map(lambda value: None if pd.isna(value) else str(value).strip())
    work["stock_code"] = frame["StockCode"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    work["description"] = frame["Description"].map(lambda value: None if pd.isna(value) else str(value).strip())
    work["quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    invoice_dates = frame["InvoiceDate"]
    if pd.api.types.is_datetime64_any_dtype(invoice_dates):
        work["invoice_date"] = pd.to_datetime(invoice_dates, errors="coerce")
    else:
        # The xlsx stores real datetimes. CSV exports of this dataset are
        # month/day/year, matching the UCI timestamp values.
        work["invoice_date"] = pd.to_datetime(invoice_dates, errors="coerce", dayfirst=False)
    work["unit_price"] = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    work["customer_id"] = frame["CustomerID"].map(_normalize_customer_id)
    work["country"] = frame["Country"].map(lambda value: None if pd.isna(value) else str(value).strip())

    missing_invoice = work["invoice_no"].isna() | work["invoice_no"].eq("")
    bad_date = work["invoice_date"].isna()
    bad_quantity = work["quantity"].isna()
    rejected_mask = missing_invoice | bad_date | bad_quantity
    rejected = work.loc[rejected_mask].copy()
    report["rows_removed_unusable"] = int(rejected_mask.sum())
    report["removed_missing_invoice_no"] = int(missing_invoice.sum())
    report["removed_unparseable_invoice_date"] = int((bad_date & ~missing_invoice).sum())
    report["removed_non_numeric_quantity"] = int((bad_quantity & ~missing_invoice & ~bad_date).sum())
    report["decisions"].append(
        {
            "rule": "Drop rows with no invoice number, an unparseable invoice date, or a non-numeric quantity.",
            "rows_affected": int(rejected_mask.sum()),
            "action": "Removed from the analysis table and counted here. A sample is stored in rejected_row_sample.",
            "reason": "These rows cannot support a transaction key or a timeline.",
        }
    )
    report["rejected_row_sample"] = (
        rejected[["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"]]
        .head(20)
        .assign(invoice_date=lambda df: df["invoice_date"].astype(str))
        .to_dict(orient="records")
    )

    kept = work.loc[~rejected_mask].copy()
    kept["quantity"] = kept["quantity"].round().astype(int)
    kept["invoice_no_upper"] = kept["invoice_no"].str.upper()
    kept["is_cancellation"] = kept["invoice_no_upper"].str.startswith("C").astype(int)
    kept["is_return"] = (kept["quantity"] < 0).astype(int)
    kept["invalid_price"] = (kept["unit_price"].isna() | (kept["unit_price"] <= 0)).astype(int)
    kept["missing_customer"] = kept["customer_id"].isna().astype(int)
    kept["is_non_merchandise"] = kept["stock_code"].map(_is_non_merchandise).astype(int)
    kept["missing_description"] = kept["description"].isna().astype(int)

    duplicate_keys = [
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
    ]
    kept = kept.sort_values(["invoice_date", "invoice_no", "stock_code"], kind="mergesort")
    kept["is_exact_duplicate"] = kept.duplicated(duplicate_keys, keep="first").astype(int)

    kept["is_purchase"] = (
        (kept["is_cancellation"] == 0)
        & (kept["quantity"] > 0)
        & (kept["invalid_price"] == 0)
        & (kept["missing_customer"] == 0)
        & (kept["is_exact_duplicate"] == 0)
        & (kept["is_non_merchandise"] == 0)
    ).astype(int)

    priced = kept["unit_price"].fillna(0)
    kept["line_revenue"] = np.where(
        kept["is_exact_duplicate"] == 1,
        0.0,
        kept["quantity"] * priced,
    )

    report["rows_retained"] = int(len(kept))
    report["missing_customer_rows"] = int(kept["missing_customer"].sum())
    report["cancellation_rows"] = int(kept["is_cancellation"].sum())
    report["return_rows"] = int(kept["is_return"].sum())
    report["invalid_price_rows"] = int(kept["invalid_price"].sum())
    report["exact_duplicate_rows"] = int(kept["is_exact_duplicate"].sum())
    report["non_merchandise_rows"] = int(kept["is_non_merchandise"].sum())
    report["missing_description_rows"] = int(kept["missing_description"].sum())
    report["zero_quantity_rows"] = int((kept["quantity"] == 0).sum())
    report["purchase_rows"] = int(kept["is_purchase"].sum())
    report["purchase_customers"] = int(kept.loc[kept["is_purchase"] == 1, "customer_id"].nunique())
    report["purchase_invoices"] = int(kept.loc[kept["is_purchase"] == 1, "invoice_no"].nunique())
    report["purchase_revenue"] = float(kept.loc[kept["is_purchase"] == 1, "line_revenue"].sum())
    non_merch = kept["is_non_merchandise"] == 1
    report["non_merchandise_revenue_excluded"] = float(
        kept.loc[non_merch & (kept["is_exact_duplicate"] == 0), "line_revenue"].sum()
    )
    report["duplicate_revenue_suppressed"] = float(
        (
            kept.loc[kept["is_exact_duplicate"] == 1, "quantity"]
            * kept.loc[kept["is_exact_duplicate"] == 1, "unit_price"].fillna(0)
        ).sum()
    )
    report["unattributed_positive_revenue"] = float(
        kept.loc[
            (kept["missing_customer"] == 1)
            & (kept["is_exact_duplicate"] == 0)
            & (kept["quantity"] > 0)
            & (kept["invalid_price"] == 0)
            & (kept["is_cancellation"] == 0)
            & (kept["is_non_merchandise"] == 0),
            "line_revenue",
        ].sum()
    )
    code_counts = (
        kept.loc[kept["is_non_merchandise"] == 1, "stock_code"]
        .value_counts()
        .head(15)
        .to_dict()
    )
    report["non_merchandise_stock_codes"] = {str(key): int(value) for key, value in code_counts.items()}
    if not kept.empty:
        report["min_invoice_date"] = str(kept["invoice_date"].min())
        report["max_invoice_date"] = str(kept["invoice_date"].max())
    else:
        report["min_invoice_date"] = None
        report["max_invoice_date"] = None

    report["decisions"].extend(
        [
            {
                "rule": "Missing CustomerID",
                "rows_affected": report["missing_customer_rows"],
                "action": "Retained and excluded from purchase revenue, RFM, and the model population.",
                "reason": "Repeat purchase is defined at the customer level. Anonymous lines have no identity to follow.",
            },
            {
                "rule": "Invoice number starts with C, or quantity is negative",
                "rows_affected": int(((kept["is_cancellation"] == 1) | (kept["is_return"] == 1)).sum()),
                "action": "Retained with is_cancellation and is_return flags. Excluded from the purchase definition.",
                "reason": "The UCI documentation defines a leading C as a cancellation. Negative quantities are returns.",
            },
            {
                "rule": "Unit price missing or not strictly positive",
                "rows_affected": report["invalid_price_rows"],
                "action": "Retained and excluded from purchase revenue.",
                "reason": "Zero and negative prices are adjustments, not sold merchandise.",
            },
            {
                "rule": "Exact duplicate of invoice, product, quantity, price, date, customer, and country",
                "rows_affected": report["exact_duplicate_rows"],
                "action": "Retained with is_exact_duplicate = 1. line_revenue set to 0 on copies after the first.",
                "reason": "Copies would double-count orders. The first copy is kept for measurement.",
            },
            {
                "rule": "Non-merchandise stock code (no digit, blank, or a known fee code such as POST, DOT, M, D, C2)",
                "rows_affected": report["non_merchandise_rows"],
                "action": "Retained and excluded from purchase revenue and product counts.",
                "reason": "Postage, manual adjustments, discounts, and bank charges are not products and would inflate monetary value.",
            },
            {
                "rule": "Missing product description",
                "rows_affected": report["missing_description_rows"],
                "action": "Retained. Description is not required for the purchase definition.",
                "reason": "The stock code still identifies the item.",
            },
        ]
    )
    report["purchase_definition"] = (
        "A purchase line has a customer ID, a parsed date, a positive quantity, "
        "a positive unit price, an invoice number that does not start with C, "
        "a merchandise stock code, and is the first copy of an exact duplicate group."
    )

    kept = kept.reset_index(drop=True)
    kept.insert(0, "line_id", np.arange(1, len(kept) + 1))
    result = kept[OUTPUT_COLUMNS].copy()
    logger.info(
        "Cleaning retained %s of %s rows; purchase rows=%s customers=%s revenue=%.2f",
        report["rows_retained"],
        source_rows,
        report["purchase_rows"],
        report["purchase_customers"],
        report["purchase_revenue"],
    )
    return result, report
