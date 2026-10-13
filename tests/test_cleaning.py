"""Cleaning rules keep inconvenient rows and record what happened to them."""

from src.data_cleaning import clean_transactions
from tests.helpers import raw_frame, raw_row


def test_exact_duplicates_are_flagged_and_not_double_counted():
    frame = raw_frame([raw_row(), raw_row(), raw_row(InvoiceNo="536366", Quantity=1, UnitPrice=10)])
    cleaned, report = clean_transactions(frame)
    duplicates = cleaned[cleaned["is_exact_duplicate"] == 1]
    purchases = cleaned[cleaned["is_purchase"] == 1]
    assert len(cleaned) == 3
    assert len(duplicates) == 1
    assert report["exact_duplicate_rows"] == 1
    assert purchases["line_revenue"].sum() == 6 * 2.55 + 10
    assert duplicates["line_revenue"].sum() == 0


def test_missing_customer_is_retained_and_excluded_from_purchases():
    frame = raw_frame([raw_row(CustomerID=None), raw_row(InvoiceNo="536366")])
    cleaned, report = clean_transactions(frame)
    assert len(cleaned) == 2
    assert report["missing_customer_rows"] == 1
    anonymous = cleaned[cleaned["missing_customer"] == 1]
    assert len(anonymous) == 1
    assert int(anonymous["is_purchase"].iloc[0]) == 0
    assert int(cleaned.loc[cleaned["invoice_no"] == "536366", "is_purchase"].iloc[0]) == 1


def test_cancellations_negative_prices_and_postage_are_not_purchases():
    frame = raw_frame(
        [
            raw_row(InvoiceNo="C536365", Quantity=-6),
            raw_row(InvoiceNo="536366", UnitPrice=0),
            raw_row(InvoiceNo="536367", StockCode="POST", UnitPrice=5, Quantity=1),
            raw_row(InvoiceNo="536368", StockCode="C2", Quantity=1, UnitPrice=50),
        ]
    )
    cleaned, report = clean_transactions(frame)
    assert len(cleaned) == 4
    assert int(cleaned["is_purchase"].sum()) == 0
    assert report["cancellation_rows"] == 1
    assert report["invalid_price_rows"] == 1
    assert report["non_merchandise_rows"] == 2
    assert "positive unit price" in report["purchase_definition"]


def test_unparseable_dates_are_removed_and_counted():
    frame = raw_frame([raw_row(InvoiceDate="not-a-date"), raw_row(InvoiceNo="536369")])
    cleaned, report = clean_transactions(frame)
    assert len(cleaned) == 1
    assert report["rows_removed_unusable"] == 1
    assert report["removed_unparseable_invoice_date"] == 1
    assert cleaned.iloc[0]["invoice_no"] == "536369"


def test_customer_id_float_is_normalized():
    cleaned, _ = clean_transactions(raw_frame([raw_row(CustomerID=17850.0)]))
    assert cleaned.iloc[0]["customer_id"] == "17850"
