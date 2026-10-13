"""Execute the SQLite analytics pack on a fixture and compare it with Python features."""

import pandas as pd

from src.database import load_transactions, make_engine, read_query
from src.feature_engineering import features_for_snapshot
from src.sql_runner import ANALYTICS_FILES, run_analytics
from tests.helpers import POSTGRES_SQL, SQLITE_SQL, purchase_line


def _fixture() -> pd.DataFrame:
    rows = [
        purchase_line(1, "A1", "1", "2021-01-01 00:00:00", quantity=2, unit_price=5),
        purchase_line(2, "A2", "1", "2021-02-01 00:00:00", quantity=1, unit_price=20),
        purchase_line(3, "B1", "2", "2021-01-15 00:00:00", quantity=1, unit_price=5),
        purchase_line(4, "C1", "3", "2020-10-01 00:00:00", quantity=1, unit_price=8),
        purchase_line(5, "C2", "3", "2020-11-01 00:00:00", quantity=1, unit_price=8),
        purchase_line(
            6,
            "C536365",
            "1",
            "2021-01-20 00:00:00",
            quantity=-1,
            unit_price=5,
            is_cancellation=1,
            is_return=1,
            is_purchase=0,
            line_revenue=-5,
        ),
        purchase_line(
            7,
            "D1",
            None,
            "2021-01-05 00:00:00",
            quantity=1,
            unit_price=4,
            missing_customer=1,
            is_purchase=0,
            line_revenue=4,
        ),
        purchase_line(
            8,
            "A1",
            "1",
            "2021-01-01 00:00:00",
            quantity=2,
            unit_price=5,
            is_exact_duplicate=1,
            is_purchase=0,
            line_revenue=0,
        ),
    ]
    return pd.DataFrame(rows)


def test_sqlite_queries_match_the_purchase_definition(tmp_path):
    database = tmp_path / "fixture.sqlite"
    engine = make_engine("sqlite:///" + database.resolve().as_posix())
    load_transactions(engine, _fixture(), SQLITE_SQL, "sqlite")
    results = run_analytics(engine, SQLITE_SQL, tmp_path / "sql", as_of="2021-02-01")
    overview = results["02_overview_kpis"].iloc[0]
    assert int(overview["purchase_invoices"]) == 5
    assert int(overview["purchase_customers"]) == 3
    assert float(overview["purchase_revenue"]) == 51
    assert int(overview["repeat_customers"]) == 2
    assert abs(float(overview["repeat_purchase_rate"]) - (2 / 3)) < 1e-4
    # 10 + 20 + 5 + 8 + 8 - 5 + 4, with the duplicate copy contributing 0.
    assert float(overview["net_line_revenue"]) == 50

    declining = results["11_declining_activity"]
    assert set(declining["customer_id"].astype(str)) == {"3"}
    assert int(declining.iloc[0]["prior_invoices"]) == 2
    assert int(declining.iloc[0]["recent_invoices"]) == 0

    sql_features = results["12_customer_features"].set_index("customer_id")
    python_features = features_for_snapshot(
        _fixture(),
        pd.Timestamp("2021-02-01"),
        horizon_days=30,
        min_history_invoices=1,
    ).set_index("customer_id")
    for customer_id in ("1", "2", "3"):
        assert int(sql_features.loc[customer_id, "frequency"]) == int(python_features.loc[customer_id, "frequency"])
        assert abs(float(sql_features.loc[customer_id, "monetary"]) - float(python_features.loc[customer_id, "monetary"])) < 0.02
        assert int(sql_features.loc[customer_id, "recency_days"]) == int(python_features.loc[customer_id, "recency_days"])

    customers = read_query(engine, "SELECT COUNT(*) AS n FROM customers")
    assert int(customers.iloc[0]["n"]) == 3
    products = results["08_top_products"]
    assert "11111" in set(products["stock_code"].astype(str))
    assert "WITH" in (SQLITE_SQL / "09_monthly_cohorts.sql").read_text(encoding="utf-8")
    assert set(ANALYTICS_FILES).issubset({path.name for path in SQLITE_SQL.glob("*.sql")})


def test_sql_files_demonstrate_the_required_constructs():
    required = ["WITH", "JOIN", "GROUP BY", "HAVING", "CASE", "ROW_NUMBER", "LAG", "NTILE"]
    for folder in (SQLITE_SQL, POSTGRES_SQL):
        text = "\n".join(path.read_text(encoding="utf-8") for path in folder.glob("*.sql"))
        for token in required:
            assert token in text, f"{folder.name} is missing {token}"
    postgres = (POSTGRES_SQL / "12_customer_features.sql").read_text(encoding="utf-8")
    sqlite = (SQLITE_SQL / "12_customer_features.sql").read_text(encoding="utf-8")
    assert "INTERVAL" in postgres
    assert "julianday" in sqlite
    assert "strftime" not in postgres
