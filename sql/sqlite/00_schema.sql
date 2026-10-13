-- SQLite schema. Flags are integers 0/1 because SQLite has no separate boolean type.
-- invoice_date is stored as ISO text YYYY-MM-DD HH:MM:SS so date() and strftime() apply.
-- PostgreSQL uses TIMESTAMP and lives in sql/postgres/00_schema.sql.

DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    line_id INTEGER PRIMARY KEY,
    invoice_no TEXT NOT NULL,
    stock_code TEXT,
    description TEXT,
    quantity INTEGER,
    invoice_date TEXT NOT NULL,
    unit_price REAL,
    customer_id TEXT,
    country TEXT,
    is_cancellation INTEGER NOT NULL,
    is_return INTEGER NOT NULL,
    invalid_price INTEGER NOT NULL,
    is_exact_duplicate INTEGER NOT NULL,
    is_non_merchandise INTEGER NOT NULL,
    missing_customer INTEGER NOT NULL,
    is_purchase INTEGER NOT NULL,
    line_revenue REAL NOT NULL
);
