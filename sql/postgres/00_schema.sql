-- PostgreSQL schema. Not executed in the environment used to build this project,
-- because PostgreSQL was not installed. The SQLite scripts in sql/sqlite were executed.
-- Flags stay INTEGER 0/1 so the Python loader can insert the same frame.
-- invoice_date is TIMESTAMP. Date logic uses ::date and INTERVAL, not SQLite date().

DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    line_id INTEGER PRIMARY KEY,
    invoice_no TEXT NOT NULL,
    stock_code TEXT,
    description TEXT,
    quantity INTEGER,
    invoice_date TIMESTAMP NOT NULL,
    unit_price DOUBLE PRECISION,
    customer_id TEXT,
    country TEXT,
    is_cancellation INTEGER NOT NULL,
    is_return INTEGER NOT NULL,
    invalid_price INTEGER NOT NULL,
    is_exact_duplicate INTEGER NOT NULL,
    is_non_merchandise INTEGER NOT NULL,
    missing_customer INTEGER NOT NULL,
    is_purchase INTEGER NOT NULL,
    line_revenue DOUBLE PRECISION NOT NULL
);
