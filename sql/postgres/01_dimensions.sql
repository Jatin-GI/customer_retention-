-- PostgreSQL dimensions and indexes. See sql/sqlite/01_dimensions.sql for the shared logic.
-- Window functions are the same. This file was not executed here.

DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;

CREATE TABLE customers AS
WITH ranked AS (
    SELECT
        customer_id,
        country,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY invoice_date DESC, invoice_no DESC
        ) AS rn
    FROM transactions
    WHERE customer_id IS NOT NULL
      AND is_exact_duplicate = 0
)
SELECT customer_id, country AS latest_country
FROM ranked
WHERE rn = 1;

CREATE TABLE products AS
WITH ranked AS (
    SELECT
        stock_code,
        description,
        ROW_NUMBER() OVER (
            PARTITION BY stock_code
            ORDER BY
                CASE WHEN description IS NULL OR description = '' THEN 1 ELSE 0 END,
                invoice_date DESC
        ) AS rn
    FROM transactions
    WHERE is_exact_duplicate = 0
      AND stock_code IS NOT NULL
      AND stock_code <> ''
)
SELECT stock_code, description AS latest_description
FROM ranked
WHERE rn = 1;

CREATE INDEX idx_tx_customer ON transactions(customer_id);
CREATE INDEX idx_tx_invoice_date ON transactions(invoice_date);
CREATE INDEX idx_tx_invoice ON transactions(invoice_no);
CREATE INDEX idx_tx_purchase ON transactions(is_purchase);
CREATE INDEX idx_customers_id ON customers(customer_id);
CREATE INDEX idx_products_code ON products(stock_code);
