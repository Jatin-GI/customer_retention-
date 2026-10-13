-- Headline counts and revenue under the documented purchase definition.
-- purchase_revenue sums purchase lines only.
-- net_line_revenue keeps the sign of returns and cancellations and ignores duplicate copies.
-- repeat_purchase_rate is customers with 2+ purchase invoices divided by customers with at least one.

WITH purchase_invoices AS (
    SELECT
        customer_id,
        invoice_no,
        SUM(line_revenue) AS invoice_revenue
    FROM transactions
    WHERE is_purchase = 1
    GROUP BY customer_id, invoice_no
),
customer_counts AS (
    SELECT customer_id, COUNT(*) AS n_invoices
    FROM purchase_invoices
    GROUP BY customer_id
)
SELECT
    (SELECT COUNT(*) FROM transactions) AS line_items,
    (SELECT COUNT(*) FROM transactions WHERE is_purchase = 1) AS purchase_lines,
    (SELECT COUNT(*) FROM purchase_invoices) AS purchase_invoices,
    (SELECT COUNT(*) FROM customer_counts) AS purchase_customers,
    COALESCE((SELECT ROUND(SUM(invoice_revenue), 2) FROM purchase_invoices), 0) AS purchase_revenue,
    COALESCE((
        SELECT ROUND(SUM(line_revenue), 2)
        FROM transactions
        WHERE is_exact_duplicate = 0
    ), 0) AS net_line_revenue,
    (SELECT COUNT(*) FROM transactions WHERE missing_customer = 1) AS rows_missing_customer,
    (SELECT COUNT(*) FROM transactions WHERE is_cancellation = 1) AS cancellation_rows,
    (SELECT COUNT(*) FROM transactions WHERE is_return = 1) AS return_rows,
    COALESCE((
        SELECT SUM(CASE WHEN n_invoices >= 2 THEN 1 ELSE 0 END)
        FROM customer_counts
    ), 0) AS repeat_customers,
    COALESCE((
        SELECT ROUND(AVG(CASE WHEN n_invoices >= 2 THEN 1.0 ELSE 0.0 END), 4)
        FROM customer_counts
    ), 0) AS repeat_purchase_rate,
    COALESCE((
        SELECT ROUND(SUM(invoice_revenue) / NULLIF(COUNT(*), 0), 2)
        FROM purchase_invoices
    ), 0) AS average_order_value;
