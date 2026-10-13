-- Customer features available on one calendar cutoff.
-- History is invoice_date on or before that cutoff. Later invoices are not aggregated.
-- Definitions match the Python feature builder for the shared columns.
-- The runner binds the cutoff as YYYY-MM-DD. The supervised model uses the Python builder
-- because it repeats this logic at several snapshot dates.

WITH invoices AS (
    SELECT
        customer_id,
        invoice_no,
        MIN(invoice_date) AS invoice_date,
        SUM(line_revenue) AS revenue,
        SUM(quantity) AS quantity
    FROM transactions
    WHERE is_purchase = 1
      AND date(invoice_date) <= date(:as_of)
    GROUP BY customer_id, invoice_no
),
ordered AS (
    SELECT
        customer_id,
        invoice_no,
        invoice_date,
        revenue,
        quantity,
        LAG(invoice_date) OVER (
            PARTITION BY customer_id
            ORDER BY invoice_date, invoice_no
        ) AS prev_invoice_date
    FROM invoices
),
returns AS (
    SELECT customer_id, COUNT(*) AS return_lines
    FROM transactions
    WHERE customer_id IS NOT NULL
      AND is_exact_duplicate = 0
      AND date(invoice_date) <= date(:as_of)
      AND (is_cancellation = 1 OR is_return = 1)
    GROUP BY customer_id
),
purchase_lines AS (
    SELECT
        customer_id,
        COUNT(*) AS purchase_lines,
        COUNT(DISTINCT stock_code) AS n_distinct_products
    FROM transactions
    WHERE is_purchase = 1
      AND date(invoice_date) <= date(:as_of)
    GROUP BY customer_id
)
SELECT
    o.customer_id,
    date(:as_of) AS snapshot_date,
    CAST(julianday(date(:as_of)) - julianday(date(MAX(o.invoice_date))) AS INTEGER) AS recency_days,
    COUNT(*) AS frequency,
    ROUND(SUM(o.revenue), 2) AS monetary,
    ROUND(SUM(o.revenue) / COUNT(*), 2) AS avg_order_value,
    ROUND(AVG(o.quantity), 4) AS avg_quantity,
    pl.n_distinct_products,
    ROUND(AVG(
        CASE
            WHEN o.prev_invoice_date IS NOT NULL
            THEN julianday(o.invoice_date) - julianday(o.prev_invoice_date)
        END
    ), 4) AS avg_interpurchase_days,
    CAST(julianday(date(:as_of)) - julianday(date(MIN(o.invoice_date))) AS INTEGER) AS tenure_days,
    ROUND(SUM(CASE WHEN date(o.invoice_date) > date(:as_of, '-30 days') THEN o.revenue ELSE 0 END), 2) AS spend_30d,
    ROUND(SUM(CASE WHEN date(o.invoice_date) > date(:as_of, '-90 days') THEN o.revenue ELSE 0 END), 2) AS spend_90d,
    SUM(CASE WHEN date(o.invoice_date) > date(:as_of, '-30 days') THEN 1 ELSE 0 END) AS invoices_30d,
    SUM(CASE WHEN date(o.invoice_date) > date(:as_of, '-90 days') THEN 1 ELSE 0 END) AS invoices_90d,
    ROUND(
        SUM(CASE WHEN date(o.invoice_date) > date(:as_of, '-90 days') THEN o.revenue ELSE 0 END)
        / NULLIF(SUM(
            CASE
                WHEN date(o.invoice_date) <= date(:as_of, '-90 days')
                 AND date(o.invoice_date) > date(:as_of, '-180 days')
                THEN o.revenue ELSE 0
            END
        ), 0),
        4
    ) AS spend_trend_ratio,
    ROUND(SUM(o.revenue) / NULLIF(SUM(o.quantity), 0), 4) AS avg_unit_price,
    ROUND(
        COALESCE(r.return_lines, 0) * 1.0
        / NULLIF(COALESCE(r.return_lines, 0) + pl.purchase_lines, 0),
        4
    ) AS return_rate,
    ROUND(SUM(o.revenue) / (CAST(julianday(date(:as_of)) - julianday(date(MIN(o.invoice_date))) AS INTEGER) + 1), 4)
        AS spend_per_tenure_day
FROM ordered AS o
JOIN purchase_lines AS pl ON pl.customer_id = o.customer_id
LEFT JOIN returns AS r ON r.customer_id = o.customer_id
GROUP BY o.customer_id, pl.n_distinct_products, pl.purchase_lines, r.return_lines
HAVING COUNT(*) >= 1
ORDER BY monetary DESC;
