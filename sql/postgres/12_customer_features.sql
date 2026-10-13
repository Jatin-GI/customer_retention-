-- PostgreSQL customer features for one cutoff date, bound by the runner as YYYY-MM-DD.
-- History filter: invoice_date::date <= CAST of that bound date.
-- This is the single-cutoff counterpart of the Python feature builder.

WITH invoices AS (
    SELECT
        customer_id,
        invoice_no,
        MIN(invoice_date) AS invoice_date,
        SUM(line_revenue) AS revenue,
        SUM(quantity) AS quantity
    FROM transactions
    WHERE is_purchase = 1
      AND invoice_date::date <= CAST(:as_of AS date)
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
      AND invoice_date::date <= CAST(:as_of AS date)
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
      AND invoice_date::date <= CAST(:as_of AS date)
    GROUP BY customer_id
)
SELECT
    o.customer_id,
    CAST(:as_of AS date) AS snapshot_date,
    (CAST(:as_of AS date) - MAX(o.invoice_date)::date) AS recency_days,
    COUNT(*) AS frequency,
    ROUND(SUM(o.revenue)::numeric, 2) AS monetary,
    ROUND((SUM(o.revenue) / COUNT(*))::numeric, 2) AS avg_order_value,
    ROUND(AVG(o.quantity)::numeric, 4) AS avg_quantity,
    pl.n_distinct_products,
    ROUND(AVG(
        CASE
            WHEN o.prev_invoice_date IS NOT NULL
            THEN EXTRACT(EPOCH FROM (o.invoice_date - o.prev_invoice_date)) / 86400.0
        END
    )::numeric, 4) AS avg_interpurchase_days,
    (CAST(:as_of AS date) - MIN(o.invoice_date)::date) AS tenure_days,
    ROUND(SUM(CASE
        WHEN o.invoice_date::date > (CAST(:as_of AS date) - INTERVAL '30 days')::date THEN o.revenue
        ELSE 0
    END)::numeric, 2) AS spend_30d,
    ROUND(SUM(CASE
        WHEN o.invoice_date::date > (CAST(:as_of AS date) - INTERVAL '90 days')::date THEN o.revenue
        ELSE 0
    END)::numeric, 2) AS spend_90d,
    SUM(CASE
        WHEN o.invoice_date::date > (CAST(:as_of AS date) - INTERVAL '30 days')::date THEN 1
        ELSE 0
    END) AS invoices_30d,
    SUM(CASE
        WHEN o.invoice_date::date > (CAST(:as_of AS date) - INTERVAL '90 days')::date THEN 1
        ELSE 0
    END) AS invoices_90d,
    ROUND((
        SUM(CASE
            WHEN o.invoice_date::date > (CAST(:as_of AS date) - INTERVAL '90 days')::date THEN o.revenue
            ELSE 0
        END)
        / NULLIF(SUM(CASE
            WHEN o.invoice_date::date <= (CAST(:as_of AS date) - INTERVAL '90 days')::date
             AND o.invoice_date::date > (CAST(:as_of AS date) - INTERVAL '180 days')::date
            THEN o.revenue ELSE 0
        END), 0)
    )::numeric, 4) AS spend_trend_ratio,
    ROUND((SUM(o.revenue) / NULLIF(SUM(o.quantity), 0))::numeric, 4) AS avg_unit_price,
    ROUND((
        COALESCE(r.return_lines, 0)::numeric
        / NULLIF(COALESCE(r.return_lines, 0) + pl.purchase_lines, 0)
    ), 4) AS return_rate,
    ROUND((
        SUM(o.revenue) / ((CAST(:as_of AS date) - MIN(o.invoice_date)::date) + 1)
    )::numeric, 4) AS spend_per_tenure_day
FROM ordered AS o
JOIN purchase_lines AS pl ON pl.customer_id = o.customer_id
LEFT JOIN returns AS r ON r.customer_id = o.customer_id
GROUP BY o.customer_id, pl.n_distinct_products, pl.purchase_lines, r.return_lines, CAST(:as_of AS date)
HAVING COUNT(*) >= 1
ORDER BY monetary DESC;
