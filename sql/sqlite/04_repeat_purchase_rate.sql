-- How many purchase invoices each customer has, banded for the repeat-purchase distribution.
-- The rate itself is also on the overview query. One invoice means a one-time purchaser.

WITH invoice_counts AS (
    SELECT
        customer_id,
        COUNT(DISTINCT invoice_no) AS n_invoices,
        SUM(line_revenue) AS revenue
    FROM transactions
    WHERE is_purchase = 1
    GROUP BY customer_id
),
banded AS (
    SELECT
        CASE
            WHEN n_invoices = 1 THEN '1'
            WHEN n_invoices = 2 THEN '2'
            WHEN n_invoices BETWEEN 3 AND 5 THEN '3-5'
            WHEN n_invoices BETWEEN 6 AND 10 THEN '6-10'
            ELSE '11+'
        END AS invoice_band,
        CASE
            WHEN n_invoices = 1 THEN 1
            WHEN n_invoices = 2 THEN 2
            WHEN n_invoices BETWEEN 3 AND 5 THEN 3
            WHEN n_invoices BETWEEN 6 AND 10 THEN 4
            ELSE 5
        END AS band_order,
        n_invoices,
        revenue
    FROM invoice_counts
)
SELECT
    invoice_band,
    COUNT(*) AS customers,
    ROUND(SUM(revenue), 2) AS purchase_revenue,
    ROUND(AVG(n_invoices), 2) AS avg_invoices_in_band
FROM banded
GROUP BY invoice_band, band_order
ORDER BY band_order;
