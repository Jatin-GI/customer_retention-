-- Monthly purchase invoices, active customers, revenue, and a running revenue total.
-- LAG compares each month with the previous month. NULLIF avoids division by zero.

WITH monthly AS (
    SELECT
        strftime('%Y-%m', invoice_date) AS year_month,
        COUNT(DISTINCT invoice_no) AS purchase_invoices,
        COUNT(DISTINCT customer_id) AS active_customers,
        ROUND(SUM(line_revenue), 2) AS purchase_revenue
    FROM transactions
    WHERE is_purchase = 1
    GROUP BY strftime('%Y-%m', invoice_date)
)
SELECT
    year_month,
    purchase_invoices,
    active_customers,
    purchase_revenue,
    ROUND(purchase_revenue / NULLIF(purchase_invoices, 0), 2) AS average_order_value,
    ROUND(SUM(purchase_revenue) OVER (ORDER BY year_month), 2) AS running_revenue,
    LAG(purchase_revenue) OVER (ORDER BY year_month) AS prior_month_revenue,
    ROUND(
        (purchase_revenue - LAG(purchase_revenue) OVER (ORDER BY year_month))
        / NULLIF(LAG(purchase_revenue) OVER (ORDER BY year_month), 0),
        4
    ) AS revenue_change_ratio
FROM monthly
ORDER BY year_month;
