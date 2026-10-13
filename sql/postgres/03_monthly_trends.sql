-- PostgreSQL monthly trends. to_char replaces SQLite strftime.
-- Running SUM and LAG are the same window-function pattern.

WITH monthly AS (
    SELECT
        to_char(invoice_date, 'YYYY-MM') AS year_month,
        COUNT(DISTINCT invoice_no) AS purchase_invoices,
        COUNT(DISTINCT customer_id) AS active_customers,
        ROUND(SUM(line_revenue)::numeric, 2) AS purchase_revenue
    FROM transactions
    WHERE is_purchase = 1
    GROUP BY to_char(invoice_date, 'YYYY-MM')
)
SELECT
    year_month,
    purchase_invoices,
    active_customers,
    purchase_revenue,
    ROUND((purchase_revenue / NULLIF(purchase_invoices, 0))::numeric, 2) AS average_order_value,
    ROUND(SUM(purchase_revenue) OVER (ORDER BY year_month)::numeric, 2) AS running_revenue,
    LAG(purchase_revenue) OVER (ORDER BY year_month) AS prior_month_revenue,
    ROUND(
        (
            (purchase_revenue - LAG(purchase_revenue) OVER (ORDER BY year_month))
            / NULLIF(LAG(purchase_revenue) OVER (ORDER BY year_month), 0)
        )::numeric,
        4
    ) AS revenue_change_ratio
FROM monthly
ORDER BY year_month;
