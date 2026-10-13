-- One row per purchasing customer: frequency, revenue, and first/last purchase.
-- Customers with no purchase lines are absent because they have no customer-level history.

SELECT
    customer_id,
    COUNT(DISTINCT invoice_no) AS n_invoices,
    COUNT(*) AS purchase_lines,
    ROUND(SUM(line_revenue), 2) AS purchase_revenue,
    ROUND(SUM(line_revenue) * 1.0 / COUNT(DISTINCT invoice_no), 2) AS average_order_value,
    MIN(invoice_date) AS first_purchase,
    MAX(invoice_date) AS last_purchase
FROM transactions
WHERE is_purchase = 1
GROUP BY customer_id
ORDER BY purchase_revenue DESC;
