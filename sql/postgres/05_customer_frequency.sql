-- PostgreSQL customer purchase frequency. One row per customer with at least one purchase.

SELECT
    customer_id,
    COUNT(DISTINCT invoice_no) AS n_invoices,
    COUNT(*) AS purchase_lines,
    ROUND(SUM(line_revenue)::numeric, 2) AS purchase_revenue,
    ROUND((SUM(line_revenue) / COUNT(DISTINCT invoice_no))::numeric, 2) AS average_order_value,
    MIN(invoice_date) AS first_purchase,
    MAX(invoice_date) AS last_purchase
FROM transactions
WHERE is_purchase = 1
GROUP BY customer_id
ORDER BY purchase_revenue DESC;
