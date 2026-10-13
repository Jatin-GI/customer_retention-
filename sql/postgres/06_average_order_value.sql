-- PostgreSQL average order value by latest country.
-- HAVING COUNT(*) >= 15 drops countries with a thin invoice history.

WITH invoices AS (
    SELECT
        customer_id,
        invoice_no,
        SUM(line_revenue) AS invoice_revenue,
        SUM(quantity) AS quantity
    FROM transactions
    WHERE is_purchase = 1
    GROUP BY customer_id, invoice_no
)
SELECT
    COALESCE(c.latest_country, 'Unknown') AS country,
    COUNT(*) AS purchase_invoices,
    COUNT(DISTINCT i.customer_id) AS customers,
    ROUND(AVG(i.invoice_revenue)::numeric, 2) AS average_order_value,
    ROUND(SUM(i.invoice_revenue)::numeric, 2) AS purchase_revenue,
    ROUND(AVG(i.quantity)::numeric, 2) AS average_quantity
FROM invoices AS i
JOIN customers AS c ON c.customer_id = i.customer_id
GROUP BY COALESCE(c.latest_country, 'Unknown')
HAVING COUNT(*) >= 15
ORDER BY purchase_revenue DESC;
