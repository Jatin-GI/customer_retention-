-- Average order value by the customer's latest country.
-- JOIN demonstrates the customer dimension. HAVING keeps countries with at least 15 purchase invoices
-- so a handful of orders cannot dominate the comparison. The global average is on the overview query.

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
    ROUND(AVG(i.invoice_revenue), 2) AS average_order_value,
    ROUND(SUM(i.invoice_revenue), 2) AS purchase_revenue,
    ROUND(AVG(i.quantity), 2) AS average_quantity
FROM invoices AS i
JOIN customers AS c ON c.customer_id = i.customer_id
GROUP BY COALESCE(c.latest_country, 'Unknown')
HAVING COUNT(*) >= 15
ORDER BY purchase_revenue DESC;
