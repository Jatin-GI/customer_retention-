-- PostgreSQL top products. The dataset has no category column; this ranks stock codes.
-- ROW_NUMBER provides the revenue rank. The join adds the latest description.

WITH product_sales AS (
    SELECT
        t.stock_code,
        p.latest_description AS description,
        SUM(t.quantity) AS quantity_sold,
        ROUND(SUM(t.line_revenue)::numeric, 2) AS purchase_revenue,
        COUNT(DISTINCT t.invoice_no) AS invoices,
        COUNT(DISTINCT t.customer_id) AS customers
    FROM transactions AS t
    JOIN products AS p ON p.stock_code = t.stock_code
    WHERE t.is_purchase = 1
    GROUP BY t.stock_code, p.latest_description
),
ranked AS (
    SELECT
        product_sales.*,
        ROW_NUMBER() OVER (ORDER BY purchase_revenue DESC, stock_code) AS revenue_rank
    FROM product_sales
)
SELECT
    revenue_rank,
    stock_code,
    description,
    quantity_sold,
    purchase_revenue,
    invoices,
    customers
FROM ranked
WHERE revenue_rank <= 25
ORDER BY revenue_rank;
