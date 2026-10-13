-- Top merchandise by purchase revenue.
-- The source data has Description and StockCode, not a category hierarchy.
-- ROW_NUMBER ranks products. The join attaches the latest non-empty description.
-- Cancellations are excluded rather than netted, so a sale that is later reversed on a
-- different invoice can remain in this ranking. Inspect very large single-invoice rows.

WITH product_sales AS (
    SELECT
        t.stock_code,
        p.latest_description AS description,
        SUM(t.quantity) AS quantity_sold,
        ROUND(SUM(t.line_revenue), 2) AS purchase_revenue,
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
