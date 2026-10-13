-- Customer RFM as of the latest purchase date in the table.
-- Scores are NTILE(5) buckets. 5 is the favorable end: more recent, more frequent, higher monetary.
-- NTILE splits tied rows across adjacent buckets, so scores are descriptive ranks, not stable business labels.
-- This dataset has no product-category column. Do not treat description text as a category.

WITH as_of AS (
    SELECT date(MAX(invoice_date)) AS as_of_date
    FROM transactions
    WHERE is_purchase = 1
),
invoices AS (
    SELECT
        customer_id,
        invoice_no,
        MIN(invoice_date) AS invoice_date,
        SUM(line_revenue) AS revenue
    FROM transactions
    WHERE is_purchase = 1
    GROUP BY customer_id, invoice_no
),
rfm AS (
    SELECT
        i.customer_id,
        CAST(
            julianday((SELECT as_of_date FROM as_of)) - julianday(date(MAX(i.invoice_date)))
            AS INTEGER
        ) AS recency_days,
        COUNT(*) AS frequency,
        ROUND(SUM(i.revenue), 2) AS monetary
    FROM invoices AS i
    GROUP BY i.customer_id
)
SELECT
    customer_id,
    recency_days,
    frequency,
    monetary,
    (SELECT as_of_date FROM as_of) AS as_of_date,
    NTILE(5) OVER (ORDER BY recency_days DESC, customer_id) AS r_score,
    NTILE(5) OVER (ORDER BY frequency ASC, customer_id) AS f_score,
    NTILE(5) OVER (ORDER BY monetary ASC, customer_id) AS m_score
FROM rfm
ORDER BY monetary DESC;
