-- PostgreSQL revenue by RFM segment. CASE rules match src/config.py and the SQLite query.
-- NTILE boundaries can differ from the pandas quintiles on the segmentation page.

WITH as_of AS (
    SELECT MAX(invoice_date)::date AS as_of_date
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
        ((SELECT as_of_date FROM as_of) - MAX(i.invoice_date)::date) AS recency_days,
        COUNT(*) AS frequency,
        SUM(i.revenue) AS monetary
    FROM invoices AS i
    GROUP BY i.customer_id
),
scored AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency_days DESC, customer_id) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC, customer_id) AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC, customer_id) AS m_score
    FROM rfm
),
labeled AS (
    SELECT
        customer_id,
        monetary,
        recency_days,
        frequency,
        CASE
            WHEN r_score >= 4 AND f_score >= 4 THEN 'Recent and frequent'
            WHEN m_score >= 4 AND f_score >= 3 THEN 'High value'
            WHEN r_score >= 4 AND f_score <= 2 THEN 'Promising'
            WHEN r_score <= 2 AND f_score >= 3 THEN 'Needs re-engagement'
            WHEN r_score <= 2 AND f_score <= 2 THEN 'Inactive'
            ELSE 'Regular'
        END AS segment
    FROM scored
)
SELECT
    segment,
    COUNT(*) AS customers,
    ROUND(SUM(monetary)::numeric, 2) AS purchase_revenue,
    ROUND(AVG(recency_days)::numeric, 1) AS avg_recency_days,
    ROUND(AVG(frequency)::numeric, 2) AS avg_frequency,
    ROUND(AVG(monetary)::numeric, 2) AS avg_monetary,
    ROUND((SUM(monetary) / NULLIF((SELECT SUM(monetary) FROM labeled), 0))::numeric, 4) AS revenue_share,
    ROUND((COUNT(*)::numeric / NULLIF((SELECT COUNT(*) FROM labeled), 0)), 4) AS customer_share
FROM labeled
GROUP BY segment
ORDER BY purchase_revenue DESC;
