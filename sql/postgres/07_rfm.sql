-- PostgreSQL RFM scores. Date subtraction replaces julianday.
-- NTILE(5): score 5 is the favorable end of recency, frequency, and monetary value.

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
        ROUND(SUM(i.revenue)::numeric, 2) AS monetary
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
