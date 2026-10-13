-- Monthly acquisition cohorts and the share of each cohort that purchases again in later months.
-- cohort_month is the month of the customer's first purchase invoice.
-- month_offset 0 is the acquisition month, so its retention rate should be 1.

WITH purchases AS (
    SELECT
        customer_id,
        invoice_no,
        MIN(invoice_date) AS invoice_date
    FROM transactions
    WHERE is_purchase = 1
    GROUP BY customer_id, invoice_no
),
firsts AS (
    SELECT
        customer_id,
        strftime('%Y-%m', MIN(invoice_date)) AS cohort_month,
        CAST(strftime('%Y', MIN(invoice_date)) AS INTEGER) * 12
            + CAST(strftime('%m', MIN(invoice_date)) AS INTEGER) AS cohort_index
    FROM purchases
    GROUP BY customer_id
),
activity AS (
    SELECT DISTINCT
        p.customer_id,
        strftime('%Y-%m', p.invoice_date) AS activity_month,
        CAST(strftime('%Y', p.invoice_date) AS INTEGER) * 12
            + CAST(strftime('%m', p.invoice_date) AS INTEGER) AS activity_index
    FROM purchases AS p
),
cohort_size AS (
    SELECT cohort_month, cohort_index, COUNT(*) AS cohort_customers
    FROM firsts
    GROUP BY cohort_month, cohort_index
)
SELECT
    f.cohort_month,
    a.activity_index - f.cohort_index AS month_offset,
    s.cohort_customers,
    COUNT(DISTINCT a.customer_id) AS active_customers,
    ROUND(COUNT(DISTINCT a.customer_id) * 1.0 / s.cohort_customers, 4) AS retention_rate
FROM firsts AS f
JOIN activity AS a ON a.customer_id = f.customer_id
JOIN cohort_size AS s ON s.cohort_month = f.cohort_month
GROUP BY f.cohort_month, a.activity_index - f.cohort_index, s.cohort_customers
ORDER BY f.cohort_month, month_offset;
