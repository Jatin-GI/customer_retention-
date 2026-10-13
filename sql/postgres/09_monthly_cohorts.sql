-- PostgreSQL monthly cohorts.
-- Year-month indexes use EXTRACT instead of SQLite strftime.

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
        to_char(MIN(invoice_date), 'YYYY-MM') AS cohort_month,
        EXTRACT(YEAR FROM MIN(invoice_date))::int * 12
            + EXTRACT(MONTH FROM MIN(invoice_date))::int AS cohort_index
    FROM purchases
    GROUP BY customer_id
),
activity AS (
    SELECT DISTINCT
        p.customer_id,
        to_char(p.invoice_date, 'YYYY-MM') AS activity_month,
        EXTRACT(YEAR FROM p.invoice_date)::int * 12
            + EXTRACT(MONTH FROM p.invoice_date)::int AS activity_index
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
    ROUND((COUNT(DISTINCT a.customer_id)::numeric / s.cohort_customers), 4) AS retention_rate
FROM firsts AS f
JOIN activity AS a ON a.customer_id = f.customer_id
JOIN cohort_size AS s ON s.cohort_month = f.cohort_month
GROUP BY f.cohort_month, a.activity_index - f.cohort_index, s.cohort_customers
ORDER BY f.cohort_month, month_offset;
