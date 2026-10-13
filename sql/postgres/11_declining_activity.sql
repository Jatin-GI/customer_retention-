-- PostgreSQL declining purchase activity.
-- INTERVAL arithmetic replaces SQLite date(as_of, '-90 days').
-- HAVING requires at least two invoices in the prior 90-day window.

WITH bounds AS (
    SELECT MAX(invoice_date)::date AS as_of_date
    FROM transactions
    WHERE is_purchase = 1
),
invoices AS (
    SELECT
        t.customer_id,
        t.invoice_no,
        MIN(t.invoice_date)::date AS invoice_day,
        SUM(t.line_revenue) AS revenue
    FROM transactions AS t
    WHERE t.is_purchase = 1
    GROUP BY t.customer_id, t.invoice_no
)
SELECT
    i.customer_id,
    b.as_of_date,
    SUM(
        CASE
            WHEN i.invoice_day > (b.as_of_date - INTERVAL '90 days')::date
             AND i.invoice_day <= b.as_of_date THEN 1
            ELSE 0
        END
    ) AS recent_invoices,
    SUM(
        CASE
            WHEN i.invoice_day > (b.as_of_date - INTERVAL '180 days')::date
             AND i.invoice_day <= (b.as_of_date - INTERVAL '90 days')::date THEN 1
            ELSE 0
        END
    ) AS prior_invoices,
    ROUND(SUM(
        CASE
            WHEN i.invoice_day > (b.as_of_date - INTERVAL '90 days')::date
             AND i.invoice_day <= b.as_of_date THEN i.revenue
            ELSE 0
        END
    )::numeric, 2) AS recent_revenue,
    ROUND(SUM(
        CASE
            WHEN i.invoice_day > (b.as_of_date - INTERVAL '180 days')::date
             AND i.invoice_day <= (b.as_of_date - INTERVAL '90 days')::date THEN i.revenue
            ELSE 0
        END
    )::numeric, 2) AS prior_revenue
FROM invoices AS i
CROSS JOIN bounds AS b
GROUP BY i.customer_id, b.as_of_date
HAVING
    SUM(
        CASE
            WHEN i.invoice_day > (b.as_of_date - INTERVAL '180 days')::date
             AND i.invoice_day <= (b.as_of_date - INTERVAL '90 days')::date THEN 1
            ELSE 0
        END
    ) >= 2
    AND SUM(
        CASE
            WHEN i.invoice_day > (b.as_of_date - INTERVAL '90 days')::date
             AND i.invoice_day <= b.as_of_date THEN 1
            ELSE 0
        END
    ) < SUM(
        CASE
            WHEN i.invoice_day > (b.as_of_date - INTERVAL '180 days')::date
             AND i.invoice_day <= (b.as_of_date - INTERVAL '90 days')::date THEN 1
            ELSE 0
        END
    )
ORDER BY prior_invoices - recent_invoices DESC, prior_revenue DESC;
