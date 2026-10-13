-- Customers whose purchase-invoice count in the latest 90 days is below the prior 90 days.
-- HAVING requires a real prior history (at least two invoices) so a single old order is not called a decline.
-- The windows are measured from the latest purchase date in the table, not from today.

WITH bounds AS (
    SELECT date(MAX(invoice_date)) AS as_of_date
    FROM transactions
    WHERE is_purchase = 1
),
invoices AS (
    SELECT
        t.customer_id,
        t.invoice_no,
        date(MIN(t.invoice_date)) AS invoice_day,
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
            WHEN i.invoice_day > date(b.as_of_date, '-90 days') AND i.invoice_day <= b.as_of_date THEN 1
            ELSE 0
        END
    ) AS recent_invoices,
    SUM(
        CASE
            WHEN i.invoice_day > date(b.as_of_date, '-180 days')
             AND i.invoice_day <= date(b.as_of_date, '-90 days') THEN 1
            ELSE 0
        END
    ) AS prior_invoices,
    ROUND(SUM(
        CASE
            WHEN i.invoice_day > date(b.as_of_date, '-90 days') AND i.invoice_day <= b.as_of_date THEN i.revenue
            ELSE 0
        END
    ), 2) AS recent_revenue,
    ROUND(SUM(
        CASE
            WHEN i.invoice_day > date(b.as_of_date, '-180 days')
             AND i.invoice_day <= date(b.as_of_date, '-90 days') THEN i.revenue
            ELSE 0
        END
    ), 2) AS prior_revenue
FROM invoices AS i
CROSS JOIN bounds AS b
GROUP BY i.customer_id, b.as_of_date
HAVING
    SUM(
        CASE
            WHEN i.invoice_day > date(b.as_of_date, '-180 days')
             AND i.invoice_day <= date(b.as_of_date, '-90 days') THEN 1
            ELSE 0
        END
    ) >= 2
    AND SUM(
        CASE
            WHEN i.invoice_day > date(b.as_of_date, '-90 days') AND i.invoice_day <= b.as_of_date THEN 1
            ELSE 0
        END
    ) < SUM(
        CASE
            WHEN i.invoice_day > date(b.as_of_date, '-180 days')
             AND i.invoice_day <= date(b.as_of_date, '-90 days') THEN 1
            ELSE 0
        END
    )
ORDER BY prior_invoices - recent_invoices DESC, prior_revenue DESC;
