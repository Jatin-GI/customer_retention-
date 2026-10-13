# Customer Retention Intelligence

Behavioral analytics and 30-day repeat-purchase prediction on the public UCI Online Retail dataset.

This is a portfolio simulation for data science and machine learning interviews. It does not use AstroTalk data, and none of the numbers below are findings about AstroTalk customers.

## Business problem

A consumer retailer wants to know which customers are likely to buy again, which groups produce the revenue, and which behavioral signals are associated with a later purchase. The project answers that with a documented purchase definition, SQL summaries, a time-based classifier, and a dashboard that reads those outputs.

The practical question for the model is: given only what was known on a cutoff date, will this customer have at least one merchandise purchase in the next 30 days?

## Dataset and license

| Item | Value |
| --- | --- |
| Dataset | [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), UCI Machine Learning Repository |
| Citation | Chen, D. (2015). Online Retail. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33 |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the source above if you share the data. The project code is MIT; the dataset is not. |
| What it is | Invoices from a UK-based online gift retailer, 1 December 2010 through 9 December 2011. Many buyers are wholesalers. |
| Columns | `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country` |
| File used | `Online Retail.xlsx`, SHA-256 `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d` |

The UCI page lists 541,909 instances. The file read by this project has that shape. There is no product-category column. Stock code and description are the product fields.

## Problem formulation

A snapshot date is an inclusive history cutoff. Purchases on that calendar day are features. The label window is the next 30 days, starting the day after the snapshot.

| Choice | Definition used in this run |
| --- | --- |
| Horizon | 30 days (`PREDICTION_HORIZON_DAYS`) |
| Warmup | 90 days of history before the first snapshot |
| Eligible customer | At least one merchandise purchase invoice on or before the snapshot |
| Positive label | At least one purchase invoice in the 30 days after the snapshot |
| Negative label | No purchase invoice in that window. The customer is kept. |
| Incomplete horizons | A snapshot is used only when the full 30-day window ends before the last observed calendar day. The file ends mid-day on 9 December 2011, so the last snapshot is 1 November 2011. |
| Features | Computed only from lines before the next midnight after the snapshot. Customer and invoice IDs are not features. |
| Split | Train: 1 Mar 2011–1 Sep 2011. Validation: 1 Oct 2011. Test: 1 Nov 2011. Hyperparameters use an inner holdout of the 1 Sep 2011 training snapshot. The test snapshot is not used to choose the model or the threshold. |

Positive rates from `reports/modeling_metadata.json`: train 27.0%, validation 27.8%, test 33.8% (18,446 / 3,613 / 3,981 customer-snapshot rows, 26,040 rows in total). A leakage audit recomputed monetary value, frequency, and the label for 40 sampled rows and passed.

## Architecture

```mermaid
flowchart LR
  A[UCI Online Retail xlsx] --> B[Ingestion and schema check]
  B --> C[Cleaning and quality report]
  C --> D[(SQLite or PostgreSQL)]
  D --> E[SQL analytics]
  C --> F[Snapshot features and labels]
  F --> G[Train, validation, test]
  G --> H[Metrics, SHAP, RFM segments]
  E --> I[Streamlit dashboard]
  H --> I
```

| Path | Role |
| --- | --- |
| `src/data_ingestion.py` | Find a local file, extract a local zip, or download the UCI archive |
| `src/data_cleaning.py` | Validate, flag, and define a purchase line |
| `src/database.py`, `src/sql_runner.py` | Load tables and run the SQL pack |
| `sql/sqlite`, `sql/postgres` | The same analyses in each dialect |
| `src/target_creation.py`, `src/feature_engineering.py` | Cutoffs, labels, and pre-cutoff features |
| `src/train.py`, `src/evaluate.py` | Models, threshold, curves |
| `src/explainability.py` | Permutation importance and SHAP |
| `src/segmentation.py` | Ordered RFM rules |
| `app.py`, `pages/` | Dashboard over saved outputs |
| `tests/` | Unit tests on synthetic fixtures |

## SQL analysis

The development machine did not have PostgreSQL, so every query in `sql/sqlite/` was executed against a local SQLite database. `sql/postgres/` is the same logic with `TIMESTAMP`, `to_char`, `EXTRACT`, and `INTERVAL` instead of `strftime`, `date()`, and `julianday`. Those PostgreSQL scripts were not executed here.

The queries use joins, `GROUP BY`, `HAVING`, `CASE`, CTEs, `ROW_NUMBER`, `LAG`, `NTILE`, running sums, and date filters. Results are written to `reports/sql/` and shown in the dashboard. The SQL page can re-query monthly revenue for a selected month range.

There is no category table to join. Top products are stock codes, not invented categories.

## Preprocessing decisions

All 541,909 source rows were retained. Nothing was dropped for a bad date or a missing invoice number. Flags record the rows that are excluded from the purchase definition.

A purchase line has a customer ID, a parsed date, a positive quantity, a positive unit price, an invoice number that does not start with `C`, a merchandise stock code, and is the first copy of an exact duplicate group. Merchandise means the stock code contains a digit and is not a known fee code (`POST`, `DOT`, `M`, `D`, `C2`, and the rest of the set in `src/config.py`).

| Rule | Rows | Treatment |
| --- | --- | --- |
| Missing `CustomerID` | 135,080 (24.9%) | Kept. Excluded from customer revenue, RFM, and the model. Positive anonymous merchandise revenue left out of the purchase total: £1,510,382.51. |
| Cancellation invoice, or negative quantity | 9,288 cancellations; 10,624 negative-quantity rows | Kept and flagged. Excluded from purchases. Not subtracted from the original sale. |
| Non-positive unit price | 2,517 | Kept. Excluded from purchase revenue. |
| Exact duplicate | 5,268 | Kept. Revenue on the extra copies set to 0. Suppressed duplicate revenue: £21,740.98. |
| Non-merchandise stock code | 2,940 | Kept. Excluded from purchase revenue and product counts. |
| Missing description | 1,454 | Kept. |

**Purchase revenue under that definition: £8,737,227.64** across 18,402 invoices and 4,334 customers. Average order value: £474.80. Customers with two or more purchase invoices: 2,829, a repeat-purchase rate of 65.3%. Those repeat customers account for 92.8% of purchase revenue. One-time customers are 1,505 people and 7.2% of purchase revenue.

`net_line_revenue` in the overview query is a different number (£9,726,006.95). It sums every non-duplicate line, including anonymous checkouts and the signed value of returns. Do not quote it as the purchase total.

Cancellations are not matched back to the original invoice. Invoice `581483` is one line of 80,995 units of stock `23843` ("PAPER CRAFT , LITTLE BIRDIE") at £2.08, or £168,469.60, for customer 16446. Invoice `C581484` reverses that exact line. The sale stays inside purchase revenue and is rank 1 in the product query because it is one invoice. The cancellation is excluded rather than netted. The next product by revenue, `22423` (REGENCY CAKESTAND 3 TIER), is spread across 1,703 invoices and 881 customers (£142,264.75). A sensitivity run that removes fully reversed lines was not trained.

The median purchase invoice is £301.65. The 95th percentile is £1,212.16, which is about four times the median, not more than five. Among the 2,829 customers with at least two invoices, the median of their average gap between invoices is 54.0 days. November 2011 is the highest purchase-revenue month (£1,136,534). February 2011 is the lowest (£442,293.59). December 2011 stops on 9 December at 12:50, so that month is incomplete.

Across 12 cohorts that have a month-1 observation, mean month-1 retention is 20.4%. On the 1 November 2011 test snapshot, customers who bought in the next 30 days had a median recency of 33 days. Customers who did not had a median recency of 76 days. That is an association at the cutoff, not a cause.

## Feature engineering

Each eligible customer at each snapshot gets:

- Recency, tenure, historical purchase count, monetary value, average order value, average quantity, distinct products, average unit price, spend per day of tenure
- Spend and invoice counts in the trailing 30 and 90 days
- Spend in the latest 90 days divided by the previous 90 days, when the previous window is positive
- Average days between invoices, when the customer has two or more
- Return and cancellation lines as a share of historical lines
- Country on the most recent historical invoice

Median imputation, scaling for logistic regression, and rare-country grouping are fit inside the sklearn pipeline on training rows only. Missing inter-purchase gap and spend trend keep a missingness indicator. The country grouper keeps levels that appear at least 50 times in the fit fold and labels the rest `Other`.

RFM segment names are not model features. They are a separate descriptive labeling.

## Model comparison

Class imbalance is handled with `class_weight` for logistic regression and random forest, and with `scale_pos_weight` for XGBoost, computed on the rows being fit. No resampling is applied before the split. The search is a small grid scored by PR-AUC on the September 2011 training snapshot, then each model is refit on all training snapshots. The decision threshold maximizes F1 on the October validation snapshot and is frozen before the November test.

Selection rule: highest validation PR-AUC. Random forest won that comparison. The test numbers are reported for every model and were not used to pick the winner.

Validation, October 2011 (positive rate 27.8%, n = 3,613):

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy prior | 0.05 | 0.278 | 1.000 | 0.435 | 0.500 | 0.278 | 0.278 |
| Logistic regression (C = 0.1) | 0.44 | 0.430 | 0.680 | 0.527 | 0.723 | 0.553 | 0.660 |
| XGBoost (depth 3, learning rate 0.05) | 0.46 | 0.439 | 0.657 | 0.527 | 0.729 | 0.562 | 0.672 |
| Random forest (depth 8, min leaf 5) | 0.47 | 0.459 | 0.612 | 0.524 | 0.729 | 0.562 | 0.691 |

Test, November 2011 (positive rate 33.8%, n = 3,981). Thresholds are the validation thresholds:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy prior | 0.338 | 1.000 | 0.506 | 0.500 | 0.338 | 0.229 | 0.338 |
| Logistic regression | 0.527 | 0.688 | 0.597 | 0.745 | 0.644 | 0.197 | 0.686 |
| XGBoost | 0.528 | 0.678 | 0.594 | 0.742 | 0.640 | 0.197 | 0.686 |
| Random forest, selected | 0.546 | 0.630 | 0.585 | 0.742 | 0.641 | 0.194 | 0.698 |

Random forest test confusion matrix at threshold 0.47: true positives 849, false positives 706, false negatives 498, true negatives 1,928.

The validation ranking and the test ranking are not identical. On the test snapshot, logistic regression has a slightly higher PR-AUC (0.644 vs 0.641), ROC-AUC (0.745 vs 0.742), and F1 (0.597 vs 0.585). The gaps are small. The saved model is still the random forest, because it had the highest validation PR-AUC (0.5625 vs 0.5615 for XGBoost and 0.553 for logistic regression). Changing the winner after seeing November would use the test set for selection.

### Why accuracy is a weak summary

33.8% of test customers bought again. Always predicting "no purchase" would be right about 66.2% of the time and would find none of the buyers. The dummy prior model outputs the training base rate for every row. The threshold search then picks 0.05, labels everyone as a buyer, and records accuracy equal to the positive rate (0.338) with recall 1. The selected model's accuracy is 0.698, only a few points above the 66.2% majority rule. PR-AUC moves from a no-skill value of 0.338 to 0.641. That is the comparison that matches the ranking task.

If scores below 0.47 were used as a re-engagement list, that list would contain 2,426 customers (1,928 + 498). Of those, 1,928 did not buy in the next 30 days and 498 did. About 79.5% of the list would be customers who did not purchase, and about 20.5% would have purchased without the offer. That is a description of this threshold on this test window, not an estimate of what a discount would cause.

## Explainability

Permutation importance on a validation sample, and SHAP for the random forest, both ran. SHAP did not fail on this machine (`explanation_methods` is `permutation_importance` and `shap`).

The largest permutation effects on validation average precision were average days between purchases (0.049), invoice count (0.035), historical spend (0.024), recency (0.022), and 90-day spend (0.020). In the SHAP sample, longer gaps and longer recency are associated with a lower predicted probability. Higher frequency, monetary value, and 90-day spend are associated with a higher predicted probability. These are associations inside the fitted model. They do not show that changing a feature would make a customer buy.

## Segmentation

Quintile scores use rank order. Score 5 is the favorable end: more recent, more frequent, higher monetary value. The first matching rule wins.

| Segment | Rule | Customers | Revenue share |
| --- | --- | ---: | ---: |
| Recent and frequent | Recency 4–5 and frequency 4–5 | 1,119 (25.8%) | 66.0% |
| High value | Monetary 4–5 and frequency 3–5 | 649 (15.0%) | 19.9% |
| Inactive | Recency 1–2 and frequency 1–2 | 1,081 (24.9%) | 5.9% |
| Regular | No earlier rule | 790 (18.2%) | 4.5% |
| Needs re-engagement | Recency 1–2 and frequency 3–5 | 386 (8.9%) | 2.2% |
| Promising | Recency 4–5 and frequency 1–2 | 309 (7.1%) | 1.5% |

This profile uses all observed purchase history as of 9 December 2011. It is not the 30-day label. The SQL page builds the same rule names with `NTILE(5)`. Boundary customers can differ between NTILE and the pandas quintiles. Neither set of names is ground truth.

The segment with the most customers among inactive and needs-re-engagement is Inactive (1,081 customers, 5.9% of profiled revenue). A test of a retention offer on that group, with a randomized holdout and a 30-day purchase outcome, is a hypothesis. The historical segment share is not evidence that the offer would work.

## Dashboard

`streamlit run app.py` from the project root, after the pipeline has written its outputs.

| Page | Contents |
| --- | --- |
| Overview | Customer, invoice, revenue, and repeat-purchase figures from the overview SQL, plus the monthly revenue series |
| SQL analytics | Saved query results, a live month filter, and a segment filter |
| Customer segmentation | Python rules and the SQL NTILE summary |
| Model performance | Comparison table, confusion matrix, ROC, precision-recall, calibration, permutation importance, SHAP |
| Customer prediction | An example customer or typed features, a probability, and row-level SHAP when it is available |
| Business insights | Measured statements from the run, and hypotheses labeled as hypotheses |

The probability on the prediction page is a model estimate, not a certainty. Example IDs are the dataset's customer numbers.

Screenshots are not embedded. This session did not capture the browser UI. To take them, run the dashboard, open each page, and save images into `reports/figures/` if you want them in a portfolio. The figures that are already in `reports/figures/` are the EDA and evaluation charts produced by the pipeline, not pictures of the Streamlit app.

## Installation

Python 3.10 or 3.11. This project was run with Python 3.10.8 on Windows. PostgreSQL is optional.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` lists ranges that installed cleanly in this environment, including pandas 2.2.3, scikit-learn 1.5.2, XGBoost 2.1.4, SHAP 0.46.0, Streamlit 1.44.1, SQLAlchemy 2.0.54, and psycopg 3.3.6.

## Database setup

SQLite is the default. The pipeline creates `data/processed/retail.sqlite` from `DATABASE_URL` in `.env`. No server is required. That file is gitignored.

PostgreSQL, if you want it:

1. Create an empty database, for example `retail`.
2. Set `DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/retail` in `.env`. Do not commit `.env`.
3. Run `python -m src.pipeline`. The loader executes `sql/postgres/00_schema.sql` and `01_dimensions.sql`, then the analytics files.

The Python loader inserts the same integer flags either way. Only the date functions in the analytics SQL differ. If you have not executed the PostgreSQL files, do not describe those results as having been run.

## Download the data

The pipeline looks for an `.xlsx` or `.csv` in `data/raw/`, then for a zip already in that folder, then downloads the UCI archive.

Manual download if the network call fails:

1. Open https://archive.ics.uci.edu/dataset/352/online+retail
2. Download `Online Retail.xlsx` (about 23 MB)
3. Place it in `data/raw/`
4. Or set `RAW_DATA_FILE` in `.env` to the full path

The raw file, the SQLite database, and `*.joblib` models are gitignored.

## Training, evaluation, and the dashboard

From the project root, with the virtual environment active:

```bash
python -m src.pipeline
python -m pytest
python notebooks/01_eda.py
streamlit run app.py
```

The pipeline writes `reports/data_quality.json`, `reports/metrics.json`, `reports/model_comparison.csv`, `reports/insights.json`, SQL extracts, figures, and `models/repeat_purchase_pipeline.joblib`. Retrain by running the pipeline again. The seed is `RANDOM_SEED` (default 42). The notebook and `notebooks/01_eda.py` read those outputs and regenerate the EDA figures and the written interpretations. They do not fit models.

## Tests

`pytest` uses synthetic fixtures. It does not download the UCI file. Coverage includes cleaning, duplicate and cancellation flags, snapshot dates, a case where a later purchase must not change monetary value or the label, RFM score direction, metric arithmetic, invalid dashboard inputs, the SQLite pack against a hand-built fixture, and a tiny training loop that checks the test snapshot is held out. The full four-model fit is the pipeline run above, not the unit test. The unit test stubs the catalog down to the dummy and logistic models so the suite stays small, and it still checks that the real catalog lists dummy, logistic regression, random forest, and XGBoost.

## Limitations and next steps

- One year of wholesale-heavy gift invoices is not a consumer app, and it is not a second year that could confirm seasonality.
- Anonymous invoices are invisible to every customer metric.
- Fully reversed orders can remain in purchase revenue, as with invoices `581483` and `C581484`.
- The last calendar day is partial.
- Validation and test disagree slightly on which model ranks first. The winner is not stable to the third digit of PR-AUC.
- RFM names are rules. SHAP and permutation importance are not causal effects.
- A retention offer was not tested. Any action on the Inactive segment or on low scores is a hypothesis until a holdout experiment measures incremental purchases.

Reasonable extensions, none of which are done here: net exact reversals and retrain, add another year of invoices, calibrate the probabilities on a dedicated window, and evaluate the contact list as a cost-sensitive decision rather than as an F1 threshold.
