# Simulation & Forecasting Techniques — Project Brief
## German Passenger Car New Registrations: Classical Time Series Forecasting vs. VDA Institutional Forecast

---

## 1. Objective

Forecast monthly German passenger car new registrations using the full
classical forecasting toolkit from the course, evaluate every model under
one identical backtest, and report a forecast combination of the
best-performing models. Compare results against the VDA's published annual
forecast and against a random-walk benchmark.

We are not assuming a win going in — we're building the framework correctly
and reporting whatever it finds. VDA's recent annual forecasts have been
very accurate (~0.6% error in 2024 and 2025), so the honest headline result
may be "competitive, not a clean win." Our best realistic shot at a genuine
beat is the EV/BEV registrations sub-series, where VDA revised its forecast
several times after Germany ended its EV purchase subsidy in Dec 2023.

**Two-layer evaluation:**
- **Layer 1 — Monthly:** expanding-window backtest vs. random walk. Many
  data points, real statistical comparison.
- **Layer 2 — Annual:** sum 12 monthly forecasts to a full-year total,
  compare vs. VDA's published forecast at VDA's actual forecast date. Few
  data points — treat as a case study, not a stats test.

---

## 2. Dataset

`german_auto_monthly_2016_2026.csv` — already cleaned, tidy, monthly,
Jan 2016–Apr 2026, no missing months. Columns: `date`,
`pc_new_registrations` (PRIMARY TARGET), `pc_domestic_production`,
`reg_electric`, `reg_bev`, `reg_phev`, `prod_electric`, `prod_bev`,
`prod_phev`.

Source: Deutsche Bundesbank monthly compilation, built from VDA + Bundesbank
calculations.

VDA institutional benchmark figures (full-year registrations):
2024 forecast ~2.80M (actual 2,817,331) · 2025 forecast ~2.84M
(actual 2,857,591) · 2026 forecast ~2.90M.

**Known data issue — must be decided BEFORE anyone fits a model, as a team,
not independently:** registrations show a level shift — ~3.4–3.6M/year in
2016–2019 vs. ~2.6–2.86M/year from 2020 onward (COVID + chip shortage).
Lock one approach for the whole team: either (a) train on post-2020 data
only, or (b) use the full series with an intervention dummy for the COVID
period. Whichever is chosen, every model uses the same choice.

---

## 3. THE SHARED HARNESS — build this first, before any model fitting

This is the one piece of infrastructure that is NOT split, NOT parallel,
and NOT "whoever is comfortable with it." One of you builds it tonight as
the first task; both of you import it and call it. This is what makes the
comparison sheet trustworthy.

It must fix, for every model without exception:
1. **The train/test boundary(ies).** Use an expanding-window backtest, not
   a single split — define the fixed list of origin points (e.g. start
   backtesting from month 85, grow the window by 1 each step) once, in one
   place.
2. **The metric functions.** One shared `mean_error()`, `mae()`, `rmse()`,
   `theils_u()`, `white_noise_test()`, `normality_test()` — every model
   calls the same functions on the same units (raw registrations, not a
   mix of raw and log/scaled).
3. **The annual aggregation function** for Layer 2 (sum 12 monthly
   forecasts → compare to VDA figure at the correct origin date).
4. **The shared results schema** — one row per model per backtest origin,
   written to the same sheet/CSV, same column names, so results from both
   of you stack cleanly without manual reconciliation.

Whoever builds it shares the code/notebook with the other before either of
you finalizes a single model's numbers.

---

## 4. Required Models — pick freely, work in parallel, race to the sheet

All 10 are required in the final deck regardless of who claims them first.
Pick whichever you're more comfortable with; trade or split remaining ones
once you see who's faster on what.

1. Mean
2. Moving Average MA(k)
3. Weighted Moving Average WMA(k)
4. Single Exponential Smoothing (SES)
5. Double Exponential Smoothing (Holt's linear trend)
6. Triple Exponential Smoothing (Holt-Winters)
7. Linear Regression with AIC-based stepwise selection
8. ARIMA (Box-Jenkins)
9. SARIMA (seasonal Box-Jenkins)
10. Forecast Combination (e.g. weighted average of the two best individual
    models by out-of-sample Theil's U — decide combination membership by
    theoretical complementarity, not by peeking at the full leaderboard
    and cherry-picking after the fact)

For models 1–3: implement, run through the same harness, and explicitly
document why they're inappropriate given the data's trend + seasonality —
this is graded content (slide 8 rubric: "why are some models not
appropriate").

---

## 5. The shared comparison sheet

One spreadsheet/CSV, one row per (model, backtest origin), columns:
`model_name | owner | backtest_origin_date | horizon | ME | MAE | RMSE |
theils_u | white_noise_pvalue | normality_pvalue | in_sample_or_oos`

Update it as each model finishes. Once all 10 are in, the combination step
and the VDA/random-walk comparison both read directly from this sheet.

---

## 6. Workflow / Slide Structure (graded directly — follow this agenda)

1. Problem definition
2. Information gathering
3. Exploratory analysis (plot, descriptive stats, ACF/PACF, stationarity/
   unit-root test, decomposition, seasonality, qq-plot/normality)
4. Model choice (justified by exploratory analysis, before touching the
   test set)
5. Fitting the model (in-sample stats, all 10 models)
6. Model use and backtesting (out-of-sample, random walk, VDA, discussion
   of where/why we win or lose)

---

## 7. Non-negotiables, regardless of who does what

- Same harness, same metrics, same train/test windows for every model — no
  exceptions, no independently-built backtests.
- Don't choose the headline/combination model by scanning out-of-sample
  performance across all 10 and keeping the best — let exploratory
  analysis nominate ARIMA/SARIMA orders before the test set is touched.
- Granularity discipline: monthly model output vs. monthly random walk;
  monthly-summed-to-annual vs. VDA annual, at VDA's actual forecast origin
  date. Never compare a monthly number to an annual one directly.
- Be ready to explain the COVID/chip-shortage handling decision in the
  discussion.

---

# SHAREABLE AI PROMPT (paste into your own AI tool)

```
I'm working on a university group project for a Simulation & Forecasting
Techniques course (MSc Business Intelligence & Data Science). Two of us are
doing modeling in parallel; a third teammate handles slides/documentation
separately. We are racing to implement 10 required models and feeding
results into one shared comparison sheet, so consistency across our two
parallel workstreams is critical.

CONTEXT
Forecasting German passenger car new registrations (monthly, Jan 2016 -
Apr 2026, n=124, no missing months) using classical time series methods,
benchmarked against the VDA's (German Association of the Automotive
Industry) published annual forecast and against a random walk. The course
requires implementing every model below under ONE identical backtest
framework, then reporting a forecast combination of the best models.
VDA's recent annual forecasts have been very accurate (~0.6% error in 2024
and 2025) - we are not assuming we'll beat them on the headline series; our
realistic shot at a genuine win is the EV/BEV registrations sub-series,
where VDA revised forecasts repeatedly after Germany ended its EV purchase
subsidy in December 2023.

DATASET
Tidy monthly CSV: date, pc_new_registrations (PRIMARY TARGET),
pc_domestic_production, reg_electric, reg_bev, reg_phev, prod_electric,
prod_bev, prod_phev. Source: Bundesbank monthly compilation built from VDA
data. KNOWN ISSUE: level shift around 2020 (pre-2020 ~3.4-3.6M/year,
post-2020 ~2.6-2.86M/year registrations) from COVID + chip shortage - our
team has agreed to handle this as: [fill in: "training on post-2020 data
only" OR "full series with a COVID intervention dummy"] - use this
consistently in everything you generate for me.

SHARED BACKTEST HARNESS (use exactly this, do not invent your own splits
or metric formulas - my teammate's results must be comparable to mine)
- Expanding-window backtest, not a single train/test split. [Paste in the
  exact origin points / window definition once your team has built and
  shared the harness code.]
- Metric functions: ME, MAE, RMSE, Theil's U, white-noise test on
  residuals, normality test (qq-plot) - on raw registration units, not
  log/scaled.
- Output format: one row per (model, backtest origin) with columns
  model_name, owner, backtest_origin_date, horizon, ME, MAE, RMSE,
  theils_u, white_noise_pvalue, normality_pvalue, in_sample_or_oos - so it
  drops straight into our shared comparison sheet.

MY ASSIGNED MODELS: [fill in which of the 10 you're claiming, e.g. "Mean,
Moving Average, Weighted Moving Average, Single Exponential Smoothing"]

WHAT I NEED FROM YOU
1. Python code (pandas/statsmodels/pmdarima as appropriate) to fit my
   assigned models on pc_new_registrations, using the shared harness above.
2. In-sample stats for each model (ME, MAE, RMSE, Theil's U, residual
   diagnostics).
3. Out-of-sample results from the expanding-window backtest, in the exact
   output schema above.
4. For Mean/MA/WMA specifically: a clear explanation of why the model is
   NOT appropriate for this data given its trend and strong seasonality
   (seasonal index roughly: Jan 85, Feb 89, Mar 115, Jun 114, relative to
   100 average) - needed for our "why some models are inappropriate"
   slide.
5. Flag explicitly if anything I'm asking for would constitute data
   snooping (e.g. choosing final model parameters by peeking at
   out-of-sample performance) so I avoid it before it's in the sheet.

Ask me clarifying questions before writing code if anything about the
harness or data handling is ambiguous - I'd rather lose two minutes now
than produce results my teammate can't compare to theirs.
```
## 8. Repo structure 
```
auto-forecasting-project/
├── README.md                          ← the brief + objective
├── data/
│   └── german_auto_monthly_2016_2026.csv
├── harness/
│   └── backtest_harness.py            ← built FIRST, shared, nobody touches after
├── models/
│   ├── modeler1_baselines_smoothing.py
│   └── modeler2_arima_sarima_combo.py
├── results/
│   └── comparison_sheet.csv           ← both of you append rows here, harness enforces the schema
└── slides/                           
```
## 9. Links 
1. Google Sheets : https://docs.google.com/spreadsheets/d/1ZEj9fiKvT7f7GpcaDJrukRmAjcT3F2u4j8v44R8iA-k/edit?usp=sharing
2. The dataset itself — german_auto_monthly_2016_2026.csv (already built, attached above). This is the one your team actually models on. Don't make teammates re-derive it from raw sources.
3. VDA monthly figures (source of the dataset, for citation/methodology slide): https://www.vda.de/en/news/facts-and-figures/monthly-figures
4. VDA 2026 forecast (institutional benchmark numbers): https://www.vda.de/en/press/press-releases/2025/251208_PM_Forecasts_2026
5. Bundesbank Monthly Report Nov 2024 (chart source, automotive sector discussion, cite for context not as a numeric benchmark): https://publikationen.bundesbank.de/publikationen-en/reports-studies/monthly-reports/monthly-report-november-2024-943818
