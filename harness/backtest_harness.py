"""
backtest_harness.py
Shared backtest infrastructure for the German Auto Registrations forecasting
project. BOTH teammates import this. Do not write your own metric functions
or train/test splits — use these, so results are directly comparable.

Decision locked: use the FULL series 2016-2026 (10+ years, per professor's
requirement). The 2020 structural level shift (COVID + chip shortage) is
handled inside each individual model script (e.g. an intervention dummy
for SARIMA) — NOT by this harness, which is data-agnostic. This file only
standardizes: (1) metric formulas, (2) the expanding-window backtest loop,
(3) the output schema. load_data() defaults to the full series.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_data(path="data/german_auto_monthly_2016_2026.csv", post_2020_only=False):
    """Load the tidy monthly dataset. Returns a DataFrame indexed by date."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if post_2020_only:
        df = df[df.index >= "2020-01-01"]
    return df


# ---------------------------------------------------------------------------
# 2. METRIC FUNCTIONS — use these exactly, do not reimplement
# ---------------------------------------------------------------------------

def mean_error(actual, forecast):
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    return float(np.mean(actual - forecast))

def mae(actual, forecast):
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    return float(np.mean(np.abs(actual - forecast)))

def rmse(actual, forecast):
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))

def theils_u(actual, forecast):
    """
    Theil's U2 (relative to naive/random-walk forecast).
    U < 1: model beats naive. U > 1: model is worse than naive.
    Requires actual/forecast to be aligned, and uses actual[t-1] as the
    naive reference — pass at least 2 points.
    """
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if len(actual) < 2:
        return np.nan
    naive = actual[:-1]              # naive forecast for t = actual at t-1
    actual_t = actual[1:]
    forecast_t = forecast[1:]
    num = np.sqrt(np.mean((forecast_t - actual_t) ** 2))
    den = np.sqrt(np.mean((naive - actual_t) ** 2))
    return float(num / den) if den != 0 else np.nan

def white_noise_test(residuals, lags=12):
    """Ljung-Box test on residuals. Returns p-value (>0.05 = white noise, good)."""
    residuals = pd.Series(residuals).dropna()
    if len(residuals) < lags + 1:
        lags = max(1, len(residuals) // 2)
    result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return float(result["lb_pvalue"].iloc[0])

def normality_test(residuals):
    """Shapiro-Wilk test on residuals. Returns p-value (>0.05 = normal, good)."""
    residuals = pd.Series(residuals).dropna()
    if len(residuals) < 3:
        return np.nan
    stat, p = stats.shapiro(residuals)
    return float(p)

def all_metrics(actual, forecast, residuals=None):
    """Convenience: compute the full metric set in one call."""
    if residuals is None:
        residuals = np.asarray(actual) - np.asarray(forecast)
    return {
        "ME": mean_error(actual, forecast),
        "MAE": mae(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "theils_u": theils_u(actual, forecast),
        "white_noise_pvalue": white_noise_test(residuals),
        "normality_pvalue": normality_test(residuals),
    }


# ---------------------------------------------------------------------------
# 3. EXPANDING WINDOW BACKTEST
# ---------------------------------------------------------------------------

def expanding_window_origins(n_obs, min_train=36, horizon=1, step=1):
    """
    Returns the list of train-end indices ("origins") to backtest from.
    min_train=36 means: don't start backtesting until we have at least 36
    months of training data (3 years) — needed for SARIMA seasonal estimation.
    horizon=1 means we evaluate 1-step-ahead forecasts at each origin.
    """
    origins = []
    t = min_train
    while t + horizon <= n_obs:
        origins.append(t)
        t += step
    return origins


def run_backtest(series, fit_predict_fn, model_name, owner,
                  min_train=36, horizon=1, step=1):
    """
    Generic expanding-window backtest driver.

    fit_predict_fn: a function(train_series) -> single forecast value for
                     the next period (horizon=1). YOUR model code provides
                     this function; the harness handles the looping/scoring.

    Returns a DataFrame in the shared results schema, ready to append to
    results/comparison_sheet.csv
    """
    n = len(series)
    origins = expanding_window_origins(n, min_train=min_train, horizon=horizon, step=step)

    rows = []
    actuals, forecasts = [], []
    for t in origins:
        train = series.iloc[:t]
        actual_value = series.iloc[t + horizon - 1]
        origin_date = series.index[t - 1]

        forecast_value = fit_predict_fn(train)

        actuals.append(actual_value)
        forecasts.append(forecast_value)

        rows.append({
            "model_name": model_name,
            "owner": owner,
            "backtest_origin_date": origin_date.strftime("%Y-%m-%d"),
            "horizon": horizon,
            "actual": actual_value,
            "forecast": forecast_value,
        })

    results_df = pd.DataFrame(rows)

    # Compute running metrics over the whole OOS path (single summary row)
    residuals = np.array(actuals) - np.array(forecasts)
    metrics = all_metrics(actuals, forecasts, residuals)

    summary_row = {
        "model_name": model_name,
        "owner": owner,
        "backtest_origin_date": "ALL_OOS",
        "horizon": horizon,
        "ME": metrics["ME"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "theils_u": metrics["theils_u"],
        "white_noise_pvalue": metrics["white_noise_pvalue"],
        "normality_pvalue": metrics["normality_pvalue"],
        "in_sample_or_oos": "oos",
    }

    return results_df, summary_row


# ---------------------------------------------------------------------------
# 4. RESULTS SHEET — append-only, shared schema
# ---------------------------------------------------------------------------

RESULTS_COLUMNS = [
    "model_name", "owner", "backtest_origin_date", "horizon",
    "ME", "MAE", "RMSE", "theils_u",
    "white_noise_pvalue", "normality_pvalue", "in_sample_or_oos"
]

def append_to_comparison_sheet(summary_row, path="results/comparison_sheet.csv"):
    """Append one model's summary row to the shared comparison sheet."""
    import os
    row_df = pd.DataFrame([summary_row])[RESULTS_COLUMNS]
    if os.path.exists(path):
        row_df.to_csv(path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(path, mode="w", header=True, index=False)
    print(f"Appended {summary_row['model_name']} (owner: {summary_row['owner']}) to {path}")


# ---------------------------------------------------------------------------
# 5. ANNUAL AGGREGATION — for Layer 2 (VDA comparison)
# ---------------------------------------------------------------------------

VDA_ANNUAL_FORECASTS = {
    2024: {"forecast": 2_800_000, "actual": 2_817_331, "forecast_origin": "2024-01"},
    2025: {"forecast": 2_840_000, "actual": 2_857_591, "forecast_origin": "2024-12"},
    2026: {"forecast": 2_900_000, "actual": None, "forecast_origin": "2025-12"},
}

def aggregate_to_annual(monthly_forecast_series):
    """Sum monthly forecasts into annual totals for VDA comparison."""
    return monthly_forecast_series.resample("YE").sum()