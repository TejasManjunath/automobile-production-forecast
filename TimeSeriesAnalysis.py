"""
German Passenger Car Registrations — Time Series Forecasting
Models: Holt-Winters, Linear Regression, SARIMA, Forecast Combination
Author: Your Name
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath):
    df_raw = pd.read_excel(filepath, sheet_name='englisch', header=None)
    years  = df_raw.iloc[1, 1:].values
    months = df_raw.iloc[2, 1:].values
    rows = {
        'pc_new_registrations': 4,
        'reg_electric':         8,
        'reg_bev':              9,
        'reg_phev':             10,
    }
    data = {'date': []}
    for col, row_idx in rows.items():
        data[col] = pd.to_numeric(df_raw.iloc[row_idx, 1:].values, errors='coerce')
    for y, m in zip(years, months):
        try:
            data['date'].append(pd.to_datetime(f'{int(y)} {m}', format='%Y %B'))
        except:
            data['date'].append(None)
    ts = pd.DataFrame(data).dropna(subset=['date'])
    ts = ts.set_index('date').sort_index()
    print(f"✓ Data loaded: {ts.index[0].strftime('%b %Y')} → {ts.index[-1].strftime('%b %Y')} ({len(ts)} months)")
    return ts


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHARED BACKTEST HARNESS
# ─────────────────────────────────────────────────────────────────────────────

def get_post2020_series(ts):
    return ts.loc['2020-01-01':, 'pc_new_registrations'].copy()

def get_backtest_origins(series, min_train_months=36):
    n = len(series)
    return list(range(min_train_months - 1, n - 1))

# ── Metrik fonksiyonları ──────────────────────────────────────────────────────

def me(actual, forecast):
    return float(np.mean(np.array(actual) - np.array(forecast)))

def mae(actual, forecast):
    return float(np.mean(np.abs(np.array(actual) - np.array(forecast))))

def rmse(actual, forecast):
    return float(np.sqrt(np.mean((np.array(actual) - np.array(forecast))**2)))

def theils_u(actual, forecast):
    actual    = np.array(actual)
    forecast  = np.array(forecast)
    naive     = actual[:-1]
    actuals   = actual[1:]
    forecasts = forecast[1:]
    rmse_model = np.sqrt(np.mean((actuals - forecasts)**2))
    rmse_naive = np.sqrt(np.mean((actuals - naive)**2))
    return np.nan if rmse_naive == 0 else rmse_model / rmse_naive

def white_noise_test(residuals, lags=12):
    residuals = np.array(residuals)
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) < lags + 2:
        return np.nan
    result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return float(result['lb_pvalue'].iloc[0])

def normality_test(residuals):
    residuals = np.array(residuals)
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) < 3:
        return np.nan
    _, p = stats.shapiro(residuals)
    return float(p)

# ── Results schema ────────────────────────────────────────────────────────────

SCHEMA_COLS = [
    'model_name', 'owner', 'backtest_origin_date', 'horizon',
    'ME', 'MAE', 'RMSE', 'theils_u',
    'white_noise_pvalue', 'normality_pvalue', 'in_sample_or_oos'
]

def make_row(model_name, owner, origin_date, horizon,
             actual_vals, forecast_vals, residuals, oos=True):
    return {
        'model_name':           model_name,
        'owner':                owner,
        'backtest_origin_date': origin_date.strftime('%Y-%m-%d'),
        'horizon':              horizon,
        'ME':                   round(me(actual_vals, forecast_vals), 1),
        'MAE':                  round(mae(actual_vals, forecast_vals), 1),
        'RMSE':                 round(rmse(actual_vals, forecast_vals), 1),
        'theils_u':             round(theils_u(actual_vals, forecast_vals), 4),
        'white_noise_pvalue':   round(white_noise_test(residuals), 4),
        'normality_pvalue':     round(normality_test(residuals), 4),
        'in_sample_or_oos':     'oos' if oos else 'in-sample',
    }

VDA_FORECASTS = {2024: 2_800_000, 2025: 2_840_000, 2026: 2_900_000}
VDA_ACTUALS   = {2024: 2_817_331, 2025: 2_857_591}

OWNER = "Your Name"  # ← değiştir


# ─────────────────────────────────────────────────────────────────────────────
# 3. PLOT FONKSİYONU
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast(series, origins, actuals_all, forecasts_all, model_name="Model"):
    dates = [series.index[i + 1] for i in origins[:len(actuals_all)]]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"{model_name} — Forecast Evaluation", fontsize=14, fontweight='bold')

    axes[0].plot(dates, actuals_all,   label='Actual',   color='steelblue',  linewidth=2)
    axes[0].plot(dates, forecasts_all, label='Forecast', color='darkorange', linewidth=2, linestyle='--')
    axes[0].set_title("Actual vs Forecast (Zaman Serisi)")
    axes[0].set_xlabel("Tarih")
    axes[0].set_ylabel("Değer")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(actuals_all, forecasts_all, color='steelblue', alpha=0.7, edgecolors='white')
    min_val = min(min(actuals_all), min(forecasts_all))
    max_val = max(max(actuals_all), max(forecasts_all))
    axes[1].plot([min_val, max_val], [min_val, max_val],
                 color='red', linestyle='--', linewidth=1.5, label='Mükemmel tahmin')
    axes[1].set_title("Scatter: Actual vs Predicted")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_holt_winters(series, origins, model_name="Holt-Winters"):
    print(f"\n{'='*60}")
    print(f"Running {model_name}...")
    rows = []
    actuals_all, forecasts_all, residuals_all = [], [], []
    for origin_idx in origins:
        train       = series.iloc[:origin_idx + 1]
        actual_next = series.iloc[origin_idx + 1]
        origin_date = series.index[origin_idx]
        if len(train) < 24:
            continue
        try:
            model = ExponentialSmoothing(
                train, trend='add', seasonal='add',
                seasonal_periods=12, initialization_method='estimated'
            ).fit(optimized=True, use_brute=True)
            forecast = model.forecast(1).iloc[0]
            residual = actual_next - forecast
            actuals_all.append(actual_next)
            forecasts_all.append(forecast)
            residuals_all.append(residual)
            rows.append(make_row(model_name, OWNER, origin_date, 1,
                                 [actual_next], [forecast], residuals_all))
        except Exception as ex:
            print(f"  ✗ {origin_date.strftime('%Y-%m')}: {ex}")
    if actuals_all:
        print(f"  Origins run: {len(rows)}")
        print(f"  MAE  = {mae(actuals_all, forecasts_all):,.0f}")
        print(f"  RMSE = {rmse(actuals_all, forecasts_all):,.0f}")
        print(f"  ME   = {me(actuals_all, forecasts_all):,.0f}")
        print(f"  Theil's U        = {theils_u(actuals_all, forecasts_all):.4f}")
        print(f"  LB white-noise p = {white_noise_test(residuals_all):.4f}")
        print(f"  Shapiro p        = {normality_test(residuals_all):.4f}")
    return pd.DataFrame(rows, columns=SCHEMA_COLS), actuals_all, forecasts_all, residuals_all


def build_regression_features(series, up_to_idx):
    n  = up_to_idx + 1
    df = pd.DataFrame({'y': series.iloc[:n].values}, index=series.index[:n])
    df['trend'] = np.arange(1, n + 1)
    for m in range(1, 12):
        df[f'month_{m+1}'] = (df.index.month == m + 1).astype(int)
    df['lag1']  = df['y'].shift(1)
    df['lag12'] = df['y'].shift(12)
    return df.dropna()

def aic_linear(y, X):
    from numpy.linalg import lstsq
    n     = len(y)
    k     = X.shape[1] + 1
    X_aug = np.column_stack([np.ones(n), X])
    beta, _, _, _ = lstsq(X_aug, y, rcond=None)
    sse   = np.sum((y - X_aug @ beta)**2)
    return n * np.log(sse / n) + 2 * k, beta

def forward_select(y, feature_names, X_pool):
    selected    = []
    current_aic = aic_linear(y, np.ones((len(y), 1)))[0]
    for _ in range(len(feature_names)):
        best_aic, best_feat = current_aic, None
        for fname in feature_names:
            if fname in selected:
                continue
            trial_cols = selected + [fname]
            X_trial    = X_pool[:, [feature_names.index(c) for c in trial_cols]]
            trial_aic, _ = aic_linear(y, X_trial)
            if trial_aic < best_aic:
                best_aic, best_feat = trial_aic, fname
        if best_feat is None:
            break
        selected.append(best_feat)
        current_aic = best_aic
    return selected

def run_linear_regression(series, origins, model_name="Linear Regression (AIC)"):
    print(f"\n{'='*60}")
    print(f"Running {model_name}...")
    rows = []
    actuals_all, forecasts_all, residuals_all = [], [], []
    feature_candidates = ['trend'] + [f'month_{m}' for m in range(2, 13)] + ['lag1', 'lag12']
    for origin_idx in origins:
        train_df = build_regression_features(series, origin_idx)
        if len(train_df) < 20:
            continue
        y_train  = train_df['y'].values
        X_pool   = train_df[feature_candidates].values
        selected = forward_select(y_train, feature_candidates, X_pool) or ['trend']
        from numpy.linalg import lstsq
        X_aug        = np.column_stack([np.ones(len(y_train)), train_df[selected].values])
        beta, _, _, _ = lstsq(X_aug, y_train, rcond=None)
        next_date  = series.index[origin_idx + 1]
        next_row   = {'trend': origin_idx + 2}
        for m in range(2, 13):
            next_row[f'month_{m}'] = int(next_date.month == m)
        next_row['lag1']  = series.iloc[origin_idx]
        next_row['lag12'] = series.iloc[origin_idx - 11] if origin_idx >= 11 else np.nan
        x_next   = np.array([next_row.get(f, 0) for f in selected])
        forecast = float(np.dot(np.concatenate([[1], x_next]), beta))
        actual_next = series.iloc[origin_idx + 1]
        residual    = actual_next - forecast
        actuals_all.append(actual_next)
        forecasts_all.append(forecast)
        residuals_all.append(residual)
        rows.append(make_row(model_name, OWNER, series.index[origin_idx], 1,
                             [actual_next], [forecast], residuals_all))
    if actuals_all:
        print(f"  Origins run: {len(rows)}")
        print(f"  MAE  = {mae(actuals_all, forecasts_all):,.0f}")
        print(f"  RMSE = {rmse(actuals_all, forecasts_all):,.0f}")
        print(f"  ME   = {me(actuals_all, forecasts_all):,.0f}")
        print(f"  Theil's U        = {theils_u(actuals_all, forecasts_all):.4f}")
        print(f"  LB white-noise p = {white_noise_test(residuals_all):.4f}")
        print(f"  Shapiro p        = {normality_test(residuals_all):.4f}")
    return pd.DataFrame(rows, columns=SCHEMA_COLS), actuals_all, forecasts_all, residuals_all


def run_arima(series, origins, model_name="SARIMA"):
    """
    Parametreler ACF/PACF ve LB testinden manuel belirlendi:
    SARIMA(1,0,1)(0,1,2,12) → LB p=0.0998 ✅
    """
    print(f"\n{'='*60}")
    print(f"Running {model_name}...")
    order          = (1, 0, 1)
    seasonal_order = (0, 1, 2, 12)
    print(f"  Fixed order: SARIMA{order}{seasonal_order}")
    rows = []
    actuals_all, forecasts_all, residuals_all = [], [], []
    for origin_idx in origins:
        train       = series.iloc[:origin_idx + 1]
        actual_next = series.iloc[origin_idx + 1]
        origin_date = series.index[origin_idx]
        try:
            model = SARIMAX(
                train, order=order, seasonal_order=seasonal_order,
                trend='c', enforce_stationarity=False, enforce_invertibility=False,
            )
            fit      = model.fit(disp=False)
            forecast = float(fit.forecast(1).iloc[0])
            residual = actual_next - forecast
            actuals_all.append(actual_next)
            forecasts_all.append(forecast)
            residuals_all.append(residual)
            rows.append(make_row(model_name, OWNER, origin_date, 1,
                                 [actual_next], [forecast], residuals_all))
        except Exception as ex:
            print(f"  ✗ {origin_date.strftime('%Y-%m')}: {ex}")
    if actuals_all:
        print(f"  Origins run: {len(rows)}")
        print(f"  MAE  = {mae(actuals_all, forecasts_all):,.0f}")
        print(f"  RMSE = {rmse(actuals_all, forecasts_all):,.0f}")
        print(f"  ME   = {me(actuals_all, forecasts_all):,.0f}")
        print(f"  Theil's U        = {theils_u(actuals_all, forecasts_all):.4f}")
        print(f"  LB white-noise p = {white_noise_test(residuals_all):.4f}")
        print(f"  Shapiro p        = {normality_test(residuals_all):.4f}")
    return (pd.DataFrame(rows, columns=SCHEMA_COLS),
            actuals_all, forecasts_all, residuals_all,
            order, seasonal_order)


def run_combination(series, origins,
                    actuals_hw, forecasts_hw,
                    actuals_sarima, forecasts_sarima,
                    model_name="Forecast Combination (HW + SARIMA)"):
    """Equal-weight (0.5/0.5) — weights NOT chosen by leaderboard."""
    print(f"\n{'='*60}")
    print(f"Running {model_name}...")
    n             = min(len(actuals_hw), len(actuals_sarima))
    actuals_all   = actuals_hw[:n]
    forecasts_all = [0.5 * hw + 0.5 * sa
                     for hw, sa in zip(forecasts_hw[:n], forecasts_sarima[:n])]
    residuals_all = [a - f for a, f in zip(actuals_all, forecasts_all)]
    rows = []
    for i, origin_idx in enumerate(origins[:n]):
        rows.append(make_row(model_name, OWNER, series.index[origin_idx], 1,
                             [actuals_all[i]], [forecasts_all[i]], residuals_all[:i+1]))
    print(f"  Origins run: {len(rows)}")
    print(f"  MAE  = {mae(actuals_all, forecasts_all):,.0f}")
    print(f"  RMSE = {rmse(actuals_all, forecasts_all):,.0f}")
    print(f"  ME   = {me(actuals_all, forecasts_all):,.0f}")
    print(f"  Theil's U        = {theils_u(actuals_all, forecasts_all):.4f}")
    print(f"  LB white-noise p = {white_noise_test(residuals_all):.4f}")
    print(f"  Shapiro p        = {normality_test(residuals_all):.4f}")
    return pd.DataFrame(rows, columns=SCHEMA_COLS), actuals_all, forecasts_all


def run_random_walk(series, origins, model_name="Random Walk"):
    print(f"\n{'='*60}")
    print(f"Running {model_name} (benchmark)...")
    rows = []
    actuals_all, forecasts_all = [], []
    for origin_idx in origins:
        actual_next = series.iloc[origin_idx + 1]
        forecast    = series.iloc[origin_idx]
        origin_date = series.index[origin_idx]
        actuals_all.append(actual_next)
        forecasts_all.append(forecast)
        rows.append(make_row(model_name, OWNER, origin_date, 1,
                             [actual_next], [forecast], [actual_next - forecast]))
    print(f"  Origins run: {len(rows)}")
    print(f"  MAE  = {mae(actuals_all, forecasts_all):,.0f}")
    print(f"  RMSE = {rmse(actuals_all, forecasts_all):,.0f}")
    return pd.DataFrame(rows, columns=SCHEMA_COLS), actuals_all, forecasts_all


# ─────────────────────────────────────────────────────────────────────────────
# 5. VDA LAYER-2 COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def vda_comparison(series, order, seasonal_order):
    print(f"\n{'='*60}")
    print("VDA Annual Forecast Comparison (Layer 2)")
    print(f"{'Year':<6} {'Our Forecast':>14} {'VDA Forecast':>14} {'Actual':>12} {'Our Error%':>12} {'VDA Error%':>12}")
    print("-" * 70)
    results = []
    for year in [2024, 2025]:
        train = series.loc[:f'{year-1}-12-01']
        if len(train) < 24:
            continue
        try:
            model      = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                                 enforce_stationarity=False, enforce_invertibility=False)
            our_annual = float(model.fit(disp=False).forecast(12).sum())
            vda_f      = VDA_FORECASTS.get(year, np.nan)
            actual     = VDA_ACTUALS.get(year, np.nan)
            our_err    = 100 * (our_annual - actual) / actual if not np.isnan(actual) else np.nan
            vda_err    = 100 * (vda_f - actual) / actual     if not np.isnan(actual) else np.nan
            print(f"{year:<6} {our_annual:>14,.0f} {vda_f:>14,.0f} {actual:>12,.0f} "
                  f"{our_err:>+11.2f}% {vda_err:>+11.2f}%")
            results.append({'year': year, 'our_forecast': our_annual, 'vda_forecast': vda_f,
                            'actual': actual, 'our_error_pct': our_err, 'vda_error_pct': vda_err})
        except Exception as ex:
            print(f"  ✗ {year}: {ex}")
    try:
        train_2026 = series.loc[:'2025-12-01']
        model      = SARIMAX(train_2026, order=order, seasonal_order=seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False)
        our_2026   = float(model.fit(disp=False).forecast(12).sum())
        vda_2026   = VDA_FORECASTS[2026]
        print(f"{'2026':<6} {our_2026:>14,.0f} {vda_2026:>14,.0f} {'(no actual)':>12}")
        results.append({'year': 2026, 'our_forecast': our_2026, 'vda_forecast': vda_2026,
                        'actual': np.nan, 'our_error_pct': np.nan, 'vda_error_pct': np.nan})
    except Exception as ex:
        print(f"  ✗ 2026: {ex}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts      = load_data('DatenInternetarchivAusgabedatei.xlsx')
    series  = get_post2020_series(ts)
    origins = get_backtest_origins(series, min_train_months=36)
    print(f"  Post-2020 series: {len(series)} months")
    print(f"  Backtest origins: {len(origins)} (first: {series.index[origins[0]].strftime('%b %Y')})")

    all_rows = []

    # ── Random Walk ──────────────────────────────────────────────────────────
    rw_df, rw_act, rw_fcast = run_random_walk(series, origins)
    all_rows.append(rw_df)

    # ── Holt-Winters ─────────────────────────────────────────────────────────
    hw_df, hw_act, hw_fcast, hw_res = run_holt_winters(series, origins)
    all_rows.append(hw_df)
    plot_forecast(series, origins, hw_act, hw_fcast, model_name="Holt-Winters")

    # ── Linear Regression ────────────────────────────────────────────────────
    lr_df, lr_act, lr_fcast, lr_res = run_linear_regression(series, origins)
    all_rows.append(lr_df)
    plot_forecast(series, origins, lr_act, lr_fcast, model_name="Linear Regression (AIC)")

    # ── SARIMA ───────────────────────────────────────────────────────────────
    sa_df, sa_act, sa_fcast, sa_res, sa_order, sa_seas = run_arima(series, origins, model_name="SARIMA")
    all_rows.append(sa_df)
    plot_forecast(series, origins, sa_act, sa_fcast, model_name="SARIMA(1,0,1)(0,1,2,12)")

    # ── Forecast Combination ─────────────────────────────────────────────────
    comb_df, comb_act, comb_fcast = run_combination(
        series, origins,
        actuals_hw=hw_act,     forecasts_hw=hw_fcast,
        actuals_sarima=sa_act, forecasts_sarima=sa_fcast,
    )
    all_rows.append(comb_df)
    plot_forecast(series, origins, comb_act, comb_fcast, model_name="Forecast Combination (HW + SARIMA)")

    # ── VDA Karşılaştırması ──────────────────────────────────────────────────
    vda_comparison(series, order=sa_order, seasonal_order=sa_seas)

    # ── Leaderboard ──────────────────────────────────────────────────────────
    leaderboard = pd.concat(all_rows, ignore_index=True)
    summary = (leaderboard.groupby('model_name')
               .agg(MAE=('MAE', 'mean'), RMSE=('RMSE', 'mean'),
                    ME=('ME', 'mean'), Theils_U=('theils_u', 'mean'),
                    LB_p=('white_noise_pvalue', 'mean'))
               .sort_values('MAE'))
    print(f"\n{'='*60}")
    print("LEADERBOARD (sorted by MAE)")
    print(summary.to_string())