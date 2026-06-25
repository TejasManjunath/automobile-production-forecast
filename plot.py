#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:07:10 2026

@author: zeynepbetulbozdogan
"""
import pandas as pd
import numpy as np
import warnings
from scipy import stats
import matplotlib.pyplot as plt

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
ts = load_data("data/Daten+Internetarchiv+Ausgabedatei_e.xlsx")
def get_post2020_series(ts):
    """Return the post-2020 pc_new_registrations series."""
    return ts.loc['2020-01-01':, 'pc_new_registrations'].copy()

from statsmodels.tsa.seasonal import seasonal_decompose

series = get_post2020_series(ts)  # 2020 sonrası

result = seasonal_decompose(series, model='additive', period=12)
result.plot()
plt.tight_layout()
plt.show()
# acf─────────────────────────────────────────────────────────────────────────────
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Original series
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(series, lags=24, ax=axes[0])
axes[0].set_title("Original Series ACF")

plot_pacf(series, lags=24, ax=axes[1], method="ywm")
axes[1].set_title("Original Series PACF")

plt.tight_layout()
plt.show()

# First-differenced series
series_diff = series.diff().dropna()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(series_diff, lags=24, ax=axes[0])
axes[0].set_title("Differenced Series ACF")

plot_pacf(series_diff, lags=24, ax=axes[1], method="ywm")
axes[1].set_title("Differenced Series PACF")

plt.tight_layout()
plt.show()
#adf
from statsmodels.tsa.stattools import adfuller

sonuc = adfuller(series)
print('p-value:', sonuc[1])


