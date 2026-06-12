# Financial Time Series Forecasting
## VCU – Financial Analytics (FIRE 540)

Applied **ARIMA, SARIMA, and Holt-Winters** exponential smoothing models to 5-year daily equity price data across 6 diversified stocks. Achieved 85% variance explained with RMSE below 2.5%.

---

## Overview

Systematic forecasting study comparing three classical time series models across multiple equities, identifying seasonality, trend components, and volatility patterns. Cross-stock comparisons reveal model performance differences across market sectors.

---

## Models Applied

| Model | Description |
|-------|-------------|
| ARIMA | AutoRegressive Integrated Moving Average |
| SARIMA | Seasonal ARIMA with seasonal components |
| Holt-Winters | Triple exponential smoothing (trend + seasonality) |

---

## Key Results

- **85% variance explained** across all 6 stocks
- **RMSE < 2.5%** on out-of-sample test sets
- Identified cross-stock seasonality and volatility clustering patterns
- SARIMA outperformed ARIMA on stocks with strong seasonal components

---

## Files

```
forecasting_analysis.ipynb     # Full analysis notebook
results/                       # Model outputs and comparison plots
```

---

## Tech Stack

`Python` `pandas` `statsmodels` `pmdarima` `matplotlib` `scikit-learn`

---
*Virginia Commonwealth University - MS Business (Financial Analytics) - FIRE 540*
