# Conformal Time-Series Anomaly Detection (CSE 598 Statistical Learning Theory, final project)

Pipeline: forecaster → nonconformity score `s_t = max(lo_t − y_t, y_t − hi_t)` → online conformal
radius `q_t` → alarm if `s_t > q_t`.

## Layout
| File | What it does |
|---|---|
| `src/data.py` | Loads 5 NAB *realKnownCause* series + labelled anomaly windows; chronological 40/15/45 split |
| `src/forecasters.py` | Naive, ARIMA(2,1,2), lag-feature gradient boosting, Chronos-Bolt-small (zero-shot; point and quantile band) |
| `src/conformal.py` | Split, weighted (Barber et al. 2023), weighted-clip (our anomaly-aware variant), ACI (Gibbs & Candès 2021) |
| `experiments.py` | Runs every stage, writes CSVs and figures to `results/` |
| `figures.py` | All report figures |
| `sanity_check.py` | Checks each method hits miscoverage ≈ α on i.i.d. data |
| `data/raw/` | NAB CSVs and `combined_windows.json` (github.com/numenta/NAB) |

## Setup and run
```
python -m venv .venv
.venv\Scripts\python -m pip install numpy pandas scipy statsmodels scikit-learn matplotlib
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install chronos-forecasting
.venv\Scripts\python sanity_check.py
.venv\Scripts\python experiments.py
```
First run downloads Chronos-Bolt-small (~190 MB) and takes roughly 15–20 min on a laptop CPU;
forecasts are cached in `results/cache/`, so reruns take a few minutes.

## Design decisions
- **Chronological split** (train 40% / calibration 15% / test 45%): no shuffling, no look-ahead.
- **Clean calibration**: labelled anomalies inside the calibration segment are removed (standard
  semi-supervised assumption, Bates et al. 2023). Training data is *not* cleaned.
- **Online setting**: weighted and ACI methods see each test score after predicting it, as a
  monitoring system would. Split conformal never updates.
- **Metrics**: coverage on *normal* test points (1 − false-alarm rate), median interval width,
  point precision/recall/F1 inside labelled windows, windows detected, mean detection delay.

## Outputs (`results/`)
`main_metrics.csv`, `forecast_accuracy.csv`, `alpha_sweep.csv`, `sensitivity.csv`, `dependence.csv`,
`fig_intervals_machine_temp.png`, `fig_alpha_calibration.png`, `fig_dependence.png`,
`fig_rolling_alarm_rate.png`, `fig_sensitivity.png`, `fig_forecasters.png`, `fig_tradeoff.png`.
Summary of what the numbers show: `results/FINDINGS.md`.
