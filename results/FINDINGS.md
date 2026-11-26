# Experimental findings (numbers from the CSVs in this folder)

Setup: 5 NAB realKnownCause series, 5 forecasters (naive, ARIMA, GBM, Chronos-Bolt point,
Chronos-Bolt quantile band = CQR), 4 conformal methods, target α = 0.01 unless stated.
Coverage is measured on *normal* test points only (1 − false-alarm rate).

## F1. Split conformal is miscalibrated on time series; the online methods are not
`alpha_sweep.csv`, `fig_alpha_calibration.png`
| median realised miscoverage | α=0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|
| split | 0.0087 | 0.0156 | 0.0287 | 0.0607 | 0.1141 | 0.2131 |
| weighted | 0.0035 | 0.0091 | 0.0181 | 0.0466 | 0.1011 | 0.1983 |
| weighted-clip | 0.0044 | 0.0104 | 0.0205 | 0.0538 | 0.1197 | 0.2332 |
| ACI | 0.0045 | 0.0094 | 0.0194 | 0.0492 | 0.0986 | 0.1990 |

Split's false-alarm rate is ~1.5–1.7× the target at small α. ACI tracks α almost exactly,
as its deterministic long-run guarantee predicts. On i.i.d. data (`sanity_check.py`) all four
methods are within noise of α, so the excess is caused by the data, not by the code.

## F2. Validity and detection power pull in opposite directions
`main_metrics.csv`, `fig_tradeoff.png`, `fig_intervals_machine_temp.png`
Mean over all series × forecasters at α = 0.01:
| method | coverage | false alarms/day | point recall | point F1 | mean delay (h) |
|---|---|---|---|---|---|
| split | 0.982 | 2.85 | 0.070 | 0.109 | 18.6 |
| weighted | 0.991 | 1.61 | 0.028 | 0.051 | 22.4 |
| weighted-clip | 0.989 | 1.81 | 0.042 | 0.071 | 21.5 |
| ACI | 0.991 | 1.71 | 0.015 | 0.029 | 25.9 |

At a *matched* false-alarm budget (fig_tradeoff), split has the highest recall and ACI the lowest.
Reason: adaptive methods widen their intervals *because of* the anomaly (ACI lowers α_t after
every miss; weighted conformal adds the huge anomaly scores to its history), so they stop
alarming in the middle of the event. Adaptivity buys coverage under drift at the cost of
masking sustained anomalies.

## F3. Episode clipping recovers part of the lost power while keeping coverage
Weighted-clip vs weighted: recall 0.028 → 0.042, F1 0.051 → 0.071, coverage 0.991 → 0.989.
Largest effect with GBM: F1 0.078 → 0.151. On i.i.d. data its miscoverage is 0.0095 / 0.050 /
0.107 at α = 0.01 / 0.05 / 0.1 — accurate for small α, slightly liberal at α = 0.1 because
consecutive normal exceedances (probability ≈ α²) are then no longer rare.
Failed first design (worth reporting): clipping *every* flagged score caused a downward ratchet
(1.66% miscoverage at α = 1% on i.i.d. data).

## F4. Which threat to validity matters: shift more than dependence
`dependence.csv`, `fig_dependence.png` (split conformal, α = 0.05, 25 series×forecaster points)
- corr(coverage gap, KS distance calibration-vs-test) = **+0.28**
- corr(coverage gap, effective sample size n/τ_int) = −0.15;  corr(gap, τ_int) = −0.17
The largest gaps (ambient_temp: 2.7–7.0 points) occur on a series with *low* autocorrelation
(lag-1 ACF ≈ 0) but a seasonal level shift between calibration and test. Series with strong
autocorrelation (cpu_asg, nyc_taxi naive: lag-1 ACF ≈ 0.5) have small gaps. This matches
Barber & Pananjady: for mixing (stationary but dependent) data split conformal loses little;
non-stationarity is what breaks it. With only 25 points the correlations are suggestive, not
conclusive.

## F5. A better forecaster is not a better detector
`forecast_accuracy.csv`, `fig_forecasters.png`
- Chronos (zero-shot) has the lowest or near-lowest MAE on ambient_temp (0.87× naive) and
  cpu_asg (0.26×), and is competitive elsewhere — with no training on the data.
- But GBM has the best detection F1 (mean 0.127 vs Chronos 0.054). Part of GBM's advantage is
  an artefact: trees cannot extrapolate below their training range, so during the
  machine_temp failure their forecast stays flat and the error stays large (F1 0.31 there).
  A forecaster that *tracks* the anomaly well (Chronos) produces small errors and misses it.
- Chronos-CQR (quantile band) gives the best-calibrated split conformal coverage (smallest
  gaps in dependence.csv) but slightly lower F1 than the point version.

## F6. Hyper-parameters
`sensitivity.csv`, `fig_sensitivity.png`
- ACI: smaller γ → higher recall (0.031 at γ=5e-4 vs 0.011 at γ=0.05) with FA rate ≈ α throughout.
- Weighted: ρ = 0.99 gives effective sample size ≈ 1/(1−ρ) = 100 < 1/α, so the weighted
  quantile is +∞ and the method never alarms (recall 0). Need 1/(1−ρ) ≳ 1/α. Larger ρ → more
  recall, FA rate creeping slightly above α.

## Limitations to state honestly
- 5 series, 1 seed for deterministic models; labelled NAB windows are wide, so point recall is
  low for every method (event-level detection is 49–51 of 55 windows for all methods).
- Calibration anomalies removed using labels (semi-supervised assumption).
- Chronos-Bolt *small* only, context 512, CPU.
