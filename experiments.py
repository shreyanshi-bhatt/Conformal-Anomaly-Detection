"""Full experiment suite for the project.

    .venv\\Scripts\\python experiments.py            # everything (forecasts are cached after the first run)

Stages
  0. forecasts   : one-step forecasts for every series x forecaster  -> results/cache/*.npz
  1. main        : alpha = 0.01, every series x forecaster x conformal method -> results/main_metrics.csv
  2. alpha sweep : empirical coverage vs target 1-alpha                      -> results/alpha_sweep.csv
  3. sensitivity : ACI gamma and weighted-conformal rho                       -> results/sensitivity.csv
  4. dependence  : coverage gap of split conformal vs dependence / shift      -> results/dependence.csv
  5. figures     : results/fig_*.png
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent / "src"))
from conformal import METHODS, aci, weighted_conformal  # noqa: E402
from data import SERIES, SHORT, chronological_split, load_nab, steps_per_day  # noqa: E402
from forecasters import (ARIMAForecaster, ChronosForecaster, LagGBMForecaster,  # noqa: E402
                         NaiveForecaster)

ROOT = Path(__file__).parent
OUT = ROOT / "results"
CACHE = OUT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

ALPHA = 0.01
FORECASTERS = ["naive", "arima", "gbm", "chronos", "chronos_cqr"]
METHOD_ORDER = ["split", "weighted", "weighted_clip", "aci"]


# --------------------------------------------------------------------------- stage 0
def get_forecasts(series):
    """Return dict name -> (lo, hi) for every forecaster, computing and caching as needed."""
    df = load_nab(series)
    y = df.value.to_numpy(float)
    tr, cal, te = chronological_split(len(y))
    out = {}
    for name, make in [("naive", NaiveForecaster), ("arima", ARIMAForecaster), ("gbm", LagGBMForecaster)]:
        f = CACHE / f"{series}__{name}.npz"
        if not f.exists():
            t0 = time.time()
            lo, hi = make().fit(y, tr).predict(y, start=cal[0])
            np.savez(f, lo=lo, hi=hi)
            print(f"  {SHORT[series]:13s} {name:12s} {time.time() - t0:6.1f}s")
        d = np.load(f)
        out[name] = (d["lo"], d["hi"])
    f = CACHE / f"{series}__chronos_quantiles.npz"
    if not f.exists():
        t0 = time.time()
        q = ChronosForecaster().raw_quantiles(y, start=cal[0])
        np.savez(f, q=q)
        print(f"  {SHORT[series]:13s} chronos      {time.time() - t0:6.1f}s")
    q = np.load(f)["q"]
    out["chronos"] = ChronosForecaster("point").predict(y, cached=q)
    out["chronos_cqr"] = ChronosForecaster("band").predict(y, cached=q)
    return out


def scores_of(y, lo, hi):
    return np.maximum(lo - y, y - hi)


# --------------------------------------------------------------------------- metrics
def evaluate(y, lo, hi, q, is_anom, win_id, spd):
    s = scores_of(y, lo, hi)
    flag = s > q
    normal = ~is_anom
    tp, fp, fn = (flag & is_anom).sum(), (flag & normal).sum(), (~flag & is_anom).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    events = [i for i in np.unique(win_id) if i >= 0]
    delays, detected = [], 0
    for i in events:
        hits = np.flatnonzero(flag[win_id == i])
        if len(hits):
            detected += 1
            delays.append(hits[0] / spd * 24)            # hours from window start to first alarm
    width = (hi - lo) + 2 * q
    return {
        "coverage": 1 - fp / normal.sum(),
        "fa_per_day": fp / (normal.sum() / spd),
        "width": float(np.median(width[np.isfinite(width)])),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "events": len(events),
        "detected": detected,
        "delay_h": float(np.mean(delays)) if delays else np.nan,
    }


def prepared():
    """Yield everything needed per series x forecaster, once."""
    for series in SERIES:
        df = load_nab(series)
        y = df.value.to_numpy(float)
        is_anom, win_id = df.is_anomaly.to_numpy(), df.window_id.to_numpy()
        tr, cal, te = chronological_split(len(y))
        fc = get_forecasts(series)
        for name in FORECASTERS:
            lo, hi = fc[name]
            s = scores_of(y, lo, hi)
            # clean calibration: labelled anomalies are removed (standard assumption in
            # conformal anomaly detection, e.g. Bates et al. 2023)
            s_cal = s[cal][~is_anom[cal] & np.isfinite(s[cal])]
            yield dict(series=series, forecaster=name, y=y[te], lo=lo[te], hi=hi[te], s_cal=s_cal,
                       s_te=s[te], is_anom=is_anom[te], win_id=win_id[te], spd=steps_per_day(df),
                       timestamps=df.timestamp.to_numpy()[te])


def run(method, P, alpha, **kw):
    q = METHODS[method](P["s_cal"], P["s_te"], alpha, **kw)
    return q, evaluate(P["y"], P["lo"], P["hi"], q, P["is_anom"], P["win_id"], P["spd"])


# --------------------------------------------------------------------------- stages 1-4
def main():
    print("stage 0: forecasts")
    PREP = list(prepared())

    print("stage 1: main table")
    rows, accuracy = [], []
    for P in PREP:
        n = ~P["is_anom"]
        center = (P["lo"] + P["hi"]) / 2
        accuracy.append({"series": SHORT[P["series"]], "forecaster": P["forecaster"],
                         "mae_normal": np.nanmean(np.abs(P["y"] - center)[n]),
                         "mae_naive": np.nanmean(np.abs(np.diff(P["y"]))[n[1:]])})
        for m in METHOD_ORDER:
            _, met = run(m, P, ALPHA)
            rows.append({"series": SHORT[P["series"]], "forecaster": P["forecaster"], "method": m, **met})
    main_df = pd.DataFrame(rows)
    main_df.to_csv(OUT / "main_metrics.csv", index=False)
    acc = pd.DataFrame(accuracy)
    acc["rel_mae"] = acc.mae_normal / acc.mae_naive
    acc.to_csv(OUT / "forecast_accuracy.csv", index=False)

    print("stage 2: alpha sweep")
    alphas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    rows = []
    for P in PREP:
        if P["forecaster"] not in ("naive", "gbm", "chronos"):
            continue
        for a in alphas:
            for m in METHOD_ORDER:
                _, met = run(m, P, a)
                rows.append({"series": SHORT[P["series"]], "forecaster": P["forecaster"], "method": m,
                             "alpha": a, **met})
    pd.DataFrame(rows).to_csv(OUT / "alpha_sweep.csv", index=False)

    print("stage 3: sensitivity")
    rows = []
    for P in PREP:
        if P["forecaster"] not in ("gbm", "chronos"):
            continue
        for g in [0.0005, 0.001, 0.005, 0.01, 0.05]:
            q = aci(P["s_cal"], P["s_te"], ALPHA, gamma=g)
            met = evaluate(P["y"], P["lo"], P["hi"], q, P["is_anom"], P["win_id"], P["spd"])
            rows.append({"series": SHORT[P["series"]], "forecaster": P["forecaster"], "method": "aci",
                         "param": "gamma", "value": g, **met})
        for r in [0.99, 0.995, 0.999, 0.9995, 1.0]:
            for clip in (False, True):
                q = weighted_conformal(P["s_cal"], P["s_te"], ALPHA, rho=r, clip=clip)
                met = evaluate(P["y"], P["lo"], P["hi"], q, P["is_anom"], P["win_id"], P["spd"])
                rows.append({"series": SHORT[P["series"]], "forecaster": P["forecaster"],
                             "method": "weighted_clip" if clip else "weighted", "param": "rho", "value": r, **met})
    pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)

    print("stage 4: dependence vs coverage gap (split conformal)")
    rows = []
    for P in PREP:
        n = ~P["is_anom"]
        s_norm_te = P["s_te"][n & np.isfinite(P["s_te"])]
        sc = P["s_cal"]
        acf1 = np.corrcoef(sc[:-1], sc[1:])[0, 1]
        # integrated autocorrelation time of the calibration scores (first 50 lags, positive part)
        z = (sc - sc.mean()) / sc.std()
        acfs = [np.mean(z[:-k] * z[k:]) for k in range(1, 51)]
        tau = 1 + 2 * sum(max(a, 0) for a in acfs)
        ks = ks_2samp(sc, s_norm_te).statistic
        for a in [0.01, 0.05, 0.1]:
            _, met = run("split", P, a)
            rows.append({"series": SHORT[P["series"]], "forecaster": P["forecaster"], "alpha": a,
                         "coverage_gap": (1 - a) - met["coverage"], "acf1": acf1, "tau_int": tau,
                         "n_eff": len(sc) / tau, "ks_shift": ks, "n_cal": len(sc)})
    pd.DataFrame(rows).to_csv(OUT / "dependence.csv", index=False)

    print("stage 5: figures")
    import figures
    figures.make_all(PREP)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
