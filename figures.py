"""Figures for the report (static PNGs, sized for a single LaTeX column / full text width)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from conformal import METHODS  # noqa: E402

OUT = Path(__file__).parent / "results"

# fixed categorical order (never cycled); markers give a second, non-colour encoding
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MK = ["o", "s", "^", "D", "v"]
METHOD_ORDER = ["split", "weighted", "weighted_clip", "aci"]
METHOD_LABEL = {"split": "Split", "weighted": "Weighted", "weighted_clip": "Weighted-clip (ours)", "aci": "ACI"}
FC_ORDER = ["naive", "arima", "gbm", "chronos", "chronos_cqr"]
FC_LABEL = {"naive": "Naive", "arima": "ARIMA", "gbm": "GBM", "chronos": "Chronos", "chronos_cqr": "Chronos-CQR"}
INK, MUTED, GRID, ANOM = "#1f1f1e", "#6b6a64", "#e4e3de", "#f6d9a8"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def shade_anomalies(ax, t, is_anom):
    ax.fill_between(t, 0, 1, where=is_anom, transform=ax.get_xaxis_transform(), color=ANOM, lw=0,
                    label="labelled anomaly window", zorder=0)


def fig_intervals(PREP, series="machine_temperature_system_failure", fc="chronos", alpha=0.01):
    P = next(p for p in PREP if p["series"] == series and p["forecaster"] == fc)
    t, y = P["timestamps"], P["y"]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.4), sharex=True)
    for i, (ax, m) in enumerate(zip(axes, METHOD_ORDER)):
        q = METHODS[m](P["s_cal"], P["s_te"], alpha)
        lo, hi = P["lo"] - q, P["hi"] + q
        flag = (y < lo) | (y > hi)
        shade_anomalies(ax, t, P["is_anom"])
        ax.fill_between(t, lo, hi, color=C[0], alpha=0.28, lw=0, label=f"{int(100 * (1 - alpha))}% interval")
        ax.plot(t, y, color=INK, lw=0.6, label="observed")
        ax.scatter(t[flag], y[flag], s=7, color=C[1], zorder=3, label="alarm")
        ax.set_title(f"{METHOD_LABEL[m]}  —  alarms on normal points: {(flag & ~P['is_anom']).sum()}, "
                     f"inside anomaly windows: {(flag & P['is_anom']).sum()}", loc="left")
        ax.set_ylabel("temperature")
        ax.set_ylim(np.nanmin(y) - 15, np.nanmax(y) + 15)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.025))
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT / "fig_intervals_machine_temp.png")
    plt.close(fig)


def fig_alpha_calibration():
    d = pd.read_csv(OUT / "alpha_sweep.csv")
    d["miscov"] = 1 - d.coverage
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    a = np.array(sorted(d.alpha.unique()))
    ax.plot(a, a, color=MUTED, lw=1, ls="--", label="target (y = α)")
    for i, m in enumerate(METHOD_ORDER):
        g = d[d.method == m].groupby("alpha").miscov
        ax.fill_between(a, g.quantile(0.25), g.quantile(0.75), color=C[i], alpha=0.12, lw=0)
        ax.plot(a, g.median(), color=C[i], marker=MK[i], ms=4, label=METHOD_LABEL[m])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("target miscoverage α"); ax.set_ylabel("realised false-alarm rate on normal points")
    ax.set_title("Calibration: median and IQR over 15 series×forecaster runs", loc="left", fontsize=8.5)
    ax.legend(frameon=False)
    fig.savefig(OUT / "fig_alpha_calibration.png")
    plt.close(fig)


def fig_dependence():
    d = pd.read_csv(OUT / "dependence.csv")
    d = d[d.alpha == 0.05]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, col, lab in [(axes[0], "ks_shift", "KS distance, calibration vs. test scores"),
                         (axes[1], "n_eff", "effective calibration size n / τ_int")]:
        for i, fc in enumerate(FC_ORDER):
            g = d[d.forecaster == fc]
            ax.scatter(g[col], g.coverage_gap, color=C[i], marker=MK[i], s=26, label=FC_LABEL[fc],
                       edgecolor="white", linewidth=0.8)
        r = np.corrcoef(d[col], d.coverage_gap)[0, 1]
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel(lab); ax.set_title(f"Pearson r = {r:.2f}", loc="left")
    axes[1].set_xscale("log")
    axes[0].set_ylabel("coverage gap (1−α) − coverage, α = 0.05")
    axes[1].legend(frameon=False, fontsize=7.5)
    fig.savefig(OUT / "fig_dependence.png")
    plt.close(fig)


def fig_rolling(PREP, series="machine_temperature_system_failure", fc="chronos", alpha=0.01):
    P = next(p for p in PREP if p["series"] == series and p["forecaster"] == fc)
    t = P["timestamps"]
    win = 3 * P["spd"]  # three-day window
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    shade_anomalies(ax, t, P["is_anom"])
    for i, m in enumerate(METHOD_ORDER):
        q = METHODS[m](P["s_cal"], P["s_te"], alpha)
        miss = (P["s_te"] > q).astype(float)
        roll = pd.Series(miss).rolling(win, min_periods=win // 2).mean()
        ax.plot(t, roll, color=C[i], lw=1.4, label=METHOD_LABEL[m])
    ax.axhline(alpha, color=MUTED, ls="--", lw=1, label="α = 0.01")
    ax.set_ylabel("3-day rolling alarm rate")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=7)
    fig.autofmt_xdate()
    fig.savefig(OUT / "fig_rolling_alarm_rate.png")
    plt.close(fig)


def fig_sensitivity():
    d = pd.read_csv(OUT / "sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    g = d[d.param == "gamma"].groupby("value")[["coverage", "recall"]].mean()
    ax = axes[0]
    ax.plot(g.index, 1 - g.coverage, color=C[0], marker=MK[0], ms=4, label="false-alarm rate")
    ax.plot(g.index, g.recall, color=C[1], marker=MK[1], ms=4, label="point recall")
    ax.axhline(0.01, color=MUTED, ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("ACI step size γ"); ax.set_title("ACI", loc="left")
    ax.legend(frameon=False)
    ax = axes[1]
    for j, m in enumerate(["weighted", "weighted_clip"]):
        g = d[(d.param == "rho") & (d.method == m)].groupby("value")[["coverage", "recall"]].mean()
        x = np.arange(len(g))
        ls = "-" if m == "weighted" else ":"
        ax.plot(x, 1 - g.coverage, color=C[0], marker=MK[0], ms=4, ls=ls, label=f"FA rate, {METHOD_LABEL[m]}")
        ax.plot(x, g.recall, color=C[1], marker=MK[1], ms=4, ls=ls, label=f"recall, {METHOD_LABEL[m]}")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in g.index])
    ax.axhline(0.01, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("decay ρ   (effective sample size ≈ 1/(1−ρ))")
    ax.set_title("Weighted conformal", loc="left"); ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "fig_sensitivity.png")
    plt.close(fig)


def fig_forecasters():
    acc = pd.read_csv(OUT / "forecast_accuracy.csv")
    main = pd.read_csv(OUT / "main_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    series = list(acc.series.unique())
    x = np.arange(len(series)); w = 0.16
    for i, fc in enumerate(FC_ORDER):
        v = acc[acc.forecaster == fc].set_index("series").loc[series, "rel_mae"]
        axes[0].bar(x + (i - 2) * w, v, w * 0.88, color=C[i], label=FC_LABEL[fc])
        f = main[(main.forecaster == fc) & (main.method == "weighted_clip")].set_index("series").loc[series, "f1"]
        axes[1].bar(x + (i - 2) * w, f, w * 0.88, color=C[i], label=FC_LABEL[fc])
    axes[0].axhline(1, color=MUTED, lw=1, ls="--")
    axes[0].set_ylabel("MAE / naive MAE (normal test points)")
    axes[1].set_ylabel("point F1 (weighted-clip, α = 0.01)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(series, rotation=25, ha="right"); ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.savefig(OUT / "fig_forecasters.png")
    plt.close(fig)


def fig_tradeoff():
    """Recall vs false alarms/day as alpha varies: compares methods at matched alarm budgets."""
    d = pd.read_csv(OUT / "alpha_sweep.csv")
    fcs = ["naive", "gbm", "chronos"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, fc in zip(axes, fcs):
        for i, m in enumerate(METHOD_ORDER):
            g = d[(d.forecaster == fc) & (d.method == m)].groupby("alpha")[["fa_per_day", "recall"]].mean()
            ax.plot(g.fa_per_day, g.recall, color=C[i], marker=MK[i], ms=4, lw=1.6, label=METHOD_LABEL[m])
        ax.set_xscale("log"); ax.set_title(FC_LABEL[fc], loc="left"); ax.set_xlabel("false alarms per day")
    axes[0].set_ylabel("point recall (mean over 5 series)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "fig_tradeoff.png")
    plt.close(fig)


def make_all(PREP):
    fig_tradeoff()
    fig_intervals(PREP)
    fig_alpha_calibration()
    fig_dependence()
    fig_rolling(PREP)
    fig_sensitivity()
    fig_forecasters()
