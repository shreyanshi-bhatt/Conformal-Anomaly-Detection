"""Copy figures into report/figures and generate LaTeX tables in report/tables from results/*.csv.

Run after experiments.py, so the numbers in the report always match the experiments:
    .venv\\Scripts\\python make_report_assets.py
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
REP = ROOT / "report"
(REP / "figures").mkdir(parents=True, exist_ok=True)
(REP / "tables").mkdir(parents=True, exist_ok=True)

METHOD = {"split": "Split", "weighted": "Weighted", "weighted_clip": "Weighted-clip (ours)", "aci": "ACI"}
FC = {"naive": "Naive", "arima": "ARIMA", "gbm": "GBM", "chronos": "Chronos", "chronos_cqr": "Chronos-CQR"}
M_ORDER, FC_ORDER = list(METHOD), list(FC)

for f in RES.glob("fig_*.png"):
    shutil.copy2(f, REP / "figures" / f.name)


def write(name, body):
    (REP / "tables" / name).write_text(body, encoding="utf-8")
    print("wrote", name)


# ---- Table: main results by conformal method (mean over 5 series x 5 forecasters), alpha = 0.01
m = pd.read_csv(RES / "main_metrics.csv")
g = m.groupby("method").agg(coverage=("coverage", "mean"), fa=("fa_per_day", "mean"), prec=("precision", "mean"),
                            rec=("recall", "mean"), f1=("f1", "mean"), delay=("delay_h", "mean"),
                            det=("detected", "sum"), ev=("events", "sum")).loc[M_ORDER]
rows = "\n".join(
    f"{METHOD[k]} & {r.coverage:.3f} & {r.fa:.2f} & {r.prec:.3f} & {r.rec:.3f} & {r.f1:.3f} & {r.delay:.1f} & {int(r.det)}/{int(r.ev)} \\\\"
    for k, r in g.iterrows())
write("tab_main_methods.tex", rf"""\begin{{tabular}}{{lccccccc}}
\toprule
Method & Coverage & FA/day & Precision & Recall & F1 & Delay (h) & Windows \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
""")

# ---- Table: F1 per forecaster x method
p = m.pivot_table(index="forecaster", columns="method", values="f1", aggfunc="mean").loc[FC_ORDER, M_ORDER]
cov = m.pivot_table(index="forecaster", columns="method", values="coverage", aggfunc="mean").loc[FC_ORDER, M_ORDER]
rows = "\n".join(
    f"{FC[f]} & " + " & ".join(f"{p.loc[f, k]:.3f} ({cov.loc[f, k]:.3f})" for k in M_ORDER) + r" \\"
    for f in FC_ORDER)
write("tab_forecaster_method.tex", rf"""\begin{{tabular}}{{l{'c' * len(M_ORDER)}}}
\toprule
Forecaster & {' & '.join(METHOD[k] for k in M_ORDER)} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
""")

# ---- Table: forecast accuracy (MAE relative to naive) per series
a = pd.read_csv(RES / "forecast_accuracy.csv").pivot(index="series", columns="forecaster", values="rel_mae")[FC_ORDER]
rows = "\n".join(
    rf"\texttt{{{s.replace('_', chr(92) + '_')}}} & " + " & ".join(
        (rf"\textbf{{{v:.3f}}}" if v == a.loc[s].min() else f"{v:.3f}") for v in a.loc[s]) + r" \\"
    for s in a.index)
write("tab_forecast_accuracy.tex", rf"""\begin{{tabular}}{{l{'c' * len(FC_ORDER)}}}
\toprule
Series & {' & '.join(FC[f] for f in FC_ORDER)} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
""")

# ---- Table: alpha sweep (median realised miscoverage)
s = pd.read_csv(RES / "alpha_sweep.csv")
s["mis"] = 1 - s.coverage
t = s.groupby(["method", "alpha"]).mis.median().unstack().loc[M_ORDER]
rows = "\n".join(f"{METHOD[k]} & " + " & ".join(f"{v:.4f}" for v in t.loc[k]) + r" \\" for k in M_ORDER)
write("tab_alpha_sweep.tex", rf"""\begin{{tabular}}{{l{'c' * t.shape[1]}}}
\toprule
Method & {' & '.join(f'$\\alpha={a:g}$' for a in t.columns)} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
""")

# ---- Table: data sets
from data import SERIES, SHORT, chronological_split, load_nab, steps_per_day  # noqa: E402
rows = []
for ser in SERIES:
    df = load_nab(ser)
    tr, cal, te = chronological_split(len(df))
    step = df.timestamp.diff().mode()[0]
    rows.append(rf"\texttt{{{SHORT[ser].replace('_', chr(92) + '_')}}} & {len(df):,} & {int(step.total_seconds() // 60)} min & "
                rf"{df.window_id.max() + 1} & {int(df.window_id[te].max() >= 0 and df.window_id[te][df.window_id[te] >= 0].nunique())} & "
                rf"{100 * df.is_anomaly.mean():.1f}\% \\")
write("tab_datasets.tex", r"""\begin{tabular}{lccccc}
\toprule
Series & Length & Sampling & Windows & Windows in test & Anomalous points \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
