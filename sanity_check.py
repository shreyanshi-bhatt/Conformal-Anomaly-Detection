"""Sanity check: on i.i.d. (exchangeable) scores every method should hit miscoverage ~= alpha.
If this fails, the implementation is wrong -- not the theory.

    .venv\\Scripts\\python sanity_check.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from conformal import METHODS  # noqa: E402

rng = np.random.default_rng(0)
print(f"{'method':15s} " + " ".join(f"a={a:<6}" for a in (0.01, 0.05, 0.1)))
for m, f in METHODS.items():
    res = []
    for a in (0.01, 0.05, 0.1):
        cal, test = rng.exponential(size=3000), rng.exponential(size=20000)
        res.append(np.mean(test > f(cal, test, a)))
    print(f"{m:15s} " + " ".join(f"{r:<8.4f}" for r in res))
