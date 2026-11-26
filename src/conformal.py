"""Online conformal methods that turn a stream of nonconformity scores into a radius q_t.

Setting: scores s_1, s_2, ... arrive in time order.  At time t the radius q_t may only use
scores strictly before t.  The interval is [lo_t - q_t, hi_t + q_t] and the point is flagged
as anomalous iff s_t > q_t.

Guarantees (the statistical-learning-theory side of the project)
------------------------------------------------------------------
split     : if calibration + test scores are exchangeable, P(s_t <= q) >= 1 - alpha
            (Vovk et al. 2005; Lei et al. 2018).  Barber & Pananjady (2026) show that for
            beta-mixing series the loss of coverage is bounded by a term that shrinks with
            the mixing coefficients and the calibration size.
weighted  : Barber, Candes, Ramdas & Tibshirani (2023): with fixed weights w_i,
            coverage >= 1 - alpha - sum_i w~_i * d_TV(Z, Z^i), so down-weighting old scores
            limits the damage from distribution drift.
aci       : Gibbs & Candes (2021): for ANY sequence, |mean_t err_t - alpha| <= (max(a1,1-a1)+gamma)/(gamma T).
            A deterministic long-run guarantee; no distributional assumption at all.
"""
import numpy as np


def conformal_quantile(scores, alpha):
    """ceil((n+1)(1-alpha))/n empirical quantile -- the finite-sample correction.
    Returns +inf when n is too small for the requested level."""
    scores = np.asarray(scores)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return np.partition(scores, k - 1)[k - 1]


def weighted_quantile(scores, weights, alpha):
    """1-alpha quantile of  sum_i w~_i delta_{s_i} + w~_{n+1} delta_{+inf}  with w_{n+1} = 1
    (Barber et al. 2023, eq. for nonexchangeable conformal)."""
    order = np.argsort(scores)
    s, w = scores[order], weights[order]
    total = w.sum() + 1.0
    cdf = np.cumsum(w) / total
    j = np.searchsorted(cdf, 1 - alpha, side="left")
    return np.inf if j >= len(s) else s[j]


def split_conformal(scores_cal, scores_test, alpha):
    """Static split conformal: one radius from the calibration scores, never updated."""
    return np.full(len(scores_test), conformal_quantile(scores_cal, alpha))


def aci(scores_cal, scores_test, alpha, gamma=0.005, window=2000):
    """Adaptive Conformal Inference (Gibbs & Candes 2021).
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t),   err_t = 1{s_t > q_t}
    The quantile is taken over a sliding window of the most recent scores."""
    hist = np.asarray(scores_cal[-window:], float)
    buf = np.concatenate([hist, np.asarray(scores_test, float)])
    off = len(hist)
    a_t, q = alpha, np.empty(len(scores_test))
    for t, s in enumerate(scores_test):
        past = buf[max(0, off + t - window): off + t]
        q[t] = np.inf if a_t <= 0 else (-np.inf if a_t >= 1 else conformal_quantile(past, a_t))
        a_t += gamma * (alpha - float(s > q[t]))
    return q


def weighted_conformal(scores_cal, scores_test, alpha, rho=0.999, window=3000, clip=False):
    """Nonexchangeable (weighted) conformal, Barber et al. (2023), run online.

    Weights decay geometrically with age: w_i = rho^(t - i), so the radius tracks slow drift.
    clip=True is our anomaly-aware variant ("episode clipping").  Real anomalies arrive as
    *runs* of consecutive exceedances, while exceedances of normal noise are mostly isolated
    (for i.i.d. scores two in a row has probability ~alpha^2).  So: if s_t is flagged AND
    s_{t-1} was flagged too, s_t enters the history as min(s_t, q_t^+), q_t^+ = next float above
    q_t.  The clipped value still counts as an exceedance (rank-preserving), but the anomaly's
    magnitude cannot inflate future radii.  Isolated exceedances are stored unchanged.

    Design note (kept for the report): clipping EVERY flagged score -- to q_t or to q_t^+ --
    creates a ratchet: the history can never again hold values above past thresholds, so q_t
    drifts down and the false-alarm rate rises (1.7% at alpha = 1% even on i.i.d. data).
    """
    hist = list(np.asarray(scores_cal[-window:], float))
    q = np.empty(len(scores_test))
    prev_flag = False
    for t, s in enumerate(scores_test):
        h = np.asarray(hist)
        w = rho ** np.arange(len(h), 0, -1)       # oldest gets smallest weight
        q[t] = weighted_quantile(h, w, alpha)
        flag = s > q[t]
        hist.append(min(s, np.nextafter(q[t], np.inf)) if (clip and flag and prev_flag) else s)
        prev_flag = flag
        if len(hist) > window:
            hist.pop(0)
    return q


METHODS = {
    "split": split_conformal,
    "weighted": weighted_conformal,
    "weighted_clip": lambda c, s, a, **kw: weighted_conformal(c, s, a, clip=True, **kw),
    "aci": aci,
}
