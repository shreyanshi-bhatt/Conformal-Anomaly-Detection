"""One-step-ahead forecasters.

Every forecaster exposes the same interface:
    fit(y, train_idx)        -> learn from y[train_idx] only
    predict(y, start)        -> (lo, hi) arrays; for t >= start, [lo[t], hi[t]] is a forecast
                                band for y[t] that uses only y[0..t-1].  Entries before
                                `start` may be NaN.
Point forecasters return lo == hi == yhat.  Chronos can also return a quantile band.

Conformal prediction only sees the nonconformity score
    s_t = max(lo_t - y_t, y_t - hi_t)
(= |y_t - yhat_t| for a point forecast; the CQR score of Romano et al. 2019 for a band),
so it is agnostic to which model produced the forecast.  This is what lets us compare
a 1-line baseline and a pretrained foundation model on equal footing.
"""
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class NaiveForecaster:
    """yhat[t] = y[t-1]  (random-walk forecast)."""

    name = "naive"

    def fit(self, y, train_idx):
        return self

    def predict(self, y, start=1):
        yhat = np.full(len(y), np.nan)
        yhat[1:] = y[:-1]
        return yhat, yhat


class ARIMAForecaster:
    """ARIMA(p,1,q) fitted on the training segment, then run through the Kalman filter
    on the full series with *frozen* parameters to get honest one-step-ahead forecasts."""

    name = "arima"

    def __init__(self, order=(2, 1, 2), max_train=5000):
        self.order, self.max_train = order, max_train

    def fit(self, y, train_idx):
        from statsmodels.tsa.arima.model import ARIMA

        tr = y[train_idx[-self.max_train:]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = ARIMA(tr, order=self.order).fit()
        return self

    def predict(self, y, start=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_full = self.res.apply(y, refit=False)
        yhat = np.asarray(res_full.fittedvalues, float)   # one-step-ahead predictions
        yhat[:2] = np.nan
        return yhat, yhat


class LagGBMForecaster:
    """Gradient-boosted trees on the last `n_lags` values plus rolling mean / std."""

    name = "gbm"

    def __init__(self, n_lags=24, seed=0):
        self.n_lags = n_lags
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)

    def _features(self, y):
        n, L = len(y), self.n_lags
        X = np.full((n, L + 2), np.nan)
        for k in range(1, L + 1):
            X[k:, k - 1] = y[:-k]
        past = X[:, :L]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # first rows have no history yet
            X[:, L] = np.nanmean(past, axis=1)
            X[:, L + 1] = np.nanstd(past, axis=1)
        return X

    def fit(self, y, train_idx):
        X = self._features(y)
        idx = train_idx[train_idx >= self.n_lags]
        self.model.fit(X[idx], y[idx])
        return self

    def predict(self, y, start=1):
        X = self._features(y)
        yhat = np.full(len(y), np.nan)
        ok = np.arange(len(y)) >= self.n_lags
        yhat[ok] = self.model.predict(X[ok])
        return yhat, yhat


class ChronosForecaster:
    """Pretrained time-series foundation model Chronos-Bolt (Ansari et al. 2024), used zero-shot:
    it is never trained or fine-tuned on our data, so `fit` does nothing.

    mode="point": band = median forecast (lo == hi)
    mode="band" : band = [q_lo, q_hi] predicted quantiles  -> conformalised quantile regression (CQR)
    """

    _pipe = None  # shared across instances; loading the weights once is enough

    def __init__(self, mode="point", context=512, q_lo=0.1, q_hi=0.9, batch=256):
        self.mode, self.context, self.q_lo, self.q_hi, self.batch = mode, context, q_lo, q_hi, batch
        self.name = "chronos" if mode == "point" else "chronos_cqr"

    @classmethod
    def pipeline(cls):
        if cls._pipe is None:
            import torch
            from chronos import BaseChronosPipeline

            torch.manual_seed(0)
            cls._pipe = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32
            )
        return cls._pipe

    def fit(self, y, train_idx):
        return self

    def raw_quantiles(self, y, start):
        """(n, 3) array with q_lo, median, q_hi one-step forecasts for t >= start."""
        import torch

        pipe = self.pipeline()
        out = np.full((len(y), 3), np.nan)
        ts = np.arange(max(start, 16), len(y))
        for i in range(0, len(ts), self.batch):
            chunk = ts[i:i + self.batch]
            ctx = [torch.tensor(y[max(0, t - self.context):t], dtype=torch.float32) for t in chunk]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=1, quantile_levels=[self.q_lo, 0.5, self.q_hi])
            out[chunk] = q[:, 0, :].numpy()
        return out

    def predict(self, y, start=1, cached=None):
        q = cached if cached is not None else self.raw_quantiles(y, start)
        if self.mode == "point":
            return q[:, 1], q[:, 1]
        return q[:, 0], q[:, 2]
