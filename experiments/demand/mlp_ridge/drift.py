"""Online drift detectors for the demand model.

Input-based detector (ShapeNovelty): at the end of day D, take D's 48 demand values divided by
their mean (the daily shape, which is also the model's lag-1 input) and measure its PCA
reconstruction error under a PCA fitted on a reference period (standardised shapes, components
explaining 95% of the variance). Days whose shape is unlike anything in the reference period get
a large error. WeatherNovelty does the same for the day's Sydney temperature, solar, cloud and
humidity profiles; it is used only as a diagnostic (does an input change always hurt?).

Error-based detector: ratio of the model's summed daily MAPE to the same-half-hour-yesterday
baseline's summed daily MAPE over the last 28 eligible days. Dividing by the baseline removes
most of the seasonal difficulty (summer, Christmas) that a raw error threshold would flag.

Both monitor a 28-day rolling statistic over eligible days (public holidays and the
22 Dec - 4 Jan period are skipped: calendar effects are known in advance and are not drift).
Thresholds are quantiles of the same statistic on a held-out reference year.
"""
from collections import deque

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WINDOW = 28


class _PCANovelty:
    def __init__(self, var=0.95):
        self.var = var

    def _vectors(self, block):
        raise NotImplementedError

    def fit(self, block):
        V = self._vectors(block)
        self.sc = StandardScaler().fit(V)
        self.pca = PCA(n_components=self.var, svd_solver="full").fit(self.sc.transform(V))
        return self

    def score(self, block):
        Z = self.sc.transform(self._vectors(block))
        rec = self.pca.inverse_transform(self.pca.transform(Z))
        return np.mean((Z - rec) ** 2, axis=1)


class ShapeNovelty(_PCANovelty):
    """block: (n_days, 48) demand in MW."""

    def _vectors(self, block):
        return block / block.mean(axis=1, keepdims=True)


class WeatherNovelty(_PCANovelty):
    """block: (n_days, 4*48) = temperature, solar, cloud, humidity profiles concatenated."""

    def _vectors(self, block):
        return block


def rolling_eligible(values, eligible, window=WINDOW):
    """Mean of the last `window` eligible values, carried forward over ineligible days."""
    s = pd.Series(np.where(eligible, values, np.nan))
    r = s.dropna().rolling(window).mean()
    return r.reindex(s.index).ffill().to_numpy()


def rolling_ratio(a, b, eligible, window=WINDOW):
    """sum(a)/sum(b) over the last `window` eligible days, carried forward."""
    sa = pd.Series(np.where(eligible, a, np.nan)).dropna().rolling(window).sum()
    sb = pd.Series(np.where(eligible, b, np.nan)).dropna().rolling(window).sum()
    r = (sa / sb).reindex(range(len(a))).ffill()
    return r.to_numpy()


class OnlineWindow:
    """Rolling window that is cleared when a new model is deployed."""

    def __init__(self, window=WINDOW):
        self.a, self.b = deque(maxlen=window), deque(maxlen=window)
        self.window = window

    def clear(self):
        self.a.clear()
        self.b.clear()

    def push(self, a, b=None):
        self.a.append(a)
        if b is not None:
            self.b.append(b)

    @property
    def full(self):
        return len(self.a) == self.window

    def mean(self):
        return float(np.mean(self.a))

    def ratio(self):
        return float(np.sum(self.a) / np.sum(self.b))


def episodes(dates, stat, thr):
    """Contiguous runs where stat > thr: start, end, peak value and length in days."""
    above = np.nan_to_num(stat, nan=-np.inf) > thr
    out, start = [], None
    for k, flag in enumerate(np.append(above, False)):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            seg = slice(start, k)
            out.append({"start": dates[start], "end": dates[k - 1], "days": k - start,
                        "peak": float(np.nanmax(stat[seg]))})
            start = None
    return pd.DataFrame(out)
