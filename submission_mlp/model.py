"""COMP9417 model package (demand: MLP; price direction: placeholder).

Demand model, selected in Stage 1 on the 2014 validation; final strategy from the historical
simulation: periodic retraining on the most recent four years (see train.py).
  Inputs for target day D+1 (402 values): NSW demand of day D and of day D-6 (48 each, divided by
  the mean demand of days D-6..D); Sydney temperature of D+1 and of D; humidity, cloud and solar
  radiation of D+1; solar radiation of D (48 each); weekday of D+1 (one-hot); NSW public-holiday
  flags for D+1, D and D-6; a 22 Dec - 4 Jan flag; day-of-year sine/cosine (two harmonics);
  maximum, minimum and 3-day mean temperature of D+1.
  Target: demand of D+1 divided by the same 7-day mean; prediction x mean gives MW.
  Network: 402-256-256-48, GELU, dropout 0.1, output + 1; five seeds, predictions averaged.
Price direction: PLACEHOLDER (same half-hour yesterday above day D's mean price) until the
price model is merged in.

If anything fails inside predict, the package falls back to the same-half-hour-yesterday
baseline for that output instead of failing the day.
train.py imports the feature code from this file, so training and prediction build inputs the
same way.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

P = 48
WEATHER = ["syd_temp", "syd_rh", "syd_cloud", "syd_solar"]
HISTORY_DAYS = 10                      # days of history used at prediction time (needs >= 8)


# ----------------------------------------------------------------------------- features

def market_day(index):
    """Market day of each interval (timestamps mark the end of a half-hour)."""
    return (index - pd.Timedelta("30min")).normalize()


def to_daily(frame):
    """(days, {column: array [n_days, 48]}) for a frame made of complete market days."""
    day = market_day(frame.index)
    days = pd.DatetimeIndex(day.unique())
    n = len(days)
    if len(frame) != n * P or not (np.asarray(day).reshape(n, P) == np.asarray(days)[:, None]).all():
        raise ValueError("expected complete, ordered market days")
    return days, {c: frame[c].to_numpy(float).reshape(n, P) for c in ["nsw_demand"] + WEATHER}


def calendar(days):
    import holidays
    table = holidays.country_holidays("AU", subdiv="NSW", years=range(days.min().year - 1, days.max().year + 2))
    m, dd = days.month.to_numpy(), days.day.to_numpy()
    doy = days.dayofyear.to_numpy() / 365.25
    return {"hol": np.array([d.date() in table for d in days], dtype=float),
            "dow": days.dayofweek.to_numpy(),
            "xmas": (((m == 12) & (dd >= 22)) | ((m == 1) & (dd <= 4))).astype(float),
            "s1": np.sin(2 * np.pi * doy), "c1": np.cos(2 * np.pi * doy),
            "s2": np.sin(4 * np.pi * doy), "c2": np.cos(4 * np.pi * doy)}


def build_inputs(days, arr):
    """Inputs for every day that has 7 earlier days.

    Returns X [n-7, 402] (float64), level [n-7] (mean demand of the 7 previous days) and the
    positions i of those target days in `days`. Demand of the target day itself is never used."""
    Y, T = arr["nsw_demand"], arr["syd_temp"]
    n = len(days)
    i = np.arange(7, n)
    y1, y7 = i - 1, i - 7
    level = np.lib.stride_tricks.sliding_window_view(Y.mean(axis=1), 7)[: n - 7].mean(axis=1)
    L = level[:, None]
    tmean = T.mean(axis=1)
    t3 = (tmean[i] + tmean[i - 1] + tmean[i - 2]) / 3
    cal = calendar(days)
    blocks = [Y[y1] / L, Y[y7] / L, T[i], T[y1], arr["syd_rh"][i], arr["syd_cloud"][i],
              arr["syd_solar"][i], arr["syd_solar"][y1],
              np.eye(7)[cal["dow"][i]], cal["hol"][i, None], cal["hol"][y1, None], cal["hol"][y7, None],
              np.stack([cal[k][i] for k in ("xmas", "s1", "c1", "s2", "c2")], axis=1),
              np.stack([T.max(axis=1)[i], T.min(axis=1)[i], t3], axis=1)]
    return np.concatenate(blocks, axis=1), level, i


# ----------------------------------------------------------------------------- network

class MLP(nn.Module):
    def __init__(self, d_in, h=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, P))

    def forward(self, x):
        return self.net(x) + 1.0             # targets are ratios around 1


# ----------------------------------------------------------------------------- price (placeholder)

def price_direction(history):
    """PLACEHOLDER: same half-hour yesterday above day D's mean price. Replace with the price model."""
    last = history["nsw_price"].to_numpy(float)[-P:]
    return (last > np.nanmean(last)).astype(int)


# ----------------------------------------------------------------------------- interface

class Forecaster:
    def __init__(self, artifact_dir: str):
        torch.set_num_threads(2)
        with open(os.path.join(artifact_dir, "config.json")) as fh:
            self.config = json.load(fh)
        self.members = []
        for seed in self.config["seeds"]:
            net = MLP(self.config["n_inputs"])
            net.load_state_dict(torch.load(os.path.join(artifact_dir, f"mlp_seed{seed}.pt"),
                                           weights_only=True, map_location="cpu"))
            net.eval()
            sc = np.load(os.path.join(artifact_dir, f"scaler_seed{seed}.npz"))
            self.members.append((net, sc["mean"], sc["scale"]))

    def demand(self, history, weather_next):
        day = market_day(history.index)
        tail = history.loc[day >= day.max() - pd.Timedelta(days=HISTORY_DAYS - 1), ["nsw_demand"] + WEATHER]
        nxt = weather_next[WEATHER].copy()
        nxt.insert(0, "nsw_demand", np.nan)
        frame = pd.concat([tail, nxt])
        frame[WEATHER] = frame[WEATHER].interpolate(limit_direction="both")      # gaps should not occur
        frame.iloc[:len(tail), 0] = frame["nsw_demand"].iloc[:len(tail)].interpolate(limit_direction="both")
        days, arr = to_daily(frame)
        if days[-1] != market_day(weather_next.index)[0]:
            raise ValueError("weather_next is not the day after the history")
        X, level, _ = build_inputs(days, arr)
        x = X[-1:]
        out = []
        with torch.inference_mode():
            for net, mean, scale in self.members:
                out.append(net(torch.tensor((x - mean) / scale, dtype=torch.float32)).numpy()[0])
        return np.mean(out, axis=0) * level[-1]

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        yesterday = history["nsw_demand"].to_numpy(float)[-P:]
        try:
            demand = self.demand(history, weather_next)
            if demand.shape != (P,) or not np.all(np.isfinite(demand)):
                raise ValueError("invalid demand forecast")
        except Exception:
            demand = yesterday                                   # fallback: same half-hour yesterday
        try:
            up = price_direction(history)
        except Exception:
            up = np.zeros(P, dtype=int)
        return pd.DataFrame({"nsw_demand_mw": np.clip(demand, 1.0, 19999.0).astype(float),
                             "nsw_price_up": np.asarray(up, dtype=int)}, index=weather_next.index)
