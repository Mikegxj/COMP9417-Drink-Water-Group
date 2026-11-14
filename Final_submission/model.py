"""Drink Water Group: NSW demand (five-seed MLP) and price direction (relative-price XGBoost)."""
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HALF_HOUR = pd.Timedelta(minutes=30)
PER_DAY = 48
WEATHER = ["syd_temp", "mel_temp", "bne_temp"]
MARKET = ["nsw_price", "nsw_demand"]
MODEL_FILE = "nsw_price_relative.joblib"


def market_days(index):
    return (index - HALF_HOUR).normalize()


def check_grid(index):
    if len(index) == 0 or index[0] != index[0].normalize() + HALF_HOUR:
        raise ValueError("Expected a market day starting at 00:30 AEST.")
    expected = pd.date_range(index[0], periods=len(index), freq=HALF_HOUR)
    if len(index) % PER_DAY or not index.equals(expected):
        raise ValueError("Expected contiguous 48-slot market days ending at midnight.")


def price_features(frame):
    """Common feature calculation for training and day-ahead prediction."""
    check_grid(frame.index)
    days = market_days(frame.index)
    start = frame.index - HALF_HOUR
    period = start.hour * 2 + start.minute // 30
    price = frame["nsw_price"]
    yesterday = price.shift(PER_DAY)
    last_week = price.shift(7 * PER_DAY)
    mean = price.groupby(days).transform("mean").shift(PER_DAY)
    std = yesterday.groupby(days).transform("std")
    scale = std.clip(lower=1.0)
    weekly_mean = yesterday.groupby(period).transform(
        lambda slots: slots.rolling(7, min_periods=7).mean()
    )
    X = pd.DataFrame(index=frame.index)
    X["period"] = period
    X["syd_temp"] = frame["syd_temp"]
    X["previous_day_price_std"] = std
    X["weekday_sin"] = np.sin(2 * np.pi * days.dayofweek / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * days.dayofweek / 7)
    X["demand_lag1"] = frame["nsw_demand"].shift(PER_DAY)
    X["demand_lag7"] = frame["nsw_demand"].shift(7 * PER_DAY)
    X["up_lag1"] = (yesterday > mean).astype("int8")
    X["up_lag7"] = (last_week > mean).astype("int8")
    X["mel_temp"] = frame["mel_temp"]
    X["bne_temp"] = frame["bne_temp"]
    X["nsw_price_lag1_relative"] = (yesterday - mean) / scale
    X["nsw_price_lag7_relative"] = (last_week - mean) / scale
    X["same_period_7day_mean_relative"] = (weekly_mean - mean) / scale
    return X


def next_day_features(history, weather_next):
    if len(weather_next) != PER_DAY or len(history) < 7 * PER_DAY:
        raise ValueError("Need 48 forecast slots and at least seven complete market days.")
    past = history.iloc[-7 * PER_DAY:][MARKET]
    frame = pd.concat([past, weather_next[WEATHER]])
    return price_features(frame).iloc[-PER_DAY:]


class PriceForecaster:
    """Called by the combined Forecaster to produce its nsw_price_up column."""

    def __init__(self, artifact_dir):
        self.model = joblib.load(Path(artifact_dir) / MODEL_FILE)

    def predict(self, history, weather_next):
        X = next_day_features(history, weather_next)
        return pd.Series(self.model.predict(X), index=X.index,
                         name="nsw_price_up", dtype=int)


# Demand features and network follow the team's MLP training implementation.
P = PER_DAY
DEMAND_WEATHER = ["syd_temp", "syd_rh", "syd_cloud", "syd_solar"]
HISTORY_DAYS = 10


def market_day(index):
    return market_days(index)


def to_daily(frame):
    """Return ordered market days and arrays with 48 values per day."""
    check_grid(frame.index)
    day = market_day(frame.index)
    days = pd.DatetimeIndex(day.unique())
    n = len(days)
    if len(frame) != n * P or not (np.asarray(day).reshape(n, P) == np.asarray(days)[:, None]).all():
        raise ValueError("Expected complete, ordered market days")
    return days, {c: frame[c].to_numpy(float).reshape(n, P)
                  for c in ["nsw_demand"] + DEMAND_WEATHER}


def calendar(days):
    import holidays
    table = holidays.country_holidays(
        "AU", subdiv="NSW", years=range(days.min().year - 1, days.max().year + 2)
    )
    m, dd = days.month.to_numpy(), days.day.to_numpy()
    doy = days.dayofyear.to_numpy() / 365.25
    return {"hol": np.array([d.date() in table for d in days], dtype=float),
            "dow": days.dayofweek.to_numpy(),
            "xmas": (((m == 12) & (dd >= 22)) | ((m == 1) & (dd <= 4))).astype(float),
            "s1": np.sin(2 * np.pi * doy), "c1": np.cos(2 * np.pi * doy),
            "s2": np.sin(4 * np.pi * doy), "c2": np.cos(4 * np.pi * doy)}


def build_inputs(days, arr):
    """Build 402 inputs using seven previous demand days and known weather."""
    Y, T = arr["nsw_demand"], arr["syd_temp"]
    n = len(days)
    i = np.arange(7, n)
    y1, y7 = i - 1, i - 7
    level = np.lib.stride_tricks.sliding_window_view(Y.mean(axis=1), 7)[:n - 7].mean(axis=1)
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


class MLP(nn.Module):
    def __init__(self, d_in, h=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, P)
        )

    def forward(self, x):
        return self.net(x) + 1.0


class Forecaster:
    def __init__(self, artifact_dir: str):
        torch.set_num_threads(2)
        directory = Path(artifact_dir)
        with (directory / "config.json").open() as fh:
            self.config = json.load(fh)
        self.members = []
        for seed in self.config["seeds"]:
            net = MLP(self.config["n_inputs"])
            state = torch.load(directory / f"mlp_seed{seed}.pt", weights_only=True, map_location="cpu")
            net.load_state_dict(state)
            net.eval()
            with np.load(directory / f"scaler_seed{seed}.npz") as sc:
                self.members.append((net, sc["mean"], sc["scale"]))
        self.price = PriceForecaster(directory)

    def demand(self, history, weather_next):
        day = market_day(history.index)
        tail = history.loc[day >= day.max() - pd.Timedelta(days=HISTORY_DAYS - 1),
                           ["nsw_demand"] + DEMAND_WEATHER]
        nxt = weather_next[DEMAND_WEATHER].copy()
        nxt.insert(0, "nsw_demand", np.nan)
        frame = pd.concat([tail, nxt])
        frame[DEMAND_WEATHER] = frame[DEMAND_WEATHER].interpolate(limit_direction="both")
        frame.iloc[:len(tail), 0] = frame["nsw_demand"].iloc[:len(tail)].interpolate(limit_direction="both")
        days, arr = to_daily(frame)
        if market_day(weather_next.index)[0] != day.max() + pd.Timedelta(days=1):
            raise ValueError("weather_next must be the day after the history")
        X, level, _ = build_inputs(days, arr)
        x = X[-1:]
        out = []
        with torch.inference_mode():
            for net, mean, scale in self.members:
                out.append(net(torch.tensor((x - mean) / scale, dtype=torch.float32)).numpy()[0])
        return np.mean(out, axis=0) * level[-1]

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        if len(weather_next) != P or len(history) < P:
            raise ValueError("Expected 48 forecast slots and at least 48 history slots")
        yesterday = history["nsw_demand"].to_numpy(float)[-P:]
        try:
            demand = self.demand(history, weather_next)
            if demand.shape != (P,) or not np.all(np.isfinite(demand)):
                raise ValueError("Invalid demand forecast")
        except Exception:
            demand = yesterday
        try:
            up = self.price.predict(history, weather_next).to_numpy()
            if up.shape != (P,) or not np.isin(up, [0, 1]).all():
                raise ValueError("Invalid price forecast")
        except Exception:
            last = history["nsw_price"].to_numpy(float)[-P:]
            up = (last > np.nanmean(last)).astype(int)
        return pd.DataFrame({
            "nsw_demand_mw": np.clip(demand, 1.0, 19999.0).astype(float),
            "nsw_price_up": np.asarray(up, dtype=int),
        }, index=weather_next.index)
