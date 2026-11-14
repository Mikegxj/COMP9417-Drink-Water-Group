"""NSW price component for the team's Forecaster."""
from pathlib import Path

import numpy as np
import pandas as pd

HALF_HOUR = pd.Timedelta(minutes=30)
PER_DAY = 48
WEATHER = ["syd_temp", "mel_temp", "bne_temp"]
MARKET = ["nsw_price", "nsw_demand"]
MODEL_FILE = "nsw_price_relative.ubj"


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
        from xgboost import XGBClassifier

        self.model = XGBClassifier(n_jobs=2)
        self.model.load_model(Path(artifact_dir) / MODEL_FILE)

    def predict(self, history, weather_next):
        X = next_day_features(history, weather_next)
        return pd.Series(self.model.predict(X), index=X.index,
                         name="nsw_price_up", dtype=int)
