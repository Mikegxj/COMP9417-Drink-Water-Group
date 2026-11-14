"""Drink Water Group: NSW demand (N-BEATSx) and price direction (relative-price XGBoost)."""
import json
import random
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


N_PERIODS = PER_DAY
COVARIATES = ["temp", "slot_sin", "slot_cos", "weekday_sin", "weekday_cos",
              "year_sin", "year_cos", "hol_today", "hol_lag1", "hol_lag7"]


def nsw_holidays(years):
    import holidays
    table = holidays.Australia(subdiv="NSW", years=years)
    return {pd.Timestamp(day) for day, name in table.items() if "Bank Holiday" not in name}


def known_covariates(index, temp, scalers):
    start = index - HALF_HOUR
    cov = [(np.asarray(temp, dtype=np.float32) - scalers["temp_mean"]) / scalers["temp_scale"]]
    for positions, period in [(start.hour * 2 + start.minute // 30, 48), (start.dayofweek, 7),
                              (start.dayofyear - 1, 365.25)]:
        cov.extend([np.sin(2 * np.pi * positions / period), np.cos(2 * np.pi * positions / period)])
    day = start.normalize()
    known = nsw_holidays(range(day.year.min() - 1, day.year.max() + 2))
    for offset in (0, 1, 7):
        shifted = day - pd.Timedelta(days=offset)
        cov.append(np.fromiter((d in known for d in shifted), bool, len(shifted)).astype(np.float32))
    return torch.tensor(np.column_stack(cov), dtype=torch.float32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size

    def forward(self, theta):
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, self.backcast_size:]
        return backcast, forecast.reshape(len(forecast), -1, 1)


class NBEATSBlock(nn.Module):
    def __init__(self, input_size, h, futr_input_size, mlp_units):
        super().__init__()
        self.futr_input_size = futr_input_size
        width = input_size + futr_input_size * (input_size + h)
        layers = [nn.Linear(width, mlp_units[0][0])]
        for units in mlp_units:
            layers += [nn.Linear(units[0], units[1]), nn.ReLU()]
        layers.append(nn.Linear(mlp_units[-1][1], input_size + h))
        self.layers = nn.Sequential(*layers)
        self.basis = IdentityBasis(input_size, h)

    def forward(self, insample_y, futr_exog):
        if self.futr_input_size > 0:
            insample_y = torch.cat((insample_y, futr_exog.reshape(len(insample_y), -1)), dim=1)
        return self.basis(self.layers(insample_y))


class NBEATSx(nn.Module):
    def __init__(self, h, input_size, futr_size=0, n_stacks=3, mlp_units=None, seed=1):
        super().__init__()
        seed_everything(seed)
        mlp_units = mlp_units or 3 * [[512, 512]]
        self.blocks = nn.ModuleList([NBEATSBlock(input_size, h, futr_size, mlp_units)
                                     for _ in range(n_stacks)])

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"].squeeze(-1)
        residuals = insample_y.flip(dims=(-1,))
        mask = windows_batch["insample_mask"].squeeze(-1).flip(dims=(-1,))
        forecast = insample_y[:, -1:, None]
        for block in self.blocks:
            backcast, block_forecast = block(residuals, windows_batch["futr_exog"])
            residuals = (residuals - backcast) * mask
            forecast = forecast + block_forecast
        return forecast


def build_model(config, seed):
    torch.manual_seed(seed)
    width = config["width"]
    return NBEATSx(h=N_PERIODS, input_size=config["window"], futr_size=len(COVARIATES),
                   mlp_units=[[width, width]] * 3, seed=seed)


def make_batch(load, known):
    history = load.unsqueeze(-1)
    return dict(insample_y=history, insample_mask=torch.ones_like(history), futr_exog=known,
                hist_exog=None, stat_exog=None)


class Forecaster:
    def __init__(self, artifact_dir: str):
        torch.set_num_threads(2)
        directory = Path(artifact_dir)
        with (directory / "demand_nbeatsx.json").open() as fh:
            self.meta = json.load(fh)
        self.model = build_model(self.meta["config"], self.meta["seed"])
        state = torch.load(directory / "demand_nbeatsx.pt", weights_only=True, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self.price = PriceForecaster(directory)

    def demand(self, history, weather_next):
        window = self.meta["config"]["window"]
        if len(weather_next) != N_PERIODS or len(history) < window:
            raise ValueError("Expected 48 target half-hours and at least the input window of history")
        past = history.iloc[-window:]
        y = past["nsw_demand"].to_numpy(dtype=np.float32)
        temp = np.concatenate([past["syd_temp"].to_numpy(np.float32),
                               weather_next["syd_temp"].to_numpy(np.float32)])
        index = past.index.append(weather_next.index)
        y = pd.Series(y).interpolate(limit_direction="both").to_numpy(np.float32)
        temp = pd.Series(temp).interpolate(limit_direction="both").to_numpy(np.float32)
        load = torch.tensor((y - self.meta["demand_mean"]) / self.meta["demand_scale"]).unsqueeze(0)
        known = known_covariates(index, temp, self.meta).unsqueeze(0)
        with torch.inference_mode():
            out = self.model(make_batch(load, known)).squeeze(-1).squeeze(0).numpy()
        return out * self.meta["demand_scale"] + self.meta["demand_mean"]

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "nsw_demand_mw": self.demand(history, weather_next).astype(float),
            "nsw_price_up": self.price.predict(history, weather_next),
        }, index=weather_next.index)
