"""COMP9417 final model package (Drink Water Group).

Demand: N-BEATSx (Olivares et al., 2023) in plain PyTorch, re-implemented from neuralforecast 3.3.0
because neuralforecast is not available in marking. Selected in Stage 1 with exp2 inputs: 10 daily
history days (480 half-hours) of NSW demand plus known covariates for the history window and the
target day (Sydney temperature, half-hour / weekday / day-of-year sin-cos, NSW public-holiday flags
for the day, the day before and the week before). Final strategy (Stage 3): the periodic retraining
that won the historical simulation, i.e. a refit on the most recent four years of released data.

Price direction: PLACEHOLDER (same half-hour yesterday baseline from harness/dayahead.py) until the
price model is added. Replace `price_direction` below.

train.py imports the feature code from this file, so training and prediction build inputs the
same way.
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

N_PERIODS = 48
COVARIATES = ["temp", "slot_sin", "slot_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos",
              "hol_today", "hol_lag1", "hol_lag7"]


# ----------------------------------------------------------------------------- features

def nsw_holidays(years):
    """NSW public holidays from the holidays package; "Bank Holiday" (banks only) is excluded."""
    import holidays
    table = holidays.Australia(subdiv="NSW", years=years)
    return {pd.Timestamp(day) for day, name in table.items() if "Bank Holiday" not in name}


def known_covariates(index, temp, scalers):
    """Covariates for half-hours stamped `index` (interval end). Calendar inputs use the interval
    start; holiday flags come from the calendar only. Returns float32 [len(index), 10]."""
    start = index - pd.Timedelta(minutes=30)
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


# ----------------------------------------------------------------------------- N-BEATSx

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
        # neuralforecast's layer order: Linear(in, u0) then, for each unit pair, Linear + ReLU.
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
    """Identity-stack N-BEATSx; matches neuralforecast 3.3.0 parameter for parameter."""

    def __init__(self, h, input_size, futr_size=0, n_stacks=3, mlp_units=None, seed=1):
        super().__init__()
        seed_everything(seed)
        mlp_units = mlp_units or 3 * [[512, 512]]
        self.blocks = nn.ModuleList([NBEATSBlock(input_size, h, futr_size, mlp_units)
                                     for _ in range(n_stacks)])

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"].squeeze(-1)
        insample_mask = windows_batch["insample_mask"].squeeze(-1)
        futr_exog = windows_batch["futr_exog"]
        residuals = insample_y.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))
        forecast = insample_y[:, -1:, None]
        for block in self.blocks:
            backcast, block_forecast = block(residuals, futr_exog)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
        return forecast


def build_model(config, seed):
    torch.manual_seed(seed)
    width = config["width"]
    return NBEATSx(h=N_PERIODS, input_size=config["window"], futr_size=len(COVARIATES),
                   mlp_units=[[width, width]] * 3, seed=seed)


def make_batch(load, known):
    """load [B, L] scaled demand; known [B, L+48, C] covariates."""
    history = load.unsqueeze(-1)
    return dict(insample_y=history, insample_mask=torch.ones_like(history), futr_exog=known,
                hist_exog=None, stat_exog=None)


# ----------------------------------------------------------------------------- price (placeholder)

def price_direction(history, weather_next):
    """PLACEHOLDER: same half-hour yesterday above day D's mean price. Replace with the price model."""
    last = history.iloc[-N_PERIODS:]
    return (last["nsw_price"].to_numpy() > last["nsw_price"].mean()).astype(int)


# ----------------------------------------------------------------------------- interface

class Forecaster:
    def __init__(self, artifact_dir: str):
        torch.set_num_threads(2)
        with open(os.path.join(artifact_dir, "demand_nbeatsx.json")) as fh:
            self.meta = json.load(fh)
        self.model = build_model(self.meta["config"], self.meta["seed"])
        state = torch.load(os.path.join(artifact_dir, "demand_nbeatsx.pt"), weights_only=True,
                           map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def demand(self, history, weather_next):
        window = self.meta["config"]["window"]
        if len(weather_next) != N_PERIODS or len(history) < window:
            raise ValueError("Expected 48 target half-hours and at least the input window of history")
        past = history.iloc[-window:]
        y = past["nsw_demand"].to_numpy(dtype=np.float32)
        temp = np.concatenate([past["syd_temp"].to_numpy(np.float32),
                               weather_next["syd_temp"].to_numpy(np.float32)])
        index = past.index.append(weather_next.index)
        # Gaps in recent history (should not happen) are filled from neighbouring half-hours.
        y = pd.Series(y).interpolate(limit_direction="both").to_numpy(np.float32)
        temp = pd.Series(temp).interpolate(limit_direction="both").to_numpy(np.float32)
        load = torch.tensor((y - self.meta["demand_mean"]) / self.meta["demand_scale"]).unsqueeze(0)
        known = known_covariates(index, temp, self.meta).unsqueeze(0)
        with torch.inference_mode():
            out = self.model(make_batch(load, known)).squeeze(-1).squeeze(0).numpy()
        return out * self.meta["demand_scale"] + self.meta["demand_mean"]

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"nsw_demand_mw": self.demand(history, weather_next).astype(float),
                             "nsw_price_up": price_direction(history, weather_next).astype(int)},
                            index=weather_next.index)
