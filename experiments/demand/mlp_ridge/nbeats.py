"""N-BEATSx as used by the teammate (Cattie Su, group repo submission/model.py + train.py, commit
d3e812e): identity-stack N-BEATSx re-implemented in plain PyTorch from neuralforecast 3.3.0, with
the same inputs and training settings, so it can be run under the shared protocol here.

Inputs for a target day: the previous 480 half-hours of NSW demand (scaled with the training
window's mean/std) and 10 known covariates over those 480 + the 48 target half-hours: Sydney
temperature (scaled), half-hour / weekday / day-of-year sin-cos, NSW public-holiday flags for the
day, the day before and the week before ("Bank Holiday" excluded). Training: Adam lr 1e-3,
batch 32, L1 loss on scaled demand, gradient clipping at 1, 36 epochs, no validation.
"""
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as Fn

P, WINDOW, WIDTH, EPOCHS, LR = 48, 480, 128, 36, 1e-3


def nsw_holidays(years):
    import holidays
    table = holidays.Australia(subdiv="NSW", years=years)
    return {pd.Timestamp(day) for day, name in table.items() if "Bank Holiday" not in name}


def raw_covariates(index):
    """Calendar covariates for half-hours stamped `index` (temperature added later)."""
    start = index - pd.Timedelta(minutes=30)
    cov = []
    for pos, period in [(start.hour * 2 + start.minute // 30, 48), (start.dayofweek, 7), (start.dayofyear - 1, 365.25)]:
        cov += [np.sin(2 * np.pi * pos / period), np.cos(2 * np.pi * pos / period)]
    day = start.normalize()
    known = nsw_holidays(range(day.year.min() - 1, day.year.max() + 2))
    for off in (0, 1, 7):
        shifted = day - pd.Timedelta(days=off)
        cov.append(np.fromiter((d in known for d in shifted), bool, len(shifted)).astype(np.float32))
    return np.column_stack(cov).astype(np.float32)


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size

    def forward(self, theta):
        return theta[:, :self.backcast_size], theta[:, self.backcast_size:].reshape(len(theta), -1, 1)


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
    def __init__(self, h, input_size, futr_size, mlp_units, n_stacks=3, seed=1):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.blocks = nn.ModuleList([NBEATSBlock(input_size, h, futr_size, mlp_units) for _ in range(n_stacks)])

    def forward(self, y, futr):
        residuals = y.flip(dims=(-1,))
        forecast = y[:, -1:, None]
        for block in self.blocks:
            backcast, f = block(residuals, futr)
            residuals = residuals - backcast
            forecast = forecast + f
        return forecast.squeeze(-1)


def fit_predict(y, temp, cov, train_days, seed, threads=3, extra=None):
    """y, temp: demand / temperature per half-hour [n_days*48]; cov: calendar covariates
    [n_days*48, 9]; train_days: boolean over market days (targets used for training, each needs
    10 earlier days); extra: optional further known weather covariates [n_days*48, k] (e.g. Sydney
    cloud and solar), standardised on the training rows like temperature and placed after it
    (extra=None reproduces the teammate's configuration exactly).
    Returns MW predictions [n_days, 48] (NaN for the first 10 days)."""
    torch.set_num_threads(threads)
    n_days = len(y) // P
    rows = np.repeat(train_days, P)
    dm, ds = float(y[rows].mean()), float(y[rows].std())
    tm, ts = float(temp[rows].mean()), float(temp[rows].std())
    load = torch.tensor((y - dm) / ds, dtype=torch.float32)
    wx = [(temp - tm) / ts]
    if extra is not None:
        extra = np.asarray(extra, dtype=float).reshape(len(y), -1)
        em, es = extra[rows].mean(axis=0), extra[rows].std(axis=0)
        wx.append((extra - em) / es)
    known = torch.tensor(np.column_stack(wx + [cov]), dtype=torch.float32)
    days = np.arange(WINDOW // P, n_days)
    hist = torch.stack([load[d * P - WINDOW:d * P] for d in days])
    futr = torch.stack([known[d * P - WINDOW:(d + 1) * P] for d in days])
    target = torch.stack([load[d * P:(d + 1) * P] for d in days])
    tr = torch.tensor(train_days[days])
    torch.manual_seed(seed)
    model = NBEATSx(P, WINDOW, known.shape[1], [[WIDTH, WIDTH]] * 3, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xi, fi, ti = hist[tr], futr[tr], target[tr]
    for _ in range(EPOCHS):
        model.train()
        for b in torch.randperm(len(xi)).split(32):
            opt.zero_grad()
            loss = Fn.l1_loss(model(xi[b], fi[b]), ti[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
    model.eval()
    out = np.full((n_days, P), np.nan)
    with torch.inference_mode():
        for s in range(0, len(days), 512):
            out[days[s:s + 512]] = model(hist[s:s + 512], futr[s:s + 512]).numpy() * ds + dm
    return out
