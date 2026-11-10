"""Rebuild artifacts/ from the released CSV only.

Usage (from the kit's root folder):
    python submission/train.py data/nem_weather_halfhourly_2010_2026.csv

Demand: N-BEATSx, Stage 1 configuration (exp2 inputs), refit on the last four years of complete
market days with the fixed number of epochs and seed used by the periodic retraining in the
historical simulation (no validation split, no early stopping). About 5-10 minutes on 2 CPU threads.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import N_PERIODS, build_model, known_covariates, make_batch  # noqa: E402

CONFIG = dict(window=480, width=128, weather=True, lr=0.001, holidays=True)
EPOCHS, SEED, YEARS = 36, 42, 4


def market_day(index):
    return (index - pd.Timedelta("30min")).normalize()


def main(csv_path):
    torch.set_num_threads(2)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    days = market_day(df.index)
    counts = days.value_counts()
    end = counts[counts == N_PERIODS].index.max()                  # last complete market day
    start = end - pd.DateOffset(years=YEARS) + pd.Timedelta(days=1)  # e.g. 2022-11-01..2026-10-31
    size = CONFIG["window"]
    keep = (days >= start - pd.Timedelta(days=size // N_PERIODS)) & (days <= end)
    df, days = df.loc[keep], days[keep]
    expected = pd.date_range(df.index[0], df.index[-1], freq="30min")
    if not df.index.equals(expected) or not (days.value_counts() == N_PERIODS).all():
        raise ValueError("Expected complete contiguous market days in the training window")
    train = days >= start
    y = df.nsw_demand.to_numpy(dtype=np.float32)
    temp = df.syd_temp.to_numpy(dtype=np.float32)
    meta = dict(config=CONFIG, seed=SEED, epochs=EPOCHS,
                train_start=str(start.date()), train_end=str(end.date()),
                demand_mean=float(y[train].mean()), demand_scale=float(y[train].std()),
                temp_mean=float(temp[train].mean()), temp_scale=float(temp[train].std()))

    load = torch.tensor((y - meta["demand_mean"]) / meta["demand_scale"])
    history = load.unfold(0, size, N_PERIODS)[:-1]                  # [days, L]
    target = load[size:].reshape(-1, N_PERIODS)                      # [days, 48]
    known = known_covariates(df.index, temp, meta).unfold(0, size + N_PERIODS, N_PERIODS).permute(0, 2, 1)
    indices = torch.arange(len(target))

    model = build_model(CONFIG, SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    for epoch in range(EPOCHS):
        model.train()
        total = 0.
        for batch in indices[torch.randperm(len(indices))].split(32):
            optimizer.zero_grad()
            loss = F.l1_loss(model(make_batch(history[batch], known[batch])).squeeze(-1), target[batch])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += loss.item() * len(batch)
        print(f"epoch {epoch + 1}/{EPOCHS} train MAE {total / len(indices) * meta['demand_scale']:.1f} MW", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "demand_nbeatsx.pt"))
    with open(os.path.join(out, "demand_nbeatsx.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Trained on {meta['train_start']}..{meta['train_end']} ({len(indices)} days); saved to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
