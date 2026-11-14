"""Rebuild the four-year demand model and the two-year price model from the released CSV."""
import argparse
import joblib
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from xgboost import XGBClassifier

from model import MARKET, WEATHER, MODEL_FILE, PER_DAY, market_days, price_features
from model import build_model, known_covariates, make_batch

CONFIG = dict(window=480, width=128, weather=True, lr=0.001, holidays=True)
EPOCHS, SEED, YEARS = 36, 42, 4


def train_demand(csv_path, artifact_dir):
    torch.set_num_threads(2)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    days = market_days(df.index)
    counts = days.value_counts()
    end = counts[counts == PER_DAY].index.max()
    start = end - pd.DateOffset(years=YEARS) + pd.Timedelta(days=1)
    size = CONFIG["window"]
    keep = (days >= start - pd.Timedelta(days=size // PER_DAY)) & (days <= end)
    df, days = df.loc[keep], days[keep]
    expected = pd.date_range(df.index[0], df.index[-1], freq="30min")
    if not df.index.equals(expected) or not (days.value_counts() == PER_DAY).all():
        raise ValueError("Expected complete contiguous market days in the training window")
    train = days >= start
    y = df.nsw_demand.to_numpy(dtype=np.float32)
    temp = df.syd_temp.to_numpy(dtype=np.float32)
    meta = dict(config=CONFIG, seed=SEED, epochs=EPOCHS,
                train_start=str(start.date()), train_end=str(end.date()),
                demand_mean=float(y[train].mean()), demand_scale=float(y[train].std()),
                temp_mean=float(temp[train].mean()), temp_scale=float(temp[train].std()))

    load = torch.tensor((y - meta["demand_mean"]) / meta["demand_scale"])
    history = load.unfold(0, size, PER_DAY)[:-1]
    target = load[size:].reshape(-1, PER_DAY)
    known = known_covariates(df.index, temp, meta).unfold(0, size + PER_DAY, PER_DAY).permute(0, 2, 1)
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
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "demand_nbeatsx.pt")
    with (artifact_dir / "demand_nbeatsx.json").open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Trained on {meta['train_start']}..{meta['train_end']} ({len(indices)} days); saved to {artifact_dir}")


def train_price(csv_path, artifact_dir):
    frame = pd.read_csv(csv_path, usecols=["timestamp", *MARKET, *WEATHER],
                        parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    days = market_days(frame.index)
    # Leave seven additional market days for lagged inputs at the window boundary.
    end = days.max() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=2)
    frame = frame.loc[days >= start - pd.Timedelta(days=7)]
    days = market_days(frame.index)
    X = price_features(frame)
    price = frame["nsw_price"]
    previous_mean = price.groupby(days).transform("mean").shift(PER_DAY)
    keep = days >= start
    if price[keep].isna().any() or previous_mean[keep].isna().any():
        raise ValueError("Missing NSW price or comparison mean in the training window.")
    y = (price > previous_mean).astype("int8")
    model = XGBClassifier(n_estimators=300, max_depth=3, min_child_weight=5,
                          learning_rate=0.05, objective="binary:logistic",
                          eval_metric="logloss", tree_method="hist",
                          random_state=42, n_jobs=2)
    model.fit(X.loc[keep], y.loc[keep])
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / MODEL_FILE
    joblib.dump(model, target, compress=3, protocol=4)
    print(f"{start.date()} to {days.max().date()}: {sum(keep)} rows, {X.shape[1]} features")
    print(f"Saved {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / "artifacts"
    train_demand(args.csv_path, directory)
    train_price(args.csv_path, directory)
