"""Rebuild the five-seed demand MLP and relative-price XGBoost from the released CSV.

Run from the course kit root:
    python submission/train.py data/nem_weather_halfhourly_2010_2026.csv
Demand uses the latest six years. Price uses the latest two years.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from model import MARKET, WEATHER, MODEL_FILE, PER_DAY, market_days, price_features
from model import MLP, P, build_inputs, market_day, to_daily

SEEDS, EPOCHS, YEARS, BATCH = (0, 1, 2, 3, 4), 37, 6, 64


def fit(X, R, seed):
    torch.manual_seed(seed)
    sc = StandardScaler().fit(X)
    xt = torch.tensor(sc.transform(X), dtype=torch.float32)
    rt = torch.tensor(R, dtype=torch.float32)
    net = MLP(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(xt), generator=g)
        for b in range(0, len(xt), BATCH):
            ib = perm[b:b + BATCH]
            opt.zero_grad()
            loss = (torch.abs(net(xt[ib]) - rt[ib]) / rt[ib]).mean()
            loss.backward()
            opt.step()
    return net, sc


def train_demand(csv_path, artifact_dir):
    torch.set_num_threads(4)
    t0 = time.time()
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    counts = market_day(df.index).value_counts()
    complete = counts[counts == P].index
    df = df[market_day(df.index).isin(complete)]
    days, arr = to_daily(df)
    X, level, i = build_inputs(days, arr)
    R = arr["nsw_demand"][i] / level[:, None]
    end = days[-1]
    target_days = days[i]
    train = np.asarray((target_days > end - pd.DateOffset(years=YEARS)) & (target_days <= end))
    if not np.all(np.isfinite(X[train])) or not np.all(np.isfinite(R[train])):
        raise ValueError("Non-finite values in the training window")
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        net, sc = fit(X[train], R[train], seed)
        torch.save(net.state_dict(), out / f"mlp_seed{seed}.pt")
        np.savez(out / f"scaler_seed{seed}.npz", mean=sc.mean_, scale=sc.scale_)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    config = {"model": "MLP 402-256-256-48, ratio target, MAPE loss", "seeds": list(SEEDS),
              "epochs": EPOCHS, "n_inputs": int(X.shape[1]),
              "train_first_day": str(target_days[train][0].date()), "train_last_day": str(end.date()),
              "train_days": int(train.sum()), "torch": torch.__version__}
    with (out / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2)
    print(f"Trained demand on {config['train_first_day']}..{config['train_last_day']} "
          f"({config['train_days']} days); saved to {out}")


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
