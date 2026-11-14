"""Rebuild artifacts/ from the released CSV only.

Usage:
    python train.py path/to/nem_weather_halfhourly_2010_2026.csv

Demand: the MLP of model.py, refitted on the most recent six years of complete market days
(periodic retraining with a 6-year window, chosen from the historical simulation by the
window-length test), with the Stage 1 settings except the number of epochs: 37 epochs, i.e. the
gradient-update count of 4 years x 55 Stage 1 epochs; AdamW (lr 1e-3, weight decay 1e-4), batch 64,
MAPE loss, seeds 0-4, no validation split. Input scaling is fitted on the training window only and
saved as plain arrays. Takes about 1-2 minutes on 4 CPU threads (68 s measured).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MLP, P, build_inputs, market_day, to_daily  # noqa: E402

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


def main(csv_path):
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
        raise ValueError("non-finite values in the training window")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(out, exist_ok=True)
    for seed in SEEDS:
        net, sc = fit(X[train], R[train], seed)
        torch.save(net.state_dict(), os.path.join(out, f"mlp_seed{seed}.pt"))
        np.savez(os.path.join(out, f"scaler_seed{seed}.npz"), mean=sc.mean_, scale=sc.scale_)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    config = {"model": "MLP 402-256-256-48, ratio target, MAPE loss", "seeds": list(SEEDS),
              "epochs": EPOCHS, "n_inputs": int(X.shape[1]),
              "train_first_day": str(target_days[train][0].date()), "train_last_day": str(end.date()),
              "train_days": int(train.sum()), "torch": torch.__version__}
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"Trained on {config['train_first_day']}..{config['train_last_day']} "
          f"({config['train_days']} days) in {time.time() - t0:.0f}s; saved to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
