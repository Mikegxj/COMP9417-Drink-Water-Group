"""2010–2013 training, 2014 validation curves with best-checkpoint restoration."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.demand.compact_neural import build_model, day_tensors
from experiments.demand import ROOT
from harness import dayahead

CASES = {
    "nbeats": ("nbeats", "base", "compact generic N-BEATS"),
    "nbeats_exogenous": ("nbeats", "calendar_temp", "custom N-BEATS with exogenous forecast head"),
    "tft_simplified": ("tft", "base", "simplified TFT-inspired network"),
    "tcn": ("tcn", "base", "residual dilated TCN with lag skip inputs"),
    "lstm": ("lstm", "base", "LSTM with lag skip inputs"),
}


def predict(model, history, future, indices):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(history[b], future[b]) for b in indices.split(64)])


def experiment(case, seed, tensors, index, args):
    history, future, target, count, mean, scale = tensors
    name, variant, description = CASES[case]
    torch.manual_seed(seed)
    model = build_model(name, future.size(-1), variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    folder = args.output / case / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    best, best_epoch = float("inf"), 0
    start = time.monotonic()
    train_indices = torch.arange(count)
    val_indices = torch.arange(count, len(target))
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for batch in torch.randperm(count).split(32):
            optimizer.zero_grad()
            loss = F.l1_loss(model(history[batch], future[batch]), target[batch])
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss for {case}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_pred = predict(model, history, future, train_indices) * scale + mean
        val_pred = predict(model, history, future, val_indices) * scale + mean
        train_y, val_y = target[:count] * scale + mean, target[count:] * scale + mean
        val_mape = dayahead.mape(val_y.numpy(), val_pred.numpy())
        row = dict(epoch=epoch, train_mae=float((train_pred-train_y).abs().mean()),
                   train_mape=dayahead.mape(train_y.numpy(), train_pred.numpy()),
                   val_mae=float((val_pred-val_y).abs().mean()), val_mape=val_mape,
                   elapsed_seconds=time.monotonic()-start)
        rows.append(row)
        if val_mape < best:
            best, best_epoch = val_mape, epoch
            torch.save(model.state_dict(), folder / "best.pt")
        pd.DataFrame(rows).to_csv(folder / "curve.csv", index=False)
        if epoch % 10 == 0:
            print(case, seed, epoch, "best", round(best, 4), "at", best_epoch, flush=True)
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
    restored = predict(model, history, future, val_indices).numpy().ravel() * scale + mean
    actual = val_y.numpy().ravel()
    assert abs(dayahead.mape(actual, restored) - best) < 1e-4
    frame = pd.DataFrame({"actual": actual, "forecast": restored}, index=index)
    frame.index.name = "timestamp"
    frame.to_csv(folder / "validation_predictions.csv.gz")
    best_row = rows[best_epoch-1]
    record = dict(case=case, architecture=description, seed=seed, epochs_run=len(rows),
                  best_epoch=best_epoch, val_mape=best, val_mae=best_row["val_mae"],
                  epoch16_mape=rows[15]["val_mape"] if len(rows) >= 16 else None,
                  stop_reason="patience" if len(rows)-best_epoch >= args.patience else "epoch_limit",
                  elapsed_seconds=time.monotonic()-start, train_days=count, validation_days=len(val_indices),
                  target_mean=float(mean), target_scale=float(scale),
                  max_epochs=args.max_epochs, patience=args.patience,
                  batch_size=32, learning_rate=0.002, loss="MAE", history_points=336,
                  torch_version=torch.__version__,
                  source_sha256=hashlib.sha256((ROOT/"src/demand/compact_neural.py").read_bytes()).hexdigest())
    (folder / "summary.json").write_text(json.dumps(record, indent=2))
    return record


def main(args):
    torch.set_num_threads(2)
    df = dayahead.load_dataset(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    days = dayahead.market_day(df.index)
    df, days = df.loc[days.year <= 2014], days[days.year <= 2014]
    tensors = day_tensors(df, days, 2014)
    records = []
    for case in args.models:
        for seed in args.seeds:
            records.append(experiment(case, seed, tensors, df.index[days.year == 2014], args))
            pd.DataFrame(records).to_csv(args.output / "summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "results/demand/convergence")
    args = parser.parse_args()
    if args.max_epochs < 16 or args.patience < 1:
        parser.error("max-epochs must be >=16 and patience positive")
    main(args)
