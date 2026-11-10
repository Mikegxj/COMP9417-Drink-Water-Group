"""Run from the repository root: python -m experiments.demand.screen.run_demand --stage screen"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from harness import dayahead
from src.demand import tabular as demand_tabular

from experiments.demand import ROOT  # noqa: E402
MODELS = ["gam", "lightgbm", "nbeats", "tft", "tcn", "lstm"]


def run(args):
    df = dayahead.load_dataset(args.data)
    if not df.index.equals(pd.date_range(df.index[0], df.index[-1], freq="30min")):
        raise ValueError("Expected contiguous half-hour intervals")
    days = dayahead.market_day(df.index)
    if not (days.value_counts() == 48).all():
        raise ValueError("Incomplete market day")
    cutoff = 2014 if args.stage == "screen" else 2015
    end = pd.Timestamp("2014-12-31" if args.stage == "screen" else "2026-10-31")
    df = df.loc[days <= end]
    days = dayahead.market_day(df.index)
    train, test = days.year < cutoff, days.year >= cutoff
    expected = pd.date_range(f"{cutoff}-01-01", end)
    if not days[test].unique().equals(expected):
        raise ValueError("Incomplete evaluation period")
    x = demand_tabular.features(df, args.variant)
    y = df.nsw_demand
    folder = args.output / args.stage
    if args.variant != "base":
        folder = folder / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["yesterday", "last_week", *args.models]:
        start = time.monotonic()
        if name in ("yesterday", "last_week"):
            col = "lag1" if name == "yesterday" else "lag7"
            pred = x.loc[test, col].to_numpy()
        elif name in MODELS[:3]:
            pred = demand_tabular.forecast(name, x, y, train, test)
        else:
            from src.demand.compact_neural import forecast
            pred = forecast(name, df, days, cutoff, args.epochs, args.device, args.variant)
        if pred.shape != (sum(test),) or not np.isfinite(pred).all():
            raise ValueError(f"Invalid predictions from {name}")
        rows = pd.DataFrame({"actual": y[test], "forecast": pred})
        rows.index.name = "timestamp"
        rows.to_csv(folder / f"{name}_predictions.csv.gz", compression="gzip")
        scores = []
        for group, periods in [("overall", np.repeat("all", len(rows))),
                               ("year", days[test].year), ("month", days[test].to_period("M"))]:
            for period, data in rows.groupby(periods):
                scores.append(dict(group=group, period=str(period), n=len(data),
                                   mape=dayahead.mape(data.actual, data.forecast),
                                   mae=dayahead.mae(data.actual, data.forecast)))
        pd.DataFrame(scores).to_csv(folder / f"{name}_scores.csv", index=False)
        meta = dict(model=name, stage=args.stage, train_end=cutoff-1, seed=42, variant=args.variant,
                    epochs=args.epochs if name in MODELS[3:] else None, device=args.device,
                    elapsed_seconds=round(time.monotonic()-start, 2))
        (folder / f"{name}_run.json").write_text(json.dumps(meta, indent=2))
        print(name, scores[0], meta["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screen", "frozen"], default="screen")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--variant", choices=["base", "no_target_temp", "seven_lags", "prior_extrema",
                                               "humidity", "cloud", "solar", "cloud_solar",
                                               "mel_temp", "bne_temp", "calendar_only", "calendar_temp"], default="base")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results/demand")
    parser.add_argument("--data", type=Path, default=ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.variant in ("calendar_only", "calendar_temp"):
        if any(name in MODELS[:3] for name in args.models):
            parser.error("sequence feature variants require neural models")
    elif args.variant != "base" and any(name in MODELS[3:] for name in args.models):
        parser.error("tabular feature variants require tabular models")
    run(args)
