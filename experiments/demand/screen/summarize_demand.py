"""Summarize all first-round configurations and frozen/adapted forecasts."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.demand import ROOT
from harness import dayahead


def summarize(output):
    screen = output / "screen"
    trials = []
    for path in screen.rglob("*_scores.csv"):
        name = path.name.removesuffix("_scores.csv")
        variant = "base" if path.parent == screen else path.parent.name
        if name in ("yesterday", "last_week") and variant != "base":
            continue
        total = pd.read_csv(path).iloc[0]
        trials.append(dict(model=name, variant=variant, n=total.n, mape=total.mape, mae=total.mae))
    pd.DataFrame(trials).sort_values("mape").to_csv(screen / "comparison.csv", index=False)
    months, years, ranks = [], [], []
    freeze = output / "frozen"
    paths = list(freeze.rglob("*_predictions.csv.gz")) + list((output / "adaptation").glob("*_predictions.csv.gz"))
    periodic = output / "periodic"
    paths.extend(periodic.glob("*_predictions.csv.gz"))
    for path in sorted(paths):
        model = path.name.removesuffix("_predictions.csv.gz")
        if model in ("last_week", "yesterday") and path.parent != freeze:
            continue
        strategy = "frozen"
        if path.parent.name == "adaptation":
            strategy, model = model, "lightgbm"
            if strategy == "periodic" and (periodic / "lightgbm_predictions.csv.gz").exists():
                continue
        elif path.parent == periodic:
            strategy = "periodic"
        rows = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        days = dayahead.market_day(rows.index)
        if not (days.value_counts() == 48).all() or rows.isna().any().any():
            raise ValueError(f"Incomplete forecasts: {path}")
        ape = 100 * (rows.actual - rows.forecast).abs() / rows.actual
        for period, group in rows.groupby(days.year):
            years.append(dict(model=model, strategy=strategy, year=period, n=len(group),
                              mape=dayahead.mape(group.actual, group.forecast),
                              mae=dayahead.mae(group.actual, group.forecast)))
        for period, group in ape.groupby(days.to_period("M")):
            day_means = group.to_numpy().reshape(-1, 48).mean(axis=1)
            rng = np.random.default_rng(42)
            bootstrap = rng.choice(day_means, size=(1000, len(day_means))).mean(axis=1)
            margin = 1.96 * group.std() / np.sqrt(len(group))
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            months.append(dict(model=model, strategy=strategy, month=str(period), n=len(group),
                               mape=group.mean(), mae=(rows.actual[group.index]-rows.forecast[group.index]).abs().mean(),
                               independent_low=group.mean()-margin, independent_high=group.mean()+margin,
                               day_bootstrap_low=low, day_bootstrap_high=high))
        for period, mask in [("selection_2015_2024", days.year < 2025),
                             ("check_2025_2026_10", days.year >= 2025)]:
            ranks.append(dict(model=model, strategy=strategy, period=period, n=int(mask.sum()),
                              mape=ape[mask].mean(), mae=dayahead.mae(rows.actual[mask], rows.forecast[mask])))
    for name, records in [("monthly", months), ("annual", years), ("historical_comparison", ranks)]:
        if records:
            pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/demand")
    summarize(parser.parse_args().output)
