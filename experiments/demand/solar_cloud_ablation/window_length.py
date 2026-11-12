"""Periodic retraining window length for N-BEATSx + solar (section 20 of the experiment log).

Same periodic backtest as solar_backtest.py --protocol clean (refits at the start of 2017/19/21/23/25),
but each refit trains on the previous L years instead of four: L = 2, 3, 6, or all history since 2010.
The 2015 fit (2010-2014, used for 2015-2016) is unchanged and copied from the 4-year run, so the
variants differ only from 2017 on.

Epochs keep the number of gradient steps of the 4-year refit (39 epochs x 4 years): epochs =
round(39 * 4 / L). L was never part of Stage 1 (the inner split has only three years before 2013), so
per-length early stopping is not possible; equal steps is the rule fixed before the runs.

Rule fixed before the runs: compare each 3-net mean (seeds 42-44) with the 4-year 3-net mean on
periodic 2017-2024 (paired whole-day bootstrap). Switch only if the interval lies entirely below 0;
2025-2026.10 was already viewed in earlier sections and is reported, not used to choose.

    python -m experiments.demand.solar_cloud_ablation.window_length --years 2 --seed 42
    python -m experiments.demand.solar_cloud_ablation.window_length --summarize
"""
import argparse
import json
import shutil

import numpy as np
import pandas as pd

from experiments.demand import ROOT
from experiments.demand.backtest.formal_backtest import run_model
from experiments.demand.solar_cloud_ablation.solar_backtest import daily_bootstrap
from experiments.demand.solar_cloud_ablation.solar_cloud_ablation import tag
from harness import dayahead

OUTPUT = ROOT / "results/demand/window_length"
FOUR_YEAR = ROOT / "results/demand/solar_backtest_clean/solar"
CONFIG = dict(window=480, width=128, weather=True, lr=0.001, holidays=True, solar=True)
EPOCHS, SEEDS, REFITS = 39, [42, 43, 44], range(2017, 2026, 2)
# 6 years passed the rule; extended to the 8 seeds of the 4-year ensemble for confirmation.
ENSEMBLE_SEEDS = list(range(42, 50))
LENGTHS = ["2", "3", "6", "all"]
PERIODS = {"2017–2024": lambda y: (y >= 2017) & (y < 2025), "2025–2026.10": lambda y: y >= 2025,
           "2017–2026.10": lambda y: y >= 2017}


def epochs_for(length, year):
    years = year - 2010 if length == "all" else int(length)
    return round(EPOCHS * 4 / years)


def run(length, seed):
    root = OUTPUT / f"years_{length}" / f"seed_{seed}"
    (root / "nbeats").mkdir(parents=True, exist_ok=True)
    selected = dict(model="nbeats", config=CONFIG, refit_epochs=EPOCHS, refit_seed=seed, device="cpu",
                    train_years=length if length == "all" else int(length),
                    refit_epochs_by_year={str(y): epochs_for(length, y) for y in REFITS})
    (root / "nbeats/selected.json").write_text(json.dumps(selected, indent=2))
    shared = root / "backtest/nbeats/2015"
    if not (shared / "predictions.csv.gz").exists():
        shared.mkdir(parents=True, exist_ok=True)
        for name in ("predictions.csv.gz", "fit.json"):
            shutil.copy(FOUR_YEAR / f"seed_{seed}/backtest/nbeats/2015" / name, shared / name)
    df = dayahead.load_dataset(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    run_model("nbeats", df, root)


def load(path):
    return tag(pd.read_csv(path, index_col="timestamp", parse_dates=True))


def summarize(seeds=SEEDS, lengths=LENGTHS, suffix=""):
    variants = {"4": [FOUR_YEAR / f"seed_{s}/backtest/nbeats/periodic_predictions.csv.gz" for s in seeds]}
    variants.update({k: [OUTPUT / f"years_{k}/seed_{s}/backtest/nbeats/periodic_predictions.csv.gz" for s in seeds]
                     for k in lengths})
    rows, tests, means = [], [], {}
    for key, paths in variants.items():
        if not all(p.exists() for p in paths):
            print(f"skipping {key} years: not all seeds finished")
            continue
        frames = [load(p) for p in paths]
        means[key] = tag(frames[0][["actual"]].assign(
            forecast=pd.concat([f.forecast for f in frames], axis=1).mean(axis=1)))
        year = means[key].day.dt.year
        for label, rule in PERIODS.items():
            for part, extra in ((label, True), (f"{label} midday", means[key].period == "midday")):
                mask = rule(year) & extra
                rows.append(dict(years=key, subset=part, single=np.mean([f.ape[mask].mean() for f in frames]),
                                 net3=means[key].ape[mask].mean()))
        rows += [dict(years=key, subset=str(y), single=np.mean([f.ape[year == y].mean() for f in frames]),
                      net3=g.ape.mean()) for y, g in means[key].groupby(year) if y >= 2017]
    base = means["4"]
    year = base.day.dt.year
    for key, other in means.items():
        if key == "4":
            continue
        if not other.index.equals(base.index):
            raise ValueError("Predictions are not aligned")
        for label, rule in PERIODS.items():
            for part, extra in ((label, True), (f"{label} midday", base.period == "midday")):
                delta, low, high = daily_bootstrap(base, other, rule(year) & extra)
                tests.append(dict(comparison=f"{key} years - 4 years ({len(seeds)}-net)", subset=part,
                                  delta_mape=delta, ci_low=low, ci_high=high))
    rows, tests = pd.DataFrame(rows), pd.DataFrame(tests)
    rows.to_csv(OUTPUT / f"comparison{suffix}.csv", index=False)
    tests.to_csv(OUTPUT / f"bootstrap_vs_4years{suffix}.csv", index=False)
    if suffix:
        print(rows.pivot_table(index="subset", columns="years", values="net3", sort=False).round(3).to_string())
        print(tests.round(3).to_string(index=False))
        return
    (OUTPUT / "protocol.json").write_text(json.dumps(dict(
        config=CONFIG, seeds=SEEDS, lengths=["4", *LENGTHS],
        epochs={k: {y: epochs_for(k, y) for y in REFITS} for k in ["4", *LENGTHS]},
        shared="2015 fit (2010-2014, 39 epochs) identical for all lengths; periodic refits from 2017",
        rule="switch only if the 2017-2024 3-net paired CI lies entirely below 0; 2025-2026.10 reported only"),
        indent=2))
    print(rows.pivot_table(index="subset", columns="years", values="net3", sort=False).round(3).to_string())
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", choices=LENGTHS)
    parser.add_argument("--seed", type=int, choices=ENSEMBLE_SEEDS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize()
        if all((OUTPUT / f"years_6/seed_{s}/backtest/nbeats/periodic_predictions.csv.gz").exists()
               for s in ENSEMBLE_SEEDS):
            summarize(ENSEMBLE_SEEDS, ["6"], "_8net")
    else:
        run(args.years, args.seed)
