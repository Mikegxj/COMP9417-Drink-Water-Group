"""Stage 1 ablation: add Sydney cloud cover and solar radiation to N-BEATSx (train 2010-2013, validate 2014).

The baseline is the current N-BEATSx exp2 configuration (temperature, calendar, holiday flags). Variants
add syd_cloud, syd_solar or both as known future covariates, standardised on the training window only.
All four variants are re-run here with seeds 42/43/44 in the same environment so comparisons are paired.
Training follows tune_demand.neural_trial: 160-epoch cap (320 if not stopped), patience 20 on 2014 MAPE.

Besides full-year MAPE, errors are split by time of day (midday = intervals starting 10:00-14:30 AEST)
and season, because the motivating gap is at midday. A paired whole-day bootstrap compares each
variant's 3-seed mean forecast with the baseline's.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand.neural import DailyWindows
from experiments.demand.config_search.tune_demand import neural_trial
from harness import dayahead
from experiments.demand import ROOT  # noqa: E402

BASE = dict(window=480, width=128, weather=True, lr=0.001, holidays=True)
VARIANTS = {"exp2": {}, "cloud": dict(cloud=True), "solar": dict(solar=True),
            "cloud_solar": dict(cloud=True, solar=True)}
SEASONS = {12: "summer", 1: "summer", 2: "summer", 3: "autumn", 4: "autumn", 5: "autumn",
           6: "winter", 7: "winter", 8: "winter", 9: "spring", 10: "spring", 11: "spring"}


def tag(frame):
    start = frame.index - pd.Timedelta(minutes=30)
    hour = start.hour + start.minute / 60
    frame = frame.assign(ape=100 * (frame.forecast - frame.actual).abs() / frame.actual.abs(),
                         day=dayahead.market_day(frame.index))
    frame["season"] = frame.day.dt.month.map(SEASONS)
    frame["period"] = np.select([(hour >= 10) & (hour < 15), (hour >= 15) & (hour < 21),
                                 (hour >= 6) & (hour < 10)], ["midday", "evening", "morning"], "night")
    return frame


def grouped(frame):
    rows = [dict(group="all", mape=frame.ape.mean())]
    for key in ("period", "season"):
        rows += [dict(group=f"{key}={k}", mape=v) for k, v in frame.groupby(key).ape.mean().items()]
    midday = frame[frame.period == "midday"]
    rows += [dict(group=f"midday&season={k}", mape=v) for k, v in midday.groupby("season").ape.mean().items()]
    return rows


def bootstrap(base, other, mask, draws=2000, seed=0):
    """Paired whole-day bootstrap of the MAPE difference (other - base) on the selected half-hours."""
    diff = (other.ape - base.ape)[mask]
    daily = diff.groupby(base.day[mask]).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(daily), size=(draws, len(daily)))
    stats = daily["sum"].to_numpy()[pick].sum(1) / daily["count"].to_numpy()[pick].sum(1)
    return diff.mean(), *np.percentile(stats, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=Path, default=ROOT / "results/demand/solar_cloud_ablation")
    args = parser.parse_args()
    output = args.output
    df = dayahead.load_dataset(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    df = df.loc[dayahead.market_day(df.index).year <= 2014]
    runs, groups, means = [], [], {}
    for variant in args.variants:
        config = dict(BASE, **VARIANTS[variant])
        data = DailyWindows(df, config, pd.Timestamp("2010-01-01"),
                            pd.Timestamp("2013-12-31"), pd.Timestamp("2014-12-31"))
        forecasts = []
        for seed in args.seeds:
            folder = output / variant / f"seed_{seed}"
            report = neural_trial("nbeats", config, data, seed, folder, 160)
            if report["stop_reason"] == "budget":
                report = neural_trial("nbeats", config, data, seed, folder, 320)
            frame = tag(pd.read_csv(folder / "predictions.csv.gz", index_col=0, parse_dates=True))
            forecasts.append(frame.forecast)
            runs.append(dict(variant=variant, seed=seed, mape=report["mape"], best_epoch=report["best_epoch"],
                             stop_reason=report["stop_reason"], elapsed_seconds=report["elapsed_seconds"]))
            groups += [dict(variant=variant, seed=seed, **row) for row in grouped(frame)]
            print(variant, seed, round(report["mape"], 4), report["best_epoch"], report["stop_reason"], flush=True)
            pd.DataFrame(runs).to_csv(output / "runs.csv", index=False)
        mean = frame[["actual"]].assign(forecast=pd.concat(forecasts, axis=1).mean(axis=1))
        means[variant] = tag(mean)
        groups += [dict(variant=variant, seed="mean_forecast", **row) for row in grouped(means[variant])]
    groups = pd.DataFrame(groups)
    groups.to_csv(output / "grouped.csv", index=False)
    per_seed = groups[groups.seed != "mean_forecast"]
    table = per_seed.groupby(["group", "variant"]).mape.agg(["mean", "std"]).unstack("variant")
    table.to_csv(output / "summary.csv")
    tests = []
    if "exp2" in means:
        base = means["exp2"]
        masks = {"all": base.ape.notna(), "midday": base.period == "midday",
                 "midday_summer": (base.period == "midday") & (base.season == "summer")}
        for variant, other in means.items():
            if variant == "exp2":
                continue
            for name, mask in masks.items():
                delta, low, high = bootstrap(base, other, mask)
                tests.append(dict(variant=variant, subset=name, delta_mape=delta, ci_low=low, ci_high=high))
        pd.DataFrame(tests).to_csv(output / "bootstrap_vs_exp2.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(dict(
        train="2010-2013", validation="2014", base_config=BASE, variants=VARIANTS, seeds=args.seeds,
        midday="interval start 10:00-14:30 AEST", seasons="by market-day month, summer = Dec-Feb",
        bootstrap="paired whole-day resampling of 3-seed mean forecasts, 2000 draws, 95% percentile",
        selection_rule="mean 2014 MAPE over seeds; keep the simpler version if the paired CI contains 0"),
        indent=2))
    print(table.round(3).to_string())
    if tests:
        print(pd.DataFrame(tests).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
