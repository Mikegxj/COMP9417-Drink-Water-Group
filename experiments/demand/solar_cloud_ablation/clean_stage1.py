"""Stage 1 re-run with 2014 kept out of training and early stopping (same procedure as the MLP).

For each variant and seed:
  1. train on 2010-2012 and early-stop on 2013 (tune_demand.neural_trial: 160-epoch cap, 320 if not
     stopped, patience 20 on 2013 MAPE) -> best epoch E;
  2. refit on 2010-2013 for E epochs with the same seed (formal_backtest.neural_refit);
  3. predict 2014. 2014 is used only to report and compare, never to choose epochs.
The earlier Stage 1 numbers (sections 10, 15, 16 of the experiment log) picked the best epoch on 2014
itself, which made single-network 2014 MAPE about 0.11 points optimistic.

    python -m experiments.demand.solar_cloud_ablation.clean_stage1 --variant solar --seed 42
    python -m experiments.demand.solar_cloud_ablation.clean_stage1 --summarize

Section 18 adds one MLP-inspired change at a time on top of the solar variant (seeds 42-44, paired with
solar seeds 42-44): per-sample 7-day level scaling, MAPE loss, Sydney humidity, Christmas flag and
daily temperature statistics. Rule fixed before the runs: a change goes on to the backtest unless its
2014 3-net paired bootstrap shows it significantly worse (CI entirely above 0).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.demand import ROOT
from experiments.demand.backtest.formal_backtest import neural_refit
from experiments.demand.config_search.tune_demand import neural_trial
from experiments.demand.solar_cloud_ablation.solar_cloud_ablation import tag, grouped, bootstrap
from harness import dayahead
from src.demand.neural import DailyWindows

OUTPUT = ROOT / "results/demand/clean_stage1"
BASE = dict(window=480, width=128, weather=True, lr=0.001, holidays=True)
VARIANTS = {"exp2": dict(config=BASE, seeds=[42, 43, 44]),
            "solar": dict(config=dict(BASE, solar=True), seeds=list(range(42, 50)))}
SOLAR = VARIANTS["solar"]["config"]
CHANGES = {"level": dict(level=True), "mape": dict(loss="mape"), "rh": dict(rh=True),
           "xmas": dict(xmas=True), "tempstats": dict(tempstats=True)}
VARIANTS.update({f"solar_{k}": dict(config=dict(SOLAR, **v), seeds=[42, 43, 44]) for k, v in CHANGES.items()})
# Humidity was the only change kept after the backtest; extended to the same 8 seeds as solar.
VARIANTS["solar_rh"]["seeds"] = list(range(42, 50))
COMPARISONS = ([("exp2", "solar", 3)] + [("solar", f"solar_{k}", 3) for k in CHANGES]
               + [("solar", "solar_rh", 8)])


def run(variant, seed):
    config = VARIANTS[variant]["config"]
    folder = OUTPUT / variant / f"seed_{seed}"
    df = dayahead.load_dataset(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    days = dayahead.market_day(df.index)
    inner = DailyWindows(df.loc[days.year <= 2013], config, pd.Timestamp("2010-01-01"),
                         pd.Timestamp("2012-12-31"), pd.Timestamp("2013-12-31"))
    report = neural_trial("nbeats", config, inner, seed, folder / "inner", 160)
    if report["stop_reason"] == "budget":
        report = neural_trial("nbeats", config, inner, seed, folder / "inner", 320)
    epochs = report["best_epoch"]
    (folder / "refit").mkdir(parents=True, exist_ok=True)
    pred = neural_refit("nbeats", config, epochs, df.loc[days.year <= 2014], pd.Timestamp("2010-01-01"),
                        pd.Timestamp("2013-12-31"), pd.Timestamp("2014-12-31"), folder / "refit", seed=seed)
    frame = pd.DataFrame(dict(actual=df.nsw_demand.loc[pred.index], forecast=pred))
    frame.index.name = "timestamp"
    if len(frame) != 365 * 48 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Unexpected 2014 predictions")
    frame.to_csv(folder / "predictions_2014.csv.gz")
    summary = dict(variant=variant, seed=seed, inner_best_epoch=epochs, inner_2013_mape=report["mape"],
                   inner_stop_reason=report["stop_reason"],
                   mape_2014=dayahead.mape(frame.actual, frame.forecast),
                   mae_2014=dayahead.mae(frame.actual, frame.forecast))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


def summarize():
    runs, groups, means = [], [], {}
    for variant, spec in VARIANTS.items():
        if not all((OUTPUT / variant / f"seed_{s}/summary.json").exists() for s in spec["seeds"]):
            print(f"skipping {variant}: not all seeds finished")
            continue
        frames = []
        for seed in spec["seeds"]:
            folder = OUTPUT / variant / f"seed_{seed}"
            runs.append(json.loads((folder / "summary.json").read_text()))
            frames.append(tag(pd.read_csv(folder / "predictions_2014.csv.gz", index_col=0, parse_dates=True)))
            groups += [dict(variant=variant, members=1, seed=seed, **row) for row in grouped(frames[-1])]
        for k in sorted({3, len(frames)}):
            mean = tag(frames[0][["actual"]].assign(
                forecast=pd.concat([f.forecast for f in frames[:k]], axis=1).mean(axis=1)))
            means[(variant, k)] = mean
            groups += [dict(variant=variant, members=k, seed=f"mean_{spec['seeds'][0]}-{spec['seeds'][k-1]}",
                            **row) for row in grouped(mean)]
    runs, groups = pd.DataFrame(runs), pd.DataFrame(groups)
    runs.to_csv(OUTPUT / "runs.csv", index=False)
    groups.to_csv(OUTPUT / "grouped.csv", index=False)
    tests = []
    for reference, variant, k in COMPARISONS:
        if (reference, k) not in means or (variant, k) not in means:
            continue
        base, other = means[(reference, k)], means[(variant, k)]
        for name, mask in {"all": base.ape.notna(), "midday": base.period == "midday",
                           "midday_summer": (base.period == "midday") & (base.season == "summer")}.items():
            delta, low, high = bootstrap(base, other, mask)
            tests.append(dict(comparison=f"{variant} {k}-net - {reference} {k}-net", subset=name,
                              delta_mape=delta, ci_low=low, ci_high=high))
    tests = pd.DataFrame(tests)
    tests.to_csv(OUTPUT / "bootstrap.csv", index=False)
    (OUTPUT / "protocol.json").write_text(json.dumps(dict(
        inner="train 2010-2012, early stopping on 2013 (cap 160/320, patience 20)",
        refit="2010-2013 for the inner best epoch, same seed", evaluation="2014 only, not used for any choice",
        section18_rule="backtest a change unless its 2014 3-net paired CI lies entirely above 0",
        variants={k: dict(config=v["config"], seeds=v["seeds"]) for k, v in VARIANTS.items()}), indent=2))
    print(runs.round(3).to_string(index=False))
    print(runs.groupby("variant")[["mape_2014", "inner_best_epoch"]].agg(["mean", "std", "median"]).round(3).to_string())
    print(groups[groups.group.isin(["all", "period=midday", "midday&season=summer"])]
          .pivot_table(index=["variant", "members"], columns="group", values="mape").round(3).to_string())
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args.variant, args.seed)
