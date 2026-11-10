"""Stage 1 holiday ablation for N-BEATSx and TFT (train 2010-2013, validate 2014).

Each model keeps its formal Stage 1 configuration; only the exp2 calendar flags
(hol_today, hol_lag1, hol_lag7, as known future covariates) are added. Baseline and
exp2 are both re-run here with seeds 42/43/44 in the same environment, so the
comparison is paired and free of cross-hardware differences. Training follows
tune_demand.neural_trial: 160-epoch cap (320 if not stopped), patience 20 on 2014 MAPE.
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
CONFIGS = {"nbeats": dict(window=480, width=128, weather=True, lr=0.001),
           "tft": dict(window=336, width=32, weather=True, lr=0.002)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=Path, default=ROOT / "results/demand/holiday_ablation_neural")
    args = parser.parse_args()
    df = dayahead.load_dataset(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    df = df.loc[dayahead.market_day(df.index).year <= 2014]
    rows = []
    for name in args.models:
        for variant, holidays in (("exp0", False), ("exp2", True)):
            config = dict(CONFIGS[name], holidays=holidays)
            data = DailyWindows(df, config, pd.Timestamp("2010-01-01"),
                                pd.Timestamp("2013-12-31"), pd.Timestamp("2014-12-31"))
            for seed in args.seeds:
                folder = args.output / name / variant / f"seed_{seed}"
                report = neural_trial(name, config, data, seed, folder, 160)
                if report["stop_reason"] == "budget":
                    report = neural_trial(name, config, data, seed, folder, 320)
                rows.append(dict(model=name, variant=variant, seed=seed, mape=report["mape"],
                                 best_epoch=report["best_epoch"], stop_reason=report["stop_reason"],
                                 elapsed_seconds=report["elapsed_seconds"]))
                print(name, variant, seed, round(report["mape"], 4), report["best_epoch"],
                      report["stop_reason"], flush=True)
                pd.DataFrame(rows).to_csv(args.output / "runs.csv", index=False)
    runs = pd.DataFrame(rows)
    summary = runs.groupby(["model", "variant"]).mape.agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps(dict(
        train="2010-2013", validation="2014", configs=CONFIGS, seeds=args.seeds,
        holiday_features="hol_today, hol_lag1, hol_lag7 as futr_exog (calendar only)",
        selection_rule="mean 2014 MAPE over seeds; differences within seed spread treated as ties"), indent=2))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
