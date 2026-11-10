"""Stage 1 holiday-feature ablation for LightGBM and GAM (train 2010-2013, validate 2014).

exp0 current features; exp1 + hol_today; exp2 + hol_lag1, hol_lag7; exp3 + xmas_period.
Model configurations are fixed to the formal Stage 1 selections, so differences come from
features only. Selection rule (fixed before running): lowest 2014 overall MAPE; groups are
diagnostic only. Paired whole-day bootstrap gives 95% intervals for MAPE change vs exp0.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import tabular as demand_tabular
from harness import dayahead

from experiments.demand import ROOT  # noqa: E402
VARIANTS = {"gam": "prior_extrema", "lightgbm": "cloud"}
CONFIGS = {"gam": dict(n_knots=5, alpha=1),
           "lightgbm": dict(num_leaves=31, min_child_samples=24, learning_rate=0.04, n_estimators=500)}
EXPERIMENTS = {0: "exp0 current", 1: "exp1 +hol_today", 2: "exp2 +hol_lag1/lag7", 3: "exp3 +xmas_period"}


def day_groups(days):
    """Diagnostic groups per 2014 market day (each day in exactly one group)."""
    unique = pd.DatetimeIndex(sorted(set(days)))
    known = demand_tabular.nsw_holidays(range(2013, 2016))
    out = {}
    for d in unique:
        if d in known:
            out[d] = "holiday"
        elif (d - pd.Timedelta(days=1)) in known or (d - pd.Timedelta(days=7)) in known:
            out[d] = "after holiday (lag1/lag7)"
        elif (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 2):
            out[d] = "xmas period"
        else:
            out[d] = "normal"
    return pd.Series(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/demand/holiday_ablation")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--models", nargs="+", default=["lightgbm", "gam"], choices=list(VARIANTS))
    parser.add_argument("--levels", nargs="+", type=int, default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    args = parser.parse_args()
    if args.levels[0] != 0:
        parser.error("--levels must start with 0 (the comparison baseline)")
    args.output.mkdir(parents=True, exist_ok=True)

    df = dayahead.load_dataset(args.data)
    df = df.loc[dayahead.market_day(df.index).year <= 2014]
    days = dayahead.market_day(df.index)
    train, valid = days.year < 2014, days.year == 2014
    actual = df.nsw_demand[valid]
    vdays = days[valid]
    groups = day_groups(vdays)
    rng = np.random.default_rng(0)
    unique_days = groups.index.to_numpy()
    boot_idx = rng.integers(0, len(unique_days), size=(args.boot, len(unique_days)))

    rows, preds = [], {}
    for model in args.models:
        base_x = demand_tabular.features(df, VARIANTS[model])
        for level in args.levels:
            label = EXPERIMENTS[level]
            x = demand_tabular.add_holidays(base_x, level)
            pred = demand_tabular.forecast(model, x, df.nsw_demand, train, valid, CONFIGS[model])
            preds[(model, level)] = pred
            ape = pd.Series(np.abs(actual.to_numpy() - pred) / actual.to_numpy() * 100, index=actual.index)
            daily = ape.groupby(vdays).mean()                     # 48 points per day
            row = dict(model=model, experiment=label, overall=dayahead.mape(actual, pred))
            for name in ("holiday", "after holiday (lag1/lag7)", "xmas period", "normal"):
                row[name] = daily[groups == name].mean()
            if level == 0:
                base_daily = daily
            diff = (daily - base_daily).reindex(unique_days).to_numpy()
            boots = diff[boot_idx].mean(axis=1)
            row["change_vs_exp0"] = diff.mean()
            row["ci_low"], row["ci_high"] = np.percentile(boots, [2.5, 97.5])
            rows.append(row)
            print(model, label, round(row["overall"], 4), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "comparison.csv", index=False)
    pd.DataFrame({f"{m}_exp{l}": p for (m, l), p in preds.items()}, index=actual.index) \
        .assign(actual=actual).to_csv(args.output / "predictions_2014.csv.gz")
    counts = groups.value_counts().to_dict()
    (args.output / "protocol.json").write_text(json.dumps(dict(
        train="2010-2013", validation="2014", configs=CONFIGS, base_variants=VARIANTS,
        experiments={l: EXPERIMENTS[l] for l in args.levels}, models=args.models, group_days=counts, bootstrap=dict(reps=args.boot, unit="day", seed=0),
        holidays="holidays.Australia(subdiv='NSW'), Bank Holiday excluded",
        xmas_period="24 Dec - 2 Jan, non-public-holiday days",
        selection_rule="lowest 2014 overall MAPE; ties within CI -> simpler version"), indent=2))
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print("days per group:", counts)


if __name__ == "__main__":
    main()
