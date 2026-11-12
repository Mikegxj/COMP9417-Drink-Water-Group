"""Stage 2 check for the Sydney solar-radiation input: frozen and periodic backtests, 2015-01 to 2026-10.

Compares N-BEATSx exp2 (baseline) with exp2 + syd_solar, both re-run here with seeds 42/43/44 so the
comparison is paired and free of cross-environment differences. Each run reuses
formal_backtest.run_model: frozen = one 2010-2014 fit; periodic = refits at the start of 2017/19/21/23/25
on the previous four years. Refit epochs: exp2 keeps its locked 36; solar uses 53, the median best epoch
of its three Stage 1 seeds (41/54/53), the same rule that produced 36 for exp2.

    python -m experiments.demand.solar_cloud_ablation.solar_backtest --variant solar --seed 42
    python -m experiments.demand.solar_cloud_ablation.solar_backtest --summarize
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.demand import ROOT
from experiments.demand.backtest.formal_backtest import run_model
from experiments.demand.solar_cloud_ablation.solar_cloud_ablation import tag
from harness import dayahead

OUTPUT = ROOT / "results/demand/solar_backtest"
BASE = dict(window=480, width=128, weather=True, lr=0.001, holidays=True)
VARIANTS = {"exp2": dict(config=BASE, refit_epochs=36),
            "solar": dict(config=dict(BASE, solar=True), refit_epochs=53)}
# Section 18: MLP-inspired changes on top of solar that passed the 2014 rule (level scaling failed it).
# Clean protocol only; refit epochs = median inner best epoch from clean_stage1.
SECTION18 = {"solar_mape": dict(loss="mape"), "solar_rh": dict(rh=True), "solar_xmas": dict(xmas=True),
             "solar_tempstats": dict(tempstats=True)}
VARIANTS.update({k: dict(config=dict(BASE, solar=True, **v), refit_epochs=None) for k, v in SECTION18.items()})
SEEDS = [42, 43, 44]
# Section 16: 8-network float32 ensemble of the solar variant (seeds 42-49).
ENSEMBLE_SEEDS = list(range(42, 50))
# Section 17: epochs from the MLP-aligned Stage 1 (2010-2012 train, early stop on 2013; inner median best
# epoch), 2014 not used for any choice. Selected with --protocol clean; results go to a separate folder.
PROTOCOLS = {"2014es": dict(output=OUTPUT, epochs=dict(exp2=36, solar=53)),
             "clean": dict(output=ROOT / "results/demand/solar_backtest_clean",
                           epochs=dict(exp2=60, solar=39, solar_mape=23, solar_rh=47, solar_xmas=48,
                                       solar_tempstats=44))}
PERIODS = {"2015–2024": lambda y: y < 2025, "2025–2026.10": lambda y: y >= 2025, "overall": lambda y: y >= 2015}


def run(variant, seed):
    if VARIANTS[variant]["refit_epochs"] is None:
        raise ValueError(f"{variant} has no refit epochs under this protocol")
    root = OUTPUT / variant / f"seed_{seed}"
    (root / "nbeats").mkdir(parents=True, exist_ok=True)
    selected = dict(model="nbeats", config=VARIANTS[variant]["config"],
                    refit_epochs=VARIANTS[variant]["refit_epochs"], refit_seed=seed, device="cpu")
    (root / "nbeats/selected.json").write_text(json.dumps(selected, indent=2))
    df = dayahead.load_dataset(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    run_model("nbeats", df, root)


def load(variant, seed, strategy):
    path = OUTPUT / variant / f"seed_{seed}/backtest/nbeats/{strategy}_predictions.csv.gz"
    return tag(pd.read_csv(path, index_col="timestamp", parse_dates=True))


def daily_bootstrap(base, other, mask, draws=2000, seed=0):
    diff = (other.ape - base.ape)[mask]
    daily = diff.groupby(base.day[mask]).agg(["sum", "count"])
    pick = np.random.default_rng(seed).integers(0, len(daily), size=(draws, len(daily)))
    stats = daily["sum"].to_numpy()[pick].sum(1) / daily["count"].to_numpy()[pick].sum(1)
    return diff.mean(), *np.percentile(stats, [2.5, 97.5])


def summarize():
    rows, annual, tests = [], [], []
    for strategy in ("frozen", "periodic"):
        means = {}
        for variant in ("exp2", "solar"):
            frames = [load(variant, seed, strategy) for seed in SEEDS]
            if not all(f.index.equals(frames[0].index) for f in frames):
                raise ValueError("Seed predictions are not aligned")
            mean = tag(frames[0][["actual"]].assign(
                forecast=pd.concat([f.forecast for f in frames], axis=1).mean(axis=1)))
            means[variant] = mean
            for seed, frame in [*zip(SEEDS, frames), ("mean_forecast", mean)]:
                year = frame.day.dt.year
                for label, rule in PERIODS.items():
                    sub = frame[rule(year)]
                    rows.append(dict(strategy=strategy, variant=variant, seed=seed, subset=label,
                                     mape=sub.ape.mean(), mae=(sub.forecast - sub.actual).abs().mean()))
                    sub = sub[sub.period == "midday"]
                    rows.append(dict(strategy=strategy, variant=variant, seed=seed, subset=f"{label} midday",
                                     mape=sub.ape.mean(), mae=(sub.forecast - sub.actual).abs().mean()))
                if seed != "mean_forecast":
                    annual += [dict(strategy=strategy, variant=variant, seed=seed, year=y, mape=g.ape.mean())
                               for y, g in frame.groupby(year)]
        base, other = means["exp2"], means["solar"]
        year = base.day.dt.year
        for label, rule in PERIODS.items():
            for part, mask in ((label, rule(year)), (f"{label} midday", rule(year) & (base.period == "midday"))):
                delta, low, high = daily_bootstrap(base, other, mask)
                tests.append(dict(strategy=strategy, subset=part, delta_mape=delta, ci_low=low, ci_high=high))
    rows, annual, tests = pd.DataFrame(rows), pd.DataFrame(annual), pd.DataFrame(tests)
    rows.to_csv(OUTPUT / "comparison.csv", index=False)
    annual.to_csv(OUTPUT / "annual.csv", index=False)
    tests.to_csv(OUTPUT / "bootstrap_solar_vs_exp2.csv", index=False)
    (OUTPUT / "protocol.json").write_text(json.dumps(dict(
        variants={k: VARIANTS[k] for k in ("exp2", "solar")}, seeds=SEEDS, frozen="2010-2014 fit, predict 2015-01..2026-10",
        periodic="refit at start of 2017/2019/2021/2023/2025 on previous four years",
        midday="interval start 10:00-14:30 AEST",
        bootstrap="paired whole-day resampling of 3-seed mean forecasts, 2000 draws, 95% percentile"), indent=2))
    per_seed = rows[rows.seed != "mean_forecast"]
    table = per_seed.groupby(["strategy", "subset", "variant"]).mape.agg(["mean", "std"]).unstack("variant")
    print(table.round(3).to_string())
    print(per_seed.pivot_table(index=["strategy", "subset"], columns=["variant", "seed"], values="mape").round(3).to_string())
    print(tests.round(3).to_string(index=False))


def summarize_ensemble():
    """8-network mean forecast of the solar variant vs single networks and the 3-network mean."""
    rows, tests = [], []
    for strategy in ("frozen", "periodic"):
        solar = [load("solar", seed, strategy) for seed in ENSEMBLE_SEEDS]
        exp2 = [load("exp2", seed, strategy) for seed in SEEDS]
        if not all(f.index.equals(solar[0].index) for f in solar + exp2):
            raise ValueError("Seed predictions are not aligned")
        def mean_of(frames):
            return tag(frames[0][["actual"]].assign(
                forecast=pd.concat([f.forecast for f in frames], axis=1).mean(axis=1)))
        candidates = {"exp2 single (mean of 3 seeds)": exp2, "solar single (mean of 8 seeds)": solar,
                      "solar 3-net mean (42-44)": [mean_of(solar[:3])], "solar 8-net mean": [mean_of(solar)],
                      "exp2 seed 42 (current submission setup)": exp2[:1]}
        rh = None
        if all((OUTPUT / f"solar_rh/seed_{s}/backtest/nbeats/{strategy}_predictions.csv.gz").exists()
               for s in ENSEMBLE_SEEDS):
            rh = [load("solar_rh", seed, strategy) for seed in ENSEMBLE_SEEDS]
            if not all(f.index.equals(solar[0].index) for f in rh):
                raise ValueError("Seed predictions are not aligned")
            candidates.update({"solar+rh single (mean of 8 seeds)": rh, "solar+rh 8-net mean": [mean_of(rh)]})
        year = solar[0].day.dt.year
        for name, frames in candidates.items():
            for label, rule in PERIODS.items():
                for part, extra in ((label, True), (f"{label} midday", solar[0].period == "midday")):
                    mask = rule(year) & extra
                    rows.append(dict(strategy=strategy, candidate=name, subset=part,
                                     mape=np.mean([f.ape[mask].mean() for f in frames])))
        ensemble, single42 = mean_of(solar), exp2[0]
        pairs = [("solar 8-net mean", "exp2 seed 42", ensemble, single42),
                 ("solar 8-net mean", "solar 3-net mean", ensemble, mean_of(solar[:3]))]
        if rh is not None:
            pairs.append(("solar+rh 8-net mean", "solar 8-net mean", mean_of(rh), ensemble))
        for name, reference_name, other, reference in pairs:
            for label, rule in PERIODS.items():
                for part, extra in ((label, True), (f"{label} midday", solar[0].period == "midday")):
                    delta, low, high = daily_bootstrap(reference, other, rule(year) & extra)
                    tests.append(dict(strategy=strategy, candidate=name, reference=reference_name, subset=part,
                                      delta_mape=delta, ci_low=low, ci_high=high))
    rows, tests = pd.DataFrame(rows), pd.DataFrame(tests)
    rows.to_csv(OUTPUT / "ensemble8_comparison.csv", index=False)
    tests.to_csv(OUTPUT / "ensemble8_bootstrap.csv", index=False)
    print(rows.pivot_table(index=["strategy", "subset"], columns="candidate", values="mape", sort=False).round(3).to_string())
    print(tests.round(3).to_string(index=False))


def summarize_section18():
    """Each section-18 change (3-net mean, seeds 42-44) vs solar 3-net mean, paired whole-day bootstrap."""
    rows, tests = [], []
    for strategy in ("frozen", "periodic"):
        def mean_of(variant):
            frames = [load(variant, seed, strategy) for seed in SEEDS]
            return frames, tag(frames[0][["actual"]].assign(
                forecast=pd.concat([f.forecast for f in frames], axis=1).mean(axis=1)))
        solar_frames, solar = mean_of("solar")
        year = solar.day.dt.year
        for variant in ["solar", *SECTION18]:
            frames, mean = (solar_frames, solar) if variant == "solar" else mean_of(variant)
            if not mean.index.equals(solar.index):
                raise ValueError("Predictions are not aligned")
            for label, rule in PERIODS.items():
                for part, extra in ((label, True), (f"{label} midday", solar.period == "midday")):
                    mask = rule(year) & extra
                    rows.append(dict(strategy=strategy, variant=variant, subset=part,
                                     single=np.mean([f.ape[mask].mean() for f in frames]), net3=mean.ape[mask].mean()))
                    if variant != "solar":
                        delta, low, high = daily_bootstrap(solar, mean, mask)
                        tests.append(dict(strategy=strategy, comparison=f"{variant} - solar (3-net)", subset=part,
                                          delta_mape=delta, ci_low=low, ci_high=high))
    rows, tests = pd.DataFrame(rows), pd.DataFrame(tests)
    rows.to_csv(OUTPUT / "section18_comparison.csv", index=False)
    tests.to_csv(OUTPUT / "section18_bootstrap.csv", index=False)
    print(rows.pivot_table(index=["strategy", "subset"], columns="variant", values="net3", sort=False).round(3).to_string())
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, choices=ENSEMBLE_SEEDS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--section18", action="store_true", help="summarize the section 18 changes")
    parser.add_argument("--protocol", choices=list(PROTOCOLS), default="2014es")
    args = parser.parse_args()
    OUTPUT = PROTOCOLS[args.protocol]["output"]
    for name, epochs in PROTOCOLS[args.protocol]["epochs"].items():
        VARIANTS[name]["refit_epochs"] = epochs
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.section18:
        summarize_section18()
    elif args.summarize:
        summarize()
        summarize_ensemble()
    else:
        run(args.variant, args.seed)
