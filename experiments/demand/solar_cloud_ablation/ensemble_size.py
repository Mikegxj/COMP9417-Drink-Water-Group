"""How many N-BEATSx (exp2 + solar) networks are worth averaging? 2014 validation, 30 seeds (42-71).

Seeds 42-44 come from solar_cloud_ablation; 45-71 were trained with the same script and config:
    python -m experiments.demand.solar_cloud_ablation.solar_cloud_ablation --variants solar --seeds S \
        --output results/demand/solar_ensemble_size/seed_run_S
For each ensemble size K, MAPE is averaged over up to 200 random K-seed subsets (all subsets when fewer),
for both the mean and the median of the member forecasts (the paper [3] uses the median of 30 networks).

    python -m experiments.demand.solar_cloud_ablation.ensemble_size
"""
import itertools
import math

import numpy as np
import pandas as pd

from experiments.demand import ROOT
from experiments.demand.solar_cloud_ablation.solar_cloud_ablation import tag

OUTPUT = ROOT / "results/demand/solar_ensemble_size"
SEEDS = list(range(42, 72))


def path(seed):
    if seed <= 44:
        return ROOT / f"results/demand/solar_cloud_ablation/solar/seed_{seed}/predictions.csv.gz"
    return OUTPUT / f"seed_run_{seed}/solar/seed_{seed}/predictions.csv.gz"


def main():
    frames = [pd.read_csv(path(s), index_col=0, parse_dates=True) for s in SEEDS]
    if not all(f.index.equals(frames[0].index) and np.allclose(f.actual, frames[0].actual) for f in frames):
        raise ValueError("Seed predictions are not aligned")
    info = tag(frames[0][["actual"]].assign(forecast=frames[0].forecast))
    actual = info.actual.to_numpy()
    midday = (info.period == "midday").to_numpy()
    preds = np.stack([f.forecast.to_numpy() for f in frames])
    rng = np.random.default_rng(0)
    rows = []
    for k in range(1, len(SEEDS) + 1):
        if math.comb(len(SEEDS), k) <= 200:
            combos = list(itertools.combinations(range(len(SEEDS)), k))
        else:  # sample 200 random subsets (with possible repeats; negligible at these counts)
            combos = [tuple(sorted(rng.choice(len(SEEDS), k, replace=False))) for _ in range(200)]
        for how, agg in (("mean", np.mean), ("median", np.median)):
            ape = np.stack([100 * np.abs(agg(preds[list(c)], axis=0) - actual) / actual for c in combos])
            full, noon = ape.mean(1), ape[:, midday].mean(1)
            rows.append(dict(k=k, aggregate=how, subsets=len(combos), mape=full.mean(), mape_sd=full.std(),
                             mape_min=full.min(), mape_max=full.max(), midday_mape=noon.mean()))
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "ensemble_size.csv", index=False)
    singles = pd.DataFrame(dict(seed=SEEDS, mape=[100 * np.mean(np.abs(p - actual) / actual) for p in preds]))
    singles.to_csv(OUTPUT / "single_seeds.csv", index=False)
    print(singles.round(3).to_string(index=False))
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
