"""N-BEATSx (teammate's configuration, see nbeats.py) under the shared protocol of run_protocol.py:
stage1 2010-2013 -> 2014, frozen 2010-2014, periodic refits 2017/19/21/23/25 on the previous four
years. Seeds 42, 43, 44 (the teammate's seeds); the average and seed 42 alone are stored.
Writes outputs/protocol/predictions_nbeats.npz aligned with predictions.npz (target days from
2010-01-08) and prints a check against the teammate's own exp2 annual results.
"""
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import features as F
import nbeats as NB

OUT = os.path.join(F.OUT_ROOT, "protocol")
FITS = {"stage1": (2010, 2013), "frozen": (2010, 2014), "2017": (2013, 2016), "2019": (2015, 2018),
        "2021": (2017, 2020), "2023": (2019, 2022), "2025": (2021, 2024)}
SEEDS = (42, 43, 44)

df = F.load()
days, arr = F.to_daily(df)
y = df["nsw_demand"].to_numpy(float)
temp = df["syd_temp"].to_numpy(float)
cov = NB.raw_covariates(df.index)
year = days.year.to_numpy()
masks = {k: (year >= a) & (year <= b) & (np.arange(len(days)) >= NB.WINDOW // NB.P) for k, (a, b) in FITS.items()}

t0 = time.time()
jobs = [(k, s) for k in FITS for s in SEEDS]
res = Parallel(n_jobs=4, verbose=10)(delayed(NB.fit_predict)(y, temp, cov, masks[k], s, 4) for k, s in jobs)
print(f"{len(jobs)} fits done in {time.time() - t0:.0f}s", flush=True)

idx = np.arange(7, len(days))                       # align with predictions.npz
out = {}
for k in FITS:
    runs = [r for (k_, s), r in zip(jobs, res) if k_ == k]
    out[f"nbeats|{k}"] = np.mean(runs, axis=0)[idx]
    out[f"nbeats_seed42|{k}"] = runs[0][idx]
np.savez_compressed(os.path.join(OUT, "predictions_nbeats.npz"), **out)

truth = arr["nsw_demand"][idx]
yr = year[idx]


def strategy(key, kind):
    pred = out[f"{key}|frozen"].copy()
    if kind == "periodic":
        for y0 in (2017, 2019, 2021, 2023, 2025):
            pred[yr >= y0] = out[f"{key}|{y0}"][yr >= y0]
    return pred


def mape(pred, m):
    return float(np.mean(np.abs(pred[m] - truth[m]) / truth[m]) * 100)


tm = pd.read_csv(os.path.join(OUT, "teammate_exp2_annual.csv"))
rows = []
for kind in ("frozen", "periodic"):
    t = tm[(tm.model == "nbeats") & (tm.strategy == kind)].set_index("year").mape
    for y_ in t.index:
        rows.append({"strategy": kind, "year": y_, "teammate seed 42": t[y_],
                     "ours seed 42": mape(strategy("nbeats_seed42", kind), yr == y_),
                     "ours 3-seed mean": mape(strategy("nbeats", kind), yr == y_)})
chk = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(chk.round(3).to_string(index=False))
sim = yr >= 2015
print("\n2014 validation (fixed 36 epochs): seed 42 %.3f, 3-seed mean %.3f" %
      (mape(out["nbeats_seed42|stage1"], yr == 2014), mape(out["nbeats|stage1"], yr == 2014)))
for key in ("nbeats_seed42", "nbeats"):
    print(f"{key}: frozen 2015-26 {mape(strategy(key, 'frozen'), sim):.3f}, periodic 2015-26 "
          f"{mape(strategy(key, 'periodic'), sim):.3f}, periodic 2025-26 {mape(strategy(key, 'periodic'), yr >= 2025):.3f}")
chk.to_csv(os.path.join(OUT, "nbeats_reproduction_check.csv"), index=False)
