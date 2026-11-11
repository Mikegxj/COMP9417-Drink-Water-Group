"""Stage 1 + Stage 2 under the protocol of the teammate's formal backtest (docs/demand/formal_experiment.md
in the group repo), so Ridge / MLP / LightGBM can be compared with GAM / N-BEATSx / TFT directly.

Fits (target years, inclusive):
  stage1  2010-2013  -> 2014 validation predictions (also used to choose ensemble weights)
  frozen  2010-2014  -> used for every day from 2015-01 to 2026-10
  2017    2013-2016, 2019 2015-2018, 2021 2017-2020, 2023 2019-2022, 2025 2021-2024
          -> periodic strategy: each refit replaces the model from 1 Jan of that year (5 retrains)
Hyperparameters and epoch counts stay at their Stage 1 values; scalers are refitted with the model.

Models:
  ridge_vicqld  per-half-hour Ridge + splines, Sydney weather + VIC/QLD demand (Stage 1 pick), alpha 0.1
  mlp           daily MLP, full Sydney features, 5 seeds averaged, 55 epochs (median Stage 1 epoch)
  mlp_nosolar   as mlp without Sydney cloud and solar inputs (tests the midday hypothesis), 46 epochs
  lgbm          reproduction of the teammate's LightGBM (cloud variant, holiday level 2)
Writes outputs/protocol/predictions.npz (MW, one (n_days, 48) array per model and fit).
"""
import os
import time

import numpy as np
from joblib import Parallel, delayed

import features as F
import models as M

OUT = os.path.join(F.OUT_ROOT, "protocol")
os.makedirs(OUT, exist_ok=True)

FITS = {"stage1": (2010, 2013), "frozen": (2010, 2014), "2017": (2013, 2016), "2019": (2015, 2018),
        "2021": (2017, 2020), "2023": (2019, 2022), "2025": (2021, 2024)}
SEEDS = range(5)
MLP_VARIANTS = {"mlp": ({}, 55), "mlp_nosolar": ({"no_solar_cloud": True}, 46)}

df = F.load()
days, arr = F.to_daily(df)
cal = F.calendar(days)
Y = arr["nsw_demand"]
idx = np.arange(7, len(days))
yr = days[idx].year.to_numpy()
L = F.level(arr)
Lr = L[idx][:, None]
R = Y[idx] / Lr
masks = {name: (yr >= a) & (yr <= b) for name, (a, b) in FITS.items()}
out = {"days": np.asarray(days[idx]).astype("datetime64[D]").astype(str), "truth": Y[idx]}

t0 = time.time()
frames = [F.ridge_frame(arr, cal, L, idx, p, {"regions": ["vic", "qld"]}) for p in range(F.P)]
for name, m in masks.items():
    out[f"ridge_vicqld|{name}"] = M.ridge_fit_predict(frames, R, m, 0.1) * Lr
print(f"ridge done {time.time() - t0:.0f}s", flush=True)

t0 = time.time()
lf = F.lgbm_frame(arr, cal, days, idx)
y_long = Y[idx].ravel()
for name, m in masks.items():
    out[f"lgbm|{name}"] = M.lgbm_fit_predict(lf, y_long, np.repeat(m, F.P)).reshape(-1, F.P)
print(f"lgbm done {time.time() - t0:.0f}s", flush=True)

for model, (opts, epochs) in MLP_VARIANTS.items():
    t0 = time.time()
    X = F.mlp_matrix(arr, cal, L, idx, opts)
    jobs = [(name, s) for name in masks for s in SEEDS]
    res = Parallel(n_jobs=10)(delayed(M.mlp_fit_predict)(X, R, masks[name], s, epochs) for name, s in jobs)
    for name in masks:
        runs = [r for (n_, s), r in zip(jobs, res) if n_ == name]
        out[f"{model}|{name}"] = np.mean(runs, axis=0) * Lr
        out[f"{model}_seed0|{name}"] = runs[0] * Lr
    print(f"{model} done {time.time() - t0:.0f}s", flush=True)

np.savez_compressed(os.path.join(OUT, "predictions.npz"), **out)
print("saved", os.path.join(OUT, "predictions.npz"))
