"""2x2 cross experiment: model family x information set, under the shared protocol of run_protocol.py.

                     lean set (teammate's 11 features)        rich set (Ridge/MLP features)
  LightGBM           lgbm      (raw MW target)                lgbm_rich (ratio target)    <- new
  MLP                mlp_lean  (lags scaled by L, ratio)  new  mlp       (ratio target)

Same model settings as before: LightGBM keeps the teammate's configuration (500 trees, lr 0.04,
31 leaves, min_child_samples 24); MLP keeps its architecture, 5 seeds, epochs = median of the
Stage 1 early-stopping epochs for that feature set (83 for the lean set). Fits as in
run_protocol.py (stage1 / frozen / 5 periodic 4-year refits). Writes predictions_cross.npz and
cross_summary.csv to outputs/protocol/.
"""
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import features as F
import models as M

OUT = os.path.join(F.OUT_ROOT, "protocol")
FITS = {"stage1": (2010, 2013), "frozen": (2010, 2014), "2017": (2013, 2016), "2019": (2015, 2018),
        "2021": (2017, 2020), "2023": (2019, 2022), "2025": (2021, 2024)}
SEEDS, LEAN_EPOCHS = range(5), 83
pd.set_option("display.width", 220)

df = F.load()
days, arr = F.to_daily(df)
cal = F.calendar(days)
Y = arr["nsw_demand"]
idx = np.arange(7, len(days))
yr = days[idx].year.to_numpy()
L = F.level(arr)
Lr = L[idx][:, None]
R = Y[idx] / Lr
truth = Y[idx]
masks = {name: (yr >= a) & (yr <= b) for name, (a, b) in FITS.items()}
out = {}

t0 = time.time()
lf = F.lgbm_rich_frame(arr, cal, L, idx)
print("lgbm_rich columns:", list(lf.columns))
for name, m in masks.items():
    out[f"lgbm_rich|{name}"] = M.lgbm_fit_predict(lf, R.ravel(), np.repeat(m, F.P)).reshape(-1, F.P) * Lr
print(f"lgbm_rich done {time.time() - t0:.0f}s", flush=True)

t0 = time.time()
X = F.mlp_matrix(arr, cal, L, idx, {"minimal": True})
jobs = [(name, s) for name in masks for s in SEEDS]
res = Parallel(n_jobs=10)(delayed(M.mlp_fit_predict)(X, R, masks[name], s, LEAN_EPOCHS) for name, s in jobs)
for name in masks:
    out[f"mlp_lean|{name}"] = np.mean([r for (n_, s), r in zip(jobs, res) if n_ == name], axis=0) * Lr
print(f"mlp_lean done {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(os.path.join(OUT, "predictions_cross.npz"), **out)

# --- 2x2 summary -------------------------------------------------------------------------------
base = np.load(os.path.join(OUT, "predictions.npz"))
src = {**{k: base[k] for k in base.files}, **out}
SIM, P2 = yr >= 2015, yr >= 2025
mid = slice(23, 27)


def strategy(model, kind):
    if kind == "stage1":
        return src[f"{model}|stage1"]
    pred = src[f"{model}|frozen"].copy()
    if kind == "periodic":
        for y0 in (2017, 2019, 2021, 2023, 2025):
            pred[yr >= y0] = src[f"{model}|{y0}"][yr >= y0]
    return pred


def mape(pred, mask, sl=slice(None)):
    return float(np.mean(np.abs(pred[mask][:, sl] - truth[mask][:, sl]) / truth[mask][:, sl]) * 100)


rows = []
for model, feats, key in (("LightGBM", "lean (teammate)", "lgbm"), ("LightGBM", "rich (mine)", "lgbm_rich"),
                          ("MLP", "lean (teammate)", "mlp_lean"), ("MLP", "rich (mine)", "mlp")):
    fr, pe = strategy(key, "frozen"), strategy(key, "periodic")
    rows.append({"model": model, "features": feats, "val 2014": mape(strategy(key, "stage1"), yr == 2014),
                 "frozen 2015-26": mape(fr, SIM), "frozen 2025-26": mape(fr, P2),
                 "periodic 2015-26": mape(pe, SIM), "periodic 2025-26": mape(pe, P2),
                 "periodic 2025-26 midday": mape(pe, P2, mid)})
tab = pd.DataFrame(rows).set_index(["model", "features"])
print("\n=== 2x2: model family x information set (MAPE %) ===")
print(tab.round(3))
tab.to_csv(os.path.join(OUT, "cross_summary.csv"))
