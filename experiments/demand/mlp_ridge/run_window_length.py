"""Length of the periodic-retraining window for the MLP (teammate's suggestion, experiment log
section 20 for N-BEATSx), with the same design and the same pre-registered rule.

Periodic refits stay on 1 Jan 2017/2019/2021/2023/2025; each refit uses the previous L calendar
years, L = 2, 3, 4 (current), 6, or all history from 2010. 2015-2016 use the frozen 2010-2014
model for every L. Epochs keep the number of gradient updates of 4 years x 55 epochs:
epochs = round(55 * 4 / L) (2 y: 110, 3 y: 73, 6 y: 37, all history: 31/24/20/17/15 for
2017..2025). MLP as in run_protocol.py: 5 seeds (0-4) averaged, same features and settings.
L = 4 is re-run here so every comparison is paired within the same run.

Rule (fixed before looking, as in the teammate's section 20): compare each L with 4 years on
periodic 2017-2024 with a paired day-block bootstrap (2000 draws); switch only if the 95 %
interval lies entirely below 0. 2025-26 has been looked at many times and is reported only.

Outputs (OUT_ROOT/window_length/): predictions_window.npz, comparison.csv, bootstrap_vs_4years.csv,
by_year_vs_4years.csv.
"""
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import features as F
import models as M

OUT = os.path.join(F.OUT_ROOT, "window_length")
os.makedirs(OUT, exist_ok=True)
REFITS = (2017, 2019, 2021, 2023, 2025)
WINDOWS = {"2 years": 2, "3 years": 3, "4 years": 4, "6 years": 6, "all history": None}
SEEDS = range(5)

df = F.load()
days, arr = F.to_daily(df)
cal = F.calendar(days)
Y = arr["nsw_demand"]
idx = np.arange(7, len(days))
yr = days[idx].year.to_numpy()
L = F.level(arr)
Lr = L[idx][:, None]
R = Y[idx] / Lr
X = F.mlp_matrix(arr, cal, L, idx)
truth = Y[idx]

jobs = []
for wname, w in WINDOWS.items():
    for y0 in REFITS:
        first = 2010 if w is None else y0 - w
        n_years = y0 - first
        epochs = int(round(55 * 4 / n_years))
        mask = (yr >= first) & (yr <= y0 - 1)
        jobs += [(wname, y0, epochs, mask, s) for s in SEEDS]
print("epochs:", sorted({(w, y0, e) for w, y0, e, _, _ in jobs}), flush=True)

t0 = time.time()
res = Parallel(n_jobs=8, verbose=10)(delayed(M.mlp_fit_predict)(X, R, m, s, e) for _, _, e, m, s in jobs)
print(f"{len(jobs)} fits done in {time.time() - t0:.0f}s", flush=True)

z = np.load(os.path.join(F.OUT_ROOT, "protocol", "predictions.npz"))
assert np.allclose(z["truth"], truth)
out = {}
pred = {}
for wname in WINDOWS:
    p = z["mlp|frozen"].copy()                      # 2015-2016: frozen 2010-2014 model for every L
    for y0 in REFITS:
        runs = [r for (w, y, _, _, _), r in zip(jobs, res) if w == wname and y == y0]
        fit = np.mean(runs, axis=0) * Lr
        out[f"{wname}|{y0}"] = fit
        p[yr >= y0] = fit[yr >= y0]
    pred[wname] = p
np.savez_compressed(os.path.join(OUT, "predictions_window.npz"), **out)

# reproducibility of the 4-year re-run against run_protocol.py
stored = z["mlp|frozen"].copy()
for y0 in REFITS:
    stored[yr >= y0] = z[f"mlp|{y0}"][yr >= y0]

MID = slice(22, 28)                                   # half-hours ending 11:30-14:00
SEL = (yr >= 2017) & (yr <= 2024)                     # decision period
REC = yr >= 2025                                      # recent, report only
FULL = yr >= 2015


def mape(p, m, sl=slice(None)):
    return float(np.mean(np.abs(p[m][:, sl] - truth[m][:, sl]) / truth[m][:, sl]) * 100)


def day_ape(p):
    return np.mean(np.abs(p - truth) / truth, axis=1) * 100


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)], [2.5, 97.5])


rows, brows, yrows = [], [], []
base = pred["4 years"]
for wname, p in pred.items():
    rows.append({"window": wname, "2017-24": mape(p, SEL), "2017-24 midday": mape(p, SEL, MID),
                 "2025-26": mape(p, REC), "2025-26 midday": mape(p, REC, MID), "2015-26": mape(p, FULL)})
    if wname == "4 years":
        continue
    d = day_ape(p) - day_ape(base)
    for per, m in (("2017-24 (decision)", SEL), ("2025-26 (report only)", REC), ("2015-26", FULL)):
        lo, hi = boot(d[m])
        brows.append({"window": wname, "period": per, "diff vs 4 years (pp)": d[m].mean(), "95% low": lo,
                      "95% high": hi, "switch by rule": bool(hi < 0) if per.startswith("2017") else ""})
    yrows.append(pd.Series({y_: mape(p, yr == y_) - mape(base, yr == y_) for y_ in range(2017, 2027)}, name=wname))
comp, bt, byy = pd.DataFrame(rows), pd.DataFrame(brows), pd.DataFrame(yrows).T
byy.index.name = "year"
comp.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
bt.to_csv(os.path.join(OUT, "bootstrap_vs_4years.csv"), index=False)
byy.to_csv(os.path.join(OUT, "by_year_vs_4years.csv"))

pd.set_option("display.width", 220)
print(f"\n4-year re-run vs run_protocol.py: 2015-26 MAPE {mape(base, FULL):.3f} vs {mape(stored, FULL):.3f}, "
      f"mean |diff| {np.mean(np.abs(base - stored) / stored) * 100:.3f}%")
print(comp.round(3).to_string(index=False))
print()
print(bt.round(3).to_string(index=False))
print("\nper year, MAPE minus 4 years (pp):")
print(byy.round(2).to_string())
