"""N-BEATSx with Sydney cloud cover and solar radiation added to its known covariates.

Question: is the gap between N-BEATSx and the MLP only because the teammate's N-BEATSx
configuration has no cloud/solar inputs (it uses Sydney temperature, calendar sin/cos and
holiday flags), while the MLP uses Sydney temperature, humidity, cloud and solar?

Same as run_nbeats.py (teammate's configuration, seeds 42/43/44, shared protocol: stage1
2010-2013 -> 2014, frozen 2010-2014, periodic refits 2017/19/21/23/25 on the previous four
years) except that syd_cloud and syd_solar (standardised on the training rows, over the
480 past and the 48 target half-hours, like temperature) are added as known covariates.

Outputs (OUT_ROOT/protocol/): predictions_nbeats_wx.npz, nbeats_wx_summary.csv,
nbeats_wx_by_year.csv, nbeats_wx_paired.csv.
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
EXTRA = ["syd_cloud", "syd_solar"]

df = F.load()
days, arr = F.to_daily(df)
y = df["nsw_demand"].to_numpy(float)
temp = df["syd_temp"].to_numpy(float)
extra = df[EXTRA].to_numpy(float)
cov = NB.raw_covariates(df.index)
year = days.year.to_numpy()
masks = {k: (year >= a) & (year <= b) & (np.arange(len(days)) >= NB.WINDOW // NB.P) for k, (a, b) in FITS.items()}

t0 = time.time()
jobs = [(k, s) for k in FITS for s in SEEDS]
res = Parallel(n_jobs=4, verbose=10)(delayed(NB.fit_predict)(y, temp, cov, masks[k], s, 4, extra) for k, s in jobs)
print(f"{len(jobs)} fits done in {time.time() - t0:.0f}s", flush=True)

idx = np.arange(7, len(days))                       # align with predictions.npz
out = {}
for k in FITS:
    runs = [r for (k_, s), r in zip(jobs, res) if k_ == k]
    out[f"nbeats_wx|{k}"] = np.mean(runs, axis=0)[idx]
    out[f"nbeats_wx_seed42|{k}"] = runs[0][idx]
np.savez_compressed(os.path.join(OUT, "predictions_nbeats_wx.npz"), **out)

# --- comparison -------------------------------------------------------------------------------
z = dict(np.load(os.path.join(OUT, "predictions.npz")))
z.update(np.load(os.path.join(OUT, "predictions_nbeats.npz")))
z.update(out)
truth = z["truth"]
yr = pd.DatetimeIndex(z["days"]).year.to_numpy()
VAL, SIM, P1, P2 = yr == 2014, yr >= 2015, (yr >= 2015) & (yr <= 2024), yr >= 2025
MID = slice(22, 28)                                  # half-hours ending 11:30-14:00


def strategy(key, kind):
    if kind == "stage1":
        return z[f"{key}|stage1"]
    pred = z[f"{key}|frozen"].copy()
    if kind == "periodic":
        for y0 in (2017, 2019, 2021, 2023, 2025):
            pred[yr >= y0] = z[f"{key}|{y0}"][yr >= y0]
    return pred


def mape(pred, m, sl=slice(None)):
    return float(np.mean(np.abs(pred[m][:, sl] - truth[m][:, sl]) / truth[m][:, sl]) * 100)


def mae(pred, m):
    return float(np.mean(np.abs(pred[m] - truth[m])))


MODELS = {"MLP (5 seeds)": "mlp", "MLP without solar & cloud": "mlp_nosolar",
          "N-BEATSx (3 seeds)": "nbeats", "N-BEATSx + cloud & solar (3 seeds)": "nbeats_wx",
          "N-BEATSx + cloud & solar (seed 42)": "nbeats_wx_seed42"}
rows, years = [], []
for name, key in MODELS.items():
    fr, pe = strategy(key, "frozen"), strategy(key, "periodic")
    rows.append({"model": name, "val 2014": mape(strategy(key, "stage1"), VAL),
                 "frozen 2015-24": mape(fr, P1), "frozen 2025-26": mape(fr, P2), "frozen all": mape(fr, SIM),
                 "periodic 2015-24": mape(pe, P1), "periodic 2025-26": mape(pe, P2), "periodic all": mape(pe, SIM),
                 "periodic all MAE (MW)": mae(pe, SIM), "periodic 2025-26 MAE (MW)": mae(pe, P2),
                 "periodic 2025-26 midday": mape(pe, P2, MID)})
    years.append(pd.Series({y_: mape(pe, yr == y_) for y_ in range(2015, 2027)}, name=name))
summ = pd.DataFrame(rows)
by_year = pd.DataFrame(years).T
by_year.index.name = "year (periodic)"


def day_ape(pred):
    return np.mean(np.abs(pred - truth) / truth, axis=1) * 100


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)], [2.5, 97.5])


prow = []
for a, b in (("nbeats_wx", "nbeats"), ("nbeats_wx", "mlp"), ("nbeats_wx", "mlp_nosolar")):
    for kind in ("stage1", "periodic"):
        d = day_ape(strategy(a, kind)) - day_ape(strategy(b, kind))
        for per, m in ((("2014", VAL),) if kind == "stage1" else (("2015-26", SIM), ("2025-26", P2))):
            lo, hi = boot(d[m])
            prow.append({"A": a, "B": b, "strategy": kind, "period": per, "MAPE A - B (pp)": d[m].mean(),
                         "95% low": lo, "95% high": hi})
paired = pd.DataFrame(prow)

summ.to_csv(os.path.join(OUT, "nbeats_wx_summary.csv"), index=False)
by_year.to_csv(os.path.join(OUT, "nbeats_wx_by_year.csv"))
paired.to_csv(os.path.join(OUT, "nbeats_wx_paired.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(summ.round(3).to_string(index=False))
print()
print(by_year.round(2).to_string())
print()
print(paired.round(3).to_string(index=False))
