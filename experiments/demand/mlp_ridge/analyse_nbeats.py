"""Our N-BEATSx run (run_nbeats.py, teammate's configuration, seeds 42/43/44) in the same tables
as analyse_protocol.py: summary row, paired day-block bootstrap difference vs the MLP, and the
weekly improvement I over the yesterday baseline in 2025-26 (periodic retraining).
Outputs: outputs/protocol/summary_nbeats.csv, paired_differences_nbeats.csv, weekly_improvement_nbeats.csv.
"""
import os

import numpy as np
import pandas as pd

import features as F

OUT = os.path.join(F.OUT_ROOT, "protocol")
z = np.load(os.path.join(OUT, "predictions.npz"))
nb = np.load(os.path.join(OUT, "predictions_nbeats.npz"))
days = pd.DatetimeIndex(z["days"])
truth = z["truth"]
yr = days.year.to_numpy()
VAL, SIM = yr == 2014, yr >= 2015
P1, P2 = SIM & (yr <= 2024), yr >= 2025
RETRAINS = [2017, 2019, 2021, 2023, 2025]
src = {"mlp": z, "ridge_vicqld": z, "lgbm": z, "nbeats": nb, "nbeats_seed42": nb}
yday = np.vstack([np.full((1, F.P), np.nan), truth[:-1]])


def strategy(model, kind):
    s = src[model]
    if kind == "stage1":
        return s[f"{model}|stage1"]
    pred = s[f"{model}|frozen"].copy()
    if kind == "periodic":
        for y0 in RETRAINS:
            pred[yr >= y0] = s[f"{model}|{y0}"][yr >= y0]
    return pred


def mape(pred, mask):
    return float(np.mean(np.abs(pred[mask] - truth[mask]) / truth[mask]) * 100)


def day_ape(pred):
    return np.mean(np.abs(pred - truth) / truth, axis=1) * 100


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)], [2.5, 97.5])


rows = []
for m, label in (("nbeats", "nbeats (ours, 3-seed mean)"), ("nbeats_seed42", "nbeats (ours, seed 42)")):
    r = {"model": label, "val 2014": mape(strategy(m, "stage1"), VAL)}
    for kind in ("frozen", "periodic"):
        p = strategy(m, kind)
        r[f"{kind} 2015-24"], r[f"{kind} 2025-26"], r[f"{kind} all"] = mape(p, P1), mape(p, P2), mape(p, SIM)
    rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "summary_nbeats.csv"), index=False)

prow = []
mlp_d = day_ape(strategy("mlp", "periodic"))
for b in ("nbeats", "ridge_vicqld"):
    d = day_ape(strategy(b, "periodic")) - mlp_d
    for name, m in (("2015-26", SIM), ("2025-26", P2)):
        lo, hi = boot(d[m])
        prow.append({"A": b, "B": "mlp", "period": name, "MAPE A - B (pp)": d[m].mean(),
                     "95% low": lo, "95% high": hi})
ptab = pd.DataFrame(prow)
ptab.to_csv(os.path.join(OUT, "paired_differences_nbeats.csv"), index=False)

dow = days.dayofweek.to_numpy()
week = np.asarray(days - pd.to_timedelta(dow, unit="D"))
wk = []
for w in sorted(set(week[P2])):
    m = P2 & (week == w)
    if m.sum() < 7:
        continue
    b = mape(yday, m)
    wk.append({"week": pd.Timestamp(w).date(), "baseline": b,
               **{k: 100 * (1 - mape(strategy(k, "periodic"), m) / b) for k in ("mlp", "nbeats", "lgbm", "ridge_vicqld")}})
wk = pd.DataFrame(wk).set_index("week")
wk.to_csv(os.path.join(OUT, "weekly_improvement_nbeats.csv"))

pd.set_option("display.width", 200)
print(summ.round(3).to_string(index=False))
print()
print(ptab.round(3).to_string(index=False))
print(f"\nWeekly I vs yesterday, {len(wk)} complete Mon-Sun weeks of 2025-26 (periodic):")
for k in ("mlp", "nbeats", "lgbm", "ridge_vicqld"):
    print(f"  {k:<13} median {wk[k].median():5.1f}  10th pct {wk[k].quantile(.1):5.1f}  min {wk[k].min():5.1f}"
          f"  share of weeks with I >= 40: {np.mean(wk[k] >= 40) * 100:5.1f}%")
