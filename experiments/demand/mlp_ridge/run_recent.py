"""How good are the two models once retrained on recent data?

Each window is fitted only on days up to 31 Dec 2024 and tested on 1 Jan 2025 - 31 Oct 2026,
which no fit or selection ever sees. Hyperparameters are chosen on the last year of the
window (2024): Ridge alpha by validation MAPE, MLP epochs by early stopping. The model is then
refitted on the whole window, exactly as a retraining at the end of 2024 could do.

Windows: expanding 2010-2024, sliding 2021-2024 (4 years) and sliding 2023-2024 (2 years).
"""
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import features as F
import models as M

OUT = F.OUT_ROOT
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

df = F.load()
days, arr = F.to_daily(df)
cal = F.calendar(days)
Y = arr["nsw_demand"]
idx = np.arange(7, len(days))
d_idx = days[idx]
yr = d_idx.year.to_numpy()
L = F.level(arr)
Lr = L[idx][:, None]
R = Y[idx] / Lr                                    # everything is scored in ratio space
TEST = np.asarray(d_idx >= pd.Timestamp("2025-01-01"))

frames = [F.ridge_frame(arr, cal, L, idx, p) for p in range(F.P)]
X = F.mlp_matrix(arr, cal, L, idx)

preds = {"baseline_yesterday": Y[idx - 1] / Lr, "baseline_lastweek": Y[idx - 7] / Lr}

# frozen models from run_trial.py (trained 2010-2013), for comparison
trial = os.path.join(OUT, "predictions.csv.gz")
if os.path.exists(trial):
    t = pd.read_csv(trial, parse_dates=["day"])
    for k in ("ridge_ratio", "mlp_ratio"):
        wide = t.pivot(index="day", columns="period", values=k).reindex(d_idx).to_numpy()
        preds[f"{k.split('_')[0]}|frozen_2010_2013"] = wide / Lr

WINDOWS = {"expand_2010_2024": (2010, 2024), "slide_2021_2024": (2021, 2024),
           "slide_2023_2024": (2023, 2024)}
SEEDS = range(5)

for name, (a, b) in WINDOWS.items():
    t0 = time.time()
    train = (yr >= a) & (yr <= b)
    inner_tr, inner_va = (yr >= a) & (yr <= b - 1), yr == b
    rp, alpha, scores = M.ridge_select_and_fit(frames, R, inner_tr, inner_va, train)
    res = Parallel(n_jobs=len(SEEDS))(
        delayed(M.mlp_seed_run)(X, R, inner_tr, inner_va, train, s) for s in SEEDS)
    mp = np.mean([r[0] for r in res], axis=0)
    preds[f"ridge|{name}"], preds[f"mlp|{name}"] = rp, mp
    preds[f"ens|{name}"] = (rp + mp) / 2
    print(f"{name}: ridge alpha={alpha} (val {scores[alpha]:.2f}%), "
          f"mlp epochs={[r[1] for r in res]}, {time.time() - t0:.0f}s")


def mape(k, mask):
    return M.ratio_mape(preds[k][mask], R[mask])


# --- overall test results ----------------------------------------------------------------
y25, y26 = TEST & (yr == 2025), TEST & (yr == 2026)
summary = pd.DataFrame({k: {
    "MAPE test": mape(k, TEST), "MAPE 2025": mape(k, y25), "MAPE 2026 (Jan-Oct)": mape(k, y26),
    "MAE test (MW)": float(np.mean(np.abs(preds[k][TEST] - R[TEST]) * Lr[TEST])),
} for k in preds}).T
summary["I vs yesterday (%)"] = 100 * (1 - summary["MAPE test"] / summary.loc["baseline_yesterday", "MAPE test"])
print("\n=== Test 2025-01 .. 2026-10 ===")
print(summary.round(2))

mid, eve = slice(23, 27), slice(37, 40)
seg = pd.DataFrame({k: {"midday 11:30-13:30": M.ratio_mape(preds[k][TEST][:, mid], R[TEST][:, mid]),
                        "evening 18:30-20:00": M.ratio_mape(preds[k][TEST][:, eve], R[TEST][:, eve]),
                        "midday bias (%)": float(np.mean((preds[k][TEST][:, mid] - R[TEST][:, mid])
                                                         / R[TEST][:, mid]) * 100)}
                    for k in preds}).T
print("\n=== Test period by time of day ===")
print(seg.round(2))

# --- monthly -----------------------------------------------------------------------------
month = d_idx.to_period("M")
months = sorted(set(month[TEST]))
monthly = pd.DataFrame({k: [mape(k, TEST & np.asarray(month == m)) for m in months] for k in preds},
                       index=[str(m) for m in months])
show = ["baseline_yesterday", "mlp|frozen_2010_2013"] + [f"{m}|slide_2021_2024" for m in ("ridge", "mlp", "ens")] \
       + ["ens|expand_2010_2024", "ens|slide_2023_2024"]
show = [c for c in show if c in monthly]
print("\n=== Monthly MAPE (%) ===")
print(monthly[show].round(2))

# --- weekly improvement, the quantity scored in the live week ------------------------------
dow = d_idx.dayofweek.to_numpy()
week = np.asarray(d_idx - pd.to_timedelta(dow, unit="D"))
rows = []
for w in sorted(set(week[TEST])):
    m = TEST & (week == w)
    if m.sum() < 7:
        continue
    b = mape("baseline_yesterday", m)
    rows.append({"week": pd.Timestamp(w).date(), "baseline": b,
                 **{k: 100 * (1 - mape(k, m) / b) for k in preds if "|" in k}})
weekly = pd.DataFrame(rows).set_index("week")
q = weekly.drop(columns="baseline").describe(percentiles=[0.1, 0.5, 0.9]).T[["10%", "50%", "90%", "min"]]
q["share I>=40 (%)"] = (weekly.drop(columns="baseline") >= 40).mean() * 100
q["share I>=30 (%)"] = (weekly.drop(columns="baseline") >= 30).mean() * 100
print(f"\n=== Weekly improvement I over {len(weekly)} Mon-Sun test weeks ===")
print(q.round(1))
dec = weekly[[pd.Timestamp(w).month == 12 for w in weekly.index]]
print("\n=== December 2025 weeks: baseline MAPE and I (%) ===")
print(dec[["baseline"] + [c for c in show if c != "baseline_yesterday"]].round(1))

summary.to_csv(os.path.join(OUT, "recent_summary.csv"))
seg.to_csv(os.path.join(OUT, "recent_time_of_day.csv"))
monthly.to_csv(os.path.join(OUT, "recent_monthly.csv"))
weekly.to_csv(os.path.join(OUT, "recent_weekly_I.csv"))
print(f"\nsaved to {OUT}")
