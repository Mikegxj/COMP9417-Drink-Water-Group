"""Second iteration of drift-triggered retraining for the MLP (designed after run_drift.py showed
that fixed 2014 thresholds exhaust the budget by 2018-2020; report it as a second iteration).

Two changes, tested separately and together (q = 0.99 throughout, as before):
  recalibration  after every retraining at the end of day D, a calibration model is fitted on the
                 first three years of the new four-year window and scored on the held-out last
                 year; the error threshold becomes the q-quantile of its 28-day MLP/baseline ratio
                 there. (The input detector already re-fitted its reference this way.) This asks
                 "is the new model worse than a fresh model was under recent conditions?"
  pacing         a retrain is allowed only after a minimum gap of
                 (days left to Oct 2026) / (retrains left + 1) since the last deployment, so the
                 budget of 6 cannot be spent in the first years. Without pacing: 180-day gap.
Everything else as in run_drift.py. Writes outputs/drift/strategies_v2.csv and
strategies_v2_monthly.csv.
"""
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import drift as Dt
import features as F
import models as M

OUT = os.path.join(F.OUT_ROOT, "drift")
SEEDS, EPOCHS, BUDGET, COOLDOWN, Q = range(5), 55, 6, 180, 0.99
pd.set_option("display.width", 220)

z = np.load(os.path.join(F.OUT_ROOT, "protocol", "predictions.npz"))
df = F.load()
days, arr = F.to_daily(df)
cal = F.calendar(days)
Y = arr["nsw_demand"]
idx = np.arange(7, len(days))
d_idx = days[idx]
yr = d_idx.year.to_numpy()
L = F.level(arr)
Lr = L[idx][:, None]
R = Y[idx] / Lr
X = F.mlp_matrix(arr, cal, L, idx)
truth = Y[idx]
n = len(idx)
eligible = ~((cal["hol"].to_numpy()[idx] == 1) | (cal["xmas"].to_numpy()[idx] == 1))
T0 = int(np.argmax(yr >= 2015))
SIM, P1, P2 = yr >= 2015, (yr >= 2015) & (yr <= 2024), yr >= 2025
REF_FIT, REF_THR = (yr >= 2010) & (yr <= 2013), yr == 2014
month = d_idx.to_period("M")


def daily_ape(pred):
    return (np.abs(pred - truth) / truth * 100).mean(axis=1)


def mape(pred, mask):
    return float((np.abs(pred - truth) / truth * 100)[mask].mean())


b = daily_ape(np.vstack([np.full((1, F.P), np.nan), truth[:-1]]))
frozen, stage1 = z["mlp|frozen"], z["mlp|stage1"]
shape = Dt.ShapeNovelty().fit(truth[REF_FIT & eligible])
s_daily = shape.score(truth)
s_ref = Dt.rolling_eligible(s_daily[REF_THR], eligible[REF_THR])
r_ref = Dt.rolling_ratio(daily_ape(stage1)[REF_THR], b[REF_THR], eligible[REF_THR])

cache = {}


def fit_window(lo, hi):
    """5-seed MLP trained on target days in (lo, hi]; MW predictions for every row."""
    key = (lo, hi)
    if key not in cache:
        train = np.asarray((d_idx > lo) & (d_idx <= hi))
        res = Parallel(n_jobs=len(SEEDS))(delayed(M.mlp_fit_predict)(X, R, train, s, EPOCHS, 4) for s in SEEDS)
        cache[key] = np.mean(res, axis=0) * Lr
    return cache[key]


def retrain(t):
    D = d_idx[t]
    return fit_window(D - pd.DateOffset(years=4), D)


def error_threshold(t):
    D = d_idx[t]
    pred = fit_window(D - pd.DateOffset(years=4), D - pd.DateOffset(years=1))
    held = np.asarray((d_idx > D - pd.DateOffset(years=1)) & (d_idx <= D))
    return float(np.nanquantile(Dt.rolling_ratio(daily_ape(pred)[held], b[held], eligible[held]), Q))


def input_reference(t):
    D = d_idx[t]
    win = np.asarray((d_idx > D - pd.DateOffset(years=4)) & (d_idx <= D))
    fit = win & np.asarray(d_idx <= D - pd.DateOffset(years=1))
    held = win & ~fit
    sc = Dt.ShapeNovelty().fit(truth[fit & eligible]).score(truth)
    return sc, float(np.nanquantile(Dt.rolling_eligible(sc[held], eligible[held]), Q))


def simulate(kind, recal, pace):
    current, pred, retr, last, thr_log = frozen, frozen.copy(), [], T0, []
    win = Dt.OnlineWindow()
    sc = s_daily
    th = float(np.nanquantile(r_ref if kind == "error" else s_ref, Q))
    thr_log.append(round(th, 3))
    for t in range(T0, n):
        pred[t] = current[t]
        if eligible[t]:
            if kind == "error":
                win.push(float(np.mean(np.abs(current[t] - truth[t]) / truth[t]) * 100), b[t])
            else:
                win.push(sc[t])
        stat = (win.ratio() if kind == "error" else win.mean()) if win.full else -np.inf
        k = len(retr)
        gap = (n - last) / (BUDGET - k + 1) if pace else COOLDOWN
        if stat > th and (t + 1 - last) >= gap and k < BUDGET and t + 1 < n:
            current = retrain(t)
            retr.append(d_idx[t + 1].date())
            last = t + 1
            win.clear()
            if kind == "input":
                sc, th = input_reference(t)
            elif recal:
                th = error_threshold(t)
            thr_log.append(round(th, 3))
    return pred, retr, thr_log


PLANS = [("error: recalibrated + paced", "error", True, True),
         ("error: recalibrated, 180-day gap", "error", True, False),
         ("error: fixed 2014 threshold + paced", "error", False, True),
         ("input: re-fitted reference + paced", "input", True, True)]
rows, monthly = [], {}
for name, kind, recal, pace in PLANS:
    t0 = time.time()
    pred, retr, thr_log = simulate(kind, recal, pace)
    rows.append({"strategy": name, "retrains": len(retr), "dates": " ".join(map(str, retr)),
                 "thresholds": " ".join(map(str, thr_log)),
                 "MAPE 2015-24": mape(pred, P1), "MAPE 2025-26": mape(pred, P2), "MAPE all": mape(pred, SIM)})
    monthly[name] = [mape(pred, np.asarray(month == m)) for m in sorted(set(month[SIM]))]
    print(f"{name:<40} retrains={len(retr)} all={rows[-1]['MAPE all']:.3f}  {time.time() - t0:.0f}s", flush=True)
v2 = pd.DataFrame(rows).set_index("strategy")
v2.to_csv(os.path.join(OUT, "strategies_v2.csv"))
pd.DataFrame(monthly, index=[str(m) for m in sorted(set(month[SIM]))]).to_csv(
    os.path.join(OUT, "strategies_v2_monthly.csv"))
v1 = pd.read_csv(os.path.join(OUT, "strategies.csv"), index_col=0)
keep = ["frozen", "periodic, 5 evenly spaced", "periodic, 6 evenly spaced",
        "periodic, 1 Jan 2017/19/21/23/25 (teammate)", "error-triggered q99", "input-triggered q99"]
print("\n=== Second iteration vs first iteration and periodic (MLP, MAPE %) ===")
print(pd.concat([v1.loc[keep, ["retrains", "dates", "MAPE 2015-24", "MAPE 2025-26", "MAPE all"]],
                 v2[["retrains", "dates", "MAPE 2015-24", "MAPE 2025-26", "MAPE all"]]]).round(3).to_string())
print("\nthresholds used (initial, then after each retrain):")
print(v2.thresholds.to_string())
