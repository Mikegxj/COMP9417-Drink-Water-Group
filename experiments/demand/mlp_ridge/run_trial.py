"""Trial run: per-half-hour Ridge (with temperature splines) vs MLP for NSW day-ahead demand.

Protocol (Stage 1 style): train on target days in 2010-2013, select hyperparameters on 2014.
The fitted models are then kept frozen and scored on every later year, to get a first look
at degradation. Each model is run with a ratio target (demand / mean of the previous 7 days)
and a raw target (demand / constant).
"""
import os
import time

import numpy as np
import pandas as pd
import torch

import features as F
from features import da
from models import fit_mlp, ridge_pipeline

OUT = F.OUT_ROOT
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)

df = F.load()
days, arr = F.to_daily(df)
cal = F.calendar(days)
Y = arr["nsw_demand"]
n = len(days)
year = days.year.to_numpy()
idx = np.arange(7, n)                        # target days with a full week of history
yr = year[idx]
TRAIN, VAL = yr <= 2013, yr == 2014
EVAL = yr >= 2014
CONST = Y[idx[TRAIN]].mean()                 # scale for raw mode (training data only)

# --- sanity check against the official harness -------------------------------------------
for j in (idx[TRAIN][100], idx[VAL][50], idx[-1]):
    hist, wnext = da.make_inputs(df, days[j])
    b1, b7 = da.baseline(hist, wnext, 1), da.baseline(hist, wnext, 7)
    tr = da.targets(df, days[j])
    assert np.allclose(b1["nsw_demand_mw"], Y[j - 1]) and np.allclose(b7["nsw_demand_mw"], Y[j - 7])
    assert np.allclose(tr["nsw_demand_mw"], Y[j])
print("harness check OK: baselines and targets match the daily layout")

preds = {"baseline_yesterday": Y[idx - 1], "baseline_lastweek": Y[idx - 7]}


def mape_rows(yhat, rows):
    return da.mape(Y[idx[rows]].ravel(), yhat[rows].ravel())


# --- Ridge: one model per half-hour -------------------------------------------------------
def run_ridge(mode):
    L = F.level(arr, normalize=(mode == "ratio"), const=CONST)
    frames = [F.ridge_frame(arr, cal, L, idx, p) for p in range(F.P)]
    R = Y[idx] / L[idx, None]
    best = None
    for alpha in (0.3, 1, 3, 10, 30, 100):
        out = np.zeros_like(R)
        for p in range(F.P):
            m = ridge_pipeline(alpha, list(frames[p].columns))
            m.fit(frames[p][TRAIN], R[TRAIN, p])
            out[:, p] = m.predict(frames[p])
        yhat = out * L[idx, None]
        v = mape_rows(yhat, VAL)
        print(f"  ridge[{mode}] alpha={alpha:<5} val2014 MAPE={v:.3f}")
        if best is None or v < best[0]:
            best = (v, alpha, yhat)
    print(f"  -> ridge[{mode}] selected alpha={best[1]}")
    return best[2]


# --- MLP: one sample per day, 48 outputs ---------------------------------------------------
def run_mlp(mode, seeds=(0, 1, 2, 3, 4), max_epochs=400):
    L = F.level(arr, normalize=(mode == "ratio"), const=CONST)
    X = F.mlp_matrix(arr, cal, L, idx)
    R = Y[idx] / L[idx, None]
    inner_tr, inner_va = yr <= 2012, yr == 2013          # chronological early-stopping split
    outs = []
    for s in seeds:
        _, _, ep = fit_mlp(X[inner_tr], R[inner_tr], s, max_epochs, X[inner_va], R[inner_va])
        model, sc, _ = fit_mlp(X[TRAIN], R[TRAIN], s, max(ep, 1))   # refit on 2010-2013
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(sc.transform(X), dtype=torch.float32)).numpy()
        yhat = out * L[idx, None]
        print(f"  mlp[{mode}] seed={s} epochs={ep:<3} val2014 MAPE={mape_rows(yhat, VAL):.3f}")
        outs.append(yhat)
    return np.mean(outs, axis=0)


t0 = time.time()
for mode in ("ratio", "raw"):
    print(f"Ridge, {mode} target")
    preds[f"ridge_{mode}"] = run_ridge(mode)
for mode in ("ratio", "raw"):
    print(f"MLP, {mode} target")
    preds[f"mlp_{mode}"] = run_mlp(mode)
print(f"training done in {time.time() - t0:.0f}s\n")

# --- results ------------------------------------------------------------------------------
pd.set_option("display.width", 160)
truth = Y[idx]
val = pd.DataFrame({k: {"MAPE (%)": mape_rows(v, VAL),
                        "MAE (MW)": da.mae(truth[VAL].ravel(), v[VAL].ravel())}
                    for k, v in preds.items()}).T
b = val.loc["baseline_yesterday", "MAPE (%)"]
val["improvement vs yesterday (%)"] = 100 * (1 - val["MAPE (%)"] / b)
print("=== Validation 2014 ===")
print(val.round(2), "\n")

dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_t = cal["dow"].to_numpy()[idx]
by_dow = pd.DataFrame({k: [mape_rows(v, VAL & (dow_t == d)) for d in range(7)]
                       for k, v in preds.items()}, index=dow_names)
print("=== 2014 MAPE by weekday of the target day ===")
print(by_dow.round(2), "\n")

years = sorted(set(yr[EVAL]))
by_year = pd.DataFrame({k: [mape_rows(v, yr == y) for y in years] for k, v in preds.items()},
                       index=years)
print("=== Frozen models (trained 2010-2013), MAPE by year ===")
print(by_year.round(2), "\n")

mid, eve = slice(23, 27), slice(37, 40)            # 11:30-13:30, 18:30-20:00 (interval start)
seg = {}
for k in ("baseline_yesterday", "ridge_ratio", "mlp_ratio"):
    for name, sl in (("midday", mid), ("evening", eve)):
        seg[f"{k}|{name}"] = [da.mape(truth[yr == y][:, sl].ravel(), preds[k][yr == y][:, sl].ravel())
                              for y in years]
print("=== MAPE by year, midday vs evening ===")
print(pd.DataFrame(seg, index=years).round(2))

val.to_csv(os.path.join(OUT, "val2014.csv"))
by_dow.to_csv(os.path.join(OUT, "val2014_by_dow.csv"))
by_year.to_csv(os.path.join(OUT, "frozen_by_year.csv"))
long = pd.DataFrame({"day": np.repeat(days[idx], F.P), "period": np.tile(np.arange(F.P), len(idx)),
                     "y": truth.ravel(), **{k: v.ravel() for k, v in preds.items()}})
long.to_csv(os.path.join(OUT, "predictions.csv.gz"), index=False)
print(f"\nsaved tables and predictions to {OUT}")
