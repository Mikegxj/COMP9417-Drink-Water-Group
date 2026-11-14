"""Export the selected demand strategy's half-hourly predictions as a plain CSV.

Selected strategy: 5-seed MLP, frozen fit on 2010-2014 for 2015-2016, then periodic refits on
1 Jan 2017/19/21/23/25, each on the previous six years (run_window_length.py, "6 years").
Built exactly as the "periodic 6y (selected)" series in run_metrics_all.py, so the printed MAPE
matches protocol/metrics_summary.csv.

Needs protocol/predictions.npz (run_protocol.py) and window_length/predictions_window.npz
(run_window_length.py).

Output (outputs/window_length/predictions_mlp_6years.csv), one row per half-hour 2015-01-01 ..
2026-10-31 (market days):
  timestamp      interval end, as in the data file (market day D = D 00:30 .. D+1 00:00)
  train_years    years the model that made the forecast was fitted on
  nsw_demand_mw  observed demand
  mlp_6y_mw      forecast made at the end of the previous market day
"""
import os

import numpy as np
import pandas as pd

import features as F

OUT = os.path.join(F.OUT_ROOT, "protocol")
z = np.load(os.path.join(OUT, "predictions.npz"))
wz = np.load(os.path.join(F.OUT_ROOT, "window_length", "predictions_window.npz"))
days = pd.DatetimeIndex(z["days"])
truth = z["truth"]
yr = days.year.to_numpy()
RETRAINS = [2017, 2019, 2021, 2023, 2025]
YEARS = 6

pred = z["mlp|frozen"].copy()
fit = np.full(len(days), "2010-2014", dtype=object)
for y0 in RETRAINS:
    pred[yr >= y0] = wz[f"{YEARS} years|{y0}"][yr >= y0]
    fit[yr >= y0] = f"{y0 - YEARS}-{y0 - 1}"

# interval-end timestamps from the data file, aligned to the prediction days
df = F.load()
d_all, arr = F.to_daily(df)
stamps = np.asarray(df.index).reshape(len(d_all), F.P)
pos = d_all.get_indexer(days)
assert (pos >= 0).all() and np.allclose(arr["nsw_demand"][pos], truth)

keep = yr >= 2015
k = np.flatnonzero(keep)
out = pd.DataFrame({
    "timestamp": pd.DatetimeIndex(stamps[pos[k]].ravel()).strftime("%Y-%m-%d %H:%M"),
    "train_years": np.repeat(fit[k], F.P),
    "nsw_demand_mw": np.round(truth[k].ravel(), 2),
    "mlp_6y_mw": np.round(pred[k].ravel(), 1),
})
path = os.path.join(F.OUT_ROOT, "window_length", "predictions_mlp_6years.csv")
out.to_csv(path, index=False)

ape = np.abs(pred - truth) / truth * 100
for name, m in (("2015-26", keep), ("2015-24", keep & (yr <= 2024)), ("2025-26", yr >= 2025)):
    print(f"MAPE {name}: {ape[m].mean():.3f} %")
print(f"wrote {path}: {len(out)} rows, {os.path.getsize(path) / 1e6:.1f} MB")
