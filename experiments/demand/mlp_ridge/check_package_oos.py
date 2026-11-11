"""End-to-end check of the MLP package against the experiments (run with the Python 3.11 env).

1. Copies the package's model.py and train.py (features.PACKAGE: package_mlp/ locally,
   submission_mlp/ in the group repository) into a work folder and runs train.py on the CSV cut
   at 2024-12-31, i.e. the same four-year window (2021-2024) as the 2025 refit in run_protocol.py.
2. Calls Forecaster.predict one day at a time (history up to day D, weather of D+1), as in
   marking, for every 5th day from 2025-01-01 to 2026-10-31.
3. Compares with the experiment MLP (Python 3.13 / torch 2.14, outputs/protocol/predictions.npz,
   fit "2025") on the same days.
Usage: python check_package_oos.py WORK_DIR
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import features as F  # noqa: E402  (sets the harness path, data file, output root and package folder)
import dayahead as da  # noqa: E402

work = os.path.abspath(sys.argv[1])
pkg = os.path.join(work, "pkg")
os.makedirs(pkg, exist_ok=True)
for f in ("model.py", "train.py"):
    shutil.copy(os.path.join(F.PACKAGE, f), pkg)

df = da.load_dataset(F.DATA)
day = da.market_day(df.index)
cut = os.path.join(work, "data_to_2024.csv")
df[day <= pd.Timestamp("2024-12-31")].to_csv(cut)
subprocess.run([sys.executable, os.path.join(pkg, "train.py"), cut], check=True)

sys.path.insert(0, pkg)
import model  # noqa: E402

fc = model.Forecaster(os.path.join(pkg, "artifacts"))
z = np.load(os.path.join(F.OUT_ROOT, "protocol", "predictions.npz"))
exp_days = pd.DatetimeIndex(z["days"])
targets = pd.date_range("2025-01-01", "2026-10-31", freq="5D")
pk, ex, tr = [], [], []
for d in targets:
    k = int(np.searchsorted(day.unique(), d))          # position of market day d
    history = df.iloc[:k * 48]
    weather_next = df.iloc[k * 48:(k + 1) * 48][da.WEATHER_COLS]
    pk.append(fc.predict(history, weather_next)["nsw_demand_mw"].to_numpy())
    j = exp_days.get_loc(d)
    ex.append(z["mlp|2025"][j])
    tr.append(df["nsw_demand"].to_numpy()[k * 48:(k + 1) * 48])
pk, ex, tr = map(np.array, (pk, ex, tr))
m = lambda p: float(np.mean(np.abs(p - tr) / tr) * 100)  # noqa: E731
print(f"\n{len(targets)} target days (every 5th day, 2025-01 .. 2026-10)")
print(f"MAPE  package (Python {sys.version.split()[0]}): {m(pk):.3f}%   experiment (3.13): {m(ex):.3f}%")
print(f"package vs experiment forecasts: mean |diff| {np.mean(np.abs(pk - ex) / ex) * 100:.3f}% "
      f"(max {np.max(np.abs(pk - ex) / ex) * 100:.2f}%), mean diff {np.mean(pk - ex):.1f} MW")
