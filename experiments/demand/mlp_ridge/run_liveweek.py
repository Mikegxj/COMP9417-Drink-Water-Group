"""Evidence for the live-week forecast (7-13 Dec 2026, Monday to Sunday).

The final package is trained on the four years up to 31 Oct 2026 and then predicts a week in
December. This replays that exact situation for every earlier year Y = 2015..2025: train the
MLP (5 seeds, 55 epochs) on target days in (31 Oct Y - 4 years, 31 Oct Y], then predict the
Monday-Sunday weeks of December Y that end by 21 Dec (before the Christmas period). The week
starting on the Monday between 7 and 13 Dec is the closest analogue of the live week.
Reports weekly MAPE of the MLP and of the same-half-hour-yesterday baseline, the improvement
I = 100 (1 - MAPE_model / MAPE_baseline) used in the live marking, and a trend over years.
Outputs: outputs/liveweek/weeks.csv.
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import features as F
import models as M

OUT = os.path.join(F.OUT_ROOT, "liveweek")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)

df = F.load()
days, arr = F.to_daily(df)
cal = F.calendar(days)
Y = arr["nsw_demand"]
idx = np.arange(7, len(days))
d_idx = days[idx]
L = F.level(arr)
Lr = L[idx][:, None]
R = Y[idx] / Lr
X = F.mlp_matrix(arr, cal, L, idx)
truth = Y[idx]
yday = Y[idx - 1]
YEARS = range(2015, 2026)

jobs = []
for y in YEARS:
    end = pd.Timestamp(f"{y}-10-31")
    train = np.asarray((d_idx > end - pd.DateOffset(years=4)) & (d_idx <= end))
    jobs += [(y, s, train) for s in range(5)]
res = Parallel(n_jobs=8, verbose=10)(delayed(M.mlp_fit_predict)(X, R, tr, s, 55) for y, s, tr in jobs)
pred = {y: np.mean([r for (y_, s, _), r in zip(jobs, res) if y_ == y], axis=0) * Lr for y in YEARS}


def mape(p, m):
    return float(np.mean(np.abs(p[m] - truth[m]) / truth[m]) * 100)


rows = []
for y in YEARS:
    for mon in pd.date_range(f"{y}-12-01", f"{y}-12-15", freq="W-MON"):
        if mon + pd.Timedelta(days=6) > pd.Timestamp(f"{y}-12-21"):
            continue
        m = np.asarray((d_idx >= mon) & (d_idx <= mon + pd.Timedelta(days=6)))
        bm = mape(yday, m)
        rows.append({"year": y, "week from": mon.date(), "analogue (Mon 7-13 Dec)": 7 <= mon.day <= 13,
                     "days after training end": (mon - pd.Timestamp(f"{y}-10-31")).days,
                     "MLP MAPE": mape(pred[y], m), "baseline MAPE": bm,
                     "I (%)": 100 * (1 - mape(pred[y], m) / bm)})
wk = pd.DataFrame(rows)
wk.to_csv(os.path.join(OUT, "weeks.csv"), index=False)
print(wk.round(2).to_string(index=False))

a = wk[wk["analogue (Mon 7-13 Dec)"]]
print("\nAnalogue weeks only (one per year):")
print(a[["year", "MLP MAPE", "baseline MAPE", "I (%)"]].round(2).to_string(index=False))
for name, g in (("all December weeks", wk), ("analogue weeks", a), ("analogue weeks 2021-2025", a[a.year >= 2021])):
    print(f"{name:<26} MLP MAPE median {g['MLP MAPE'].median():.2f} [min {g['MLP MAPE'].min():.2f}, max {g['MLP MAPE'].max():.2f}]"
          f" | I median {g['I (%)'].median():.1f} [min {g['I (%)'].min():.1f}, max {g['I (%)'].max():.1f}]")
coef = np.polyfit(a.year, a["MLP MAPE"], 1)
resid = a["MLP MAPE"] - np.polyval(coef, a.year)
print(f"linear trend of analogue-week MAPE: {coef[0]:+.3f} pp per year; 2026 extrapolation "
      f"{np.polyval(coef, 2026):.2f} +/- {2 * resid.std(ddof=2):.2f} (2 residual s.d.)")
