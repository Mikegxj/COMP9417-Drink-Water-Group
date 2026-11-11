"""Feature ablations for NSW day-ahead demand (Ridge and MLP).

Questions:
  features: Do Melbourne/Brisbane weather or VIC/QLD demand improve NSW forecasts? Do leaner
            feature sets trade validation accuracy for robustness when frozen?
  holidays: How much do the NSW public-holiday and Christmas-period flags contribute?

Stage 1 protocol: fit on 2010-2013 and select on 2014 (Ridge alpha on 2014; MLP epochs by
early stopping on 2013 with a 2010-2012 fit, then refit on 2010-2013). The 2014 difference to
the base set gets a 95% day-block bootstrap interval (days resampled, 48 half-hours kept
together). Models are then kept frozen and scored on 2015-2026. Errors are also broken down
by type of target day (holidays, days after holidays, Christmas period).

Usage: python run_ablation.py [features|holidays]   (default: features)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import features as F
import models as M

OUT = F.OUT_ROOT
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

GROUPS = {
    "features": {
        "base (Sydney only)": {},
        "+ Mel/Bne weather": {"cities": ["mel", "bne"]},
        "+ VIC/QLD demand": {"regions": ["vic", "qld"]},
        "+ Mel/Bne + VIC/QLD": {"cities": ["mel", "bne"], "regions": ["vic", "qld"]},
        "- solar & cloud": {"no_solar_cloud": True},
        "minimal (teammate-like)": {"minimal": True},
    },
    "holidays": {
        "base (Sydney only)": {},
        "- public holidays": {"no_holidays": True},
        "- public holidays & Christmas flag": {"no_holidays": True, "no_xmas": True},
    },
}
GROUP = sys.argv[1] if len(sys.argv) > 1 else "features"
CONFIGS = GROUPS[GROUP]
SEEDS = range(5)

df = F.load()
days, arr = F.to_daily(df)
cal = F.calendar(days)
Y = arr["nsw_demand"]
idx = np.arange(7, len(days))
yr = days[idx].year.to_numpy()
L = F.level(arr)
R = Y[idx] / L[idx][:, None]
TRAIN, VAL = yr <= 2013, yr == 2014
INNER_TR, INNER_VA = yr <= 2012, yr == 2013
FROZEN, LATE = yr >= 2015, yr >= 2023
mid = slice(23, 27)                                    # 11:30-13:30


def day_ape(pred):
    return np.mean(np.abs(pred - R) / R, axis=1) * 100   # one value per target day


def boot_ci(d, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(d)
    stats = [d[rng.integers(0, k, k)].mean() for _ in range(n_boot)]
    return np.percentile(stats, [2.5, 97.5])


preds = {}
for name, opts in CONFIGS.items():
    t0 = time.time()
    frames = [F.ridge_frame(arr, cal, L, idx, p, opts) for p in range(F.P)]
    rp, alpha, _ = M.ridge_select_and_fit(frames, R, TRAIN, VAL, TRAIN)
    X = F.mlp_matrix(arr, cal, L, idx, opts)
    res = Parallel(n_jobs=len(SEEDS))(
        delayed(M.mlp_seed_run)(X, R, INNER_TR, INNER_VA, TRAIN, s) for s in SEEDS)
    preds[("Ridge", name)] = rp
    preds[("MLP", name)] = np.mean([r[0] for r in res], axis=0)
    print(f"{name:<26} ridge cols/period={frames[0].shape[1]:<3} alpha={alpha:<5} "
          f"mlp inputs={X.shape[1]:<4} epochs={[r[1] for r in res]}  {time.time() - t0:.0f}s",
          flush=True)

rows = []
for (model, name), pr in preds.items():
    base = preds[(model, "base (Sydney only)")]
    d = (day_ape(pr) - day_ape(base))[VAL]
    lo, hi = boot_ci(d)
    rows.append({
        "model": model, "features": name,
        "val 2014": M.ratio_mape(pr[VAL], R[VAL]),
        "diff vs base (pp)": d.mean(), "95% CI low": lo, "95% CI high": hi,
        "frozen 2015-26": M.ratio_mape(pr[FROZEN], R[FROZEN]),
        "frozen 2023-26": M.ratio_mape(pr[LATE], R[LATE]),
        "midday bias 2023-26 (%)": float(np.mean((pr[LATE][:, mid] - R[LATE][:, mid])
                                                 / R[LATE][:, mid]) * 100),
    })
table = pd.DataFrame(rows).set_index(["model", "features"])
yday = Y[idx - 1] / L[idx][:, None]
print(f"\nyesterday baseline: val 2014 {M.ratio_mape(yday[VAL], R[VAL]):.2f}, "
      f"2015-26 {M.ratio_mape(yday[FROZEN], R[FROZEN]):.2f}, 2023-26 {M.ratio_mape(yday[LATE], R[LATE]):.2f}")
print(f"\n=== Ablation '{GROUP}' (MAPE %, frozen = trained on 2010-2013, never retrained) ===")
print(table.round(3))
name = f"ablation_{GROUP}.csv"
table.to_csv(os.path.join(OUT, name))

# --- error by type of target day: the calendar effects live on a few days a year ----------
hol = cal["hol"].to_numpy()
dow = cal["dow"].to_numpy()[idx]
h_t, h_y, h_7 = hol[idx] == 1, hol[idx - 1] == 1, hol[idx - 7] == 1
xm = cal["xmas"].to_numpy()[idx] == 1
daytypes = {
    "public holiday (Mon-Fri)": h_t & (dow < 5),
    "day after a public holiday": h_y & ~h_t & (dow < 5),
    "same day last week was a holiday": h_7 & ~h_t & ~h_y,
    "Christmas-New Year period": xm & ~h_t,
    "all other days": ~(h_t | h_y | h_7 | xm),
}
rows = []
for window, wmask in (("2014 (validation)", VAL), ("2014-2018 (frozen)", (yr >= 2014) & (yr <= 2018))):
    for dt, m in daytypes.items():
        mm = wmask & m
        r = {"window": window, "day type": dt, "n days": int(mm.sum()),
             "yesterday baseline": M.ratio_mape(yday[mm], R[mm])}
        for (model, cfg), pr in preds.items():
            r[f"{model} | {cfg}"] = M.ratio_mape(pr[mm], R[mm])
        rows.append(r)
by_type = pd.DataFrame(rows).set_index(["window", "day type"])
print("\n=== MAPE (%) by type of target day ===")
print(by_type.round(2).T)
by_type.to_csv(os.path.join(OUT, f"ablation_{GROUP}_by_daytype.csv"))
print(f"\nsaved {name} and ablation_{GROUP}_by_daytype.csv to {OUT}")
