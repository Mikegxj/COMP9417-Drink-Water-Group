"""All Stage 2 metric tables for the demand task, with both baselines and both metrics.

Spec: "Report both baselines throughout" and "Report the following metrics for every evaluated
period: MAPE and MAE"; "Report the frozen and adapted models separately, by year and by month";
monthly error with 95% intervals computed two ways (Appendix A).

Models: MLP, Ridge (+VIC/QLD), LightGBM (reproduction), N-BEATSx (3-seed mean), each frozen
(fit 2010-2014) and periodic (refits 1 Jan 2017/19/21/23/25 on the previous four years);
baselines: same half-hour yesterday (day D) and same half-hour last week (day D-6).
Monthly intervals: (a) half-hours independent, MAPE +/- 1.96 s / sqrt(n); (b) day-block
bootstrap, 1000 draws (whole days resampled within the month).

Outputs (outputs/protocol/):
  metrics_summary.csv   model x strategy x period (val 2014 = Stage 1 fit, 2015-24, 2025-26, 2015-26)
  metrics_by_year.csv   model x strategy x year
  metrics_by_month.csv  model x strategy x month, with both intervals
  monthly_ci_width_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd

import features as F

sys.path.insert(0, F.HARNESS)
import dayahead as da  # noqa: E402

OUT = os.path.join(F.OUT_ROOT, "protocol")
z = dict(np.load(os.path.join(OUT, "predictions.npz")))
z.update(np.load(os.path.join(OUT, "predictions_nbeats.npz")))
days = pd.DatetimeIndex(z["days"])
truth = z["truth"]
yr = days.year.to_numpy()
month = days.to_period("M")
RETRAINS = [2017, 2019, 2021, 2023, 2025]

df = F.load()
d_all, arr = F.to_daily(df)
idx = np.arange(7, len(d_all))
assert (d_all[idx] == days).all() and np.allclose(arr["nsw_demand"][idx], truth)
Y = arr["nsw_demand"]
BASE = {"baseline: same half-hour yesterday": Y[idx - 1], "baseline: same half-hour last week": Y[idx - 7]}

# spot check against the authoritative harness definition
for t in ("2016-03-14", "2021-07-02", "2026-10-31"):
    h, w = da.make_inputs(df, t)
    k = int(np.flatnonzero(days == pd.Timestamp(t))[0])
    for lag, name in ((1, "baseline: same half-hour yesterday"), (7, "baseline: same half-hour last week")):
        assert np.allclose(da.baseline(h, w, lag)["nsw_demand_mw"].to_numpy(), BASE[name][k])
print("baselines match harness/dayahead.baseline")

MODELS = {"MLP": "mlp", "Ridge": "ridge_vicqld", "LightGBM": "lgbm", "N-BEATSx": "nbeats"}


def strategy(key, kind):
    if kind == "stage1":
        return z[f"{key}|stage1"]
    pred = z[f"{key}|frozen"].copy()
    if kind == "periodic":
        for y0 in RETRAINS:
            pred[yr >= y0] = z[f"{key}|{y0}"][yr >= y0]
    return pred


series = []                                            # (model, strategy, stage-1 pred, pred)
for name, key in MODELS.items():
    for kind in ("frozen", "periodic"):
        series.append((name, kind, strategy(key, "stage1"), strategy(key, kind)))
for name, pred in BASE.items():
    series.append((name, "-", pred, pred))


def metrics(pred, m):
    return {"MAPE (%)": da.mape(truth[m].ravel(), pred[m].ravel()),
            "MAE (MW)": da.mae(truth[m].ravel(), pred[m].ravel())}


PERIODS = {"2015-24": (yr >= 2015) & (yr <= 2024), "2025-26": yr >= 2025, "2015-26": yr >= 2015}
summ, by_year, by_month = [], [], []
rng = np.random.default_rng(0)
for name, kind, p1, pred in series:
    r = {"model": name, "strategy": kind}
    for k, v in metrics(p1, yr == 2014).items():
        r[f"val 2014 {k}"] = v
    for per, m in PERIODS.items():
        for k, v in metrics(pred, m).items():
            r[f"{per} {k}"] = v
    summ.append(r)
    for y in range(2015, 2027):
        by_year.append({"model": name, "strategy": kind, "year": y, **metrics(pred, yr == y)})
    A = np.abs(pred - truth) / truth * 100
    for mo in sorted(set(month[yr >= 2015])):
        mm = np.asarray(month == mo)
        a = A[mm]
        flat, k = a.ravel(), a.shape[0]
        mp, half = flat.mean(), 1.96 * flat.std(ddof=1) / np.sqrt(flat.size)
        lo, hi = np.percentile(a.mean(axis=1)[rng.integers(0, k, (1000, k))].mean(axis=1), [2.5, 97.5])
        by_month.append({"model": name, "strategy": kind, "month": str(mo), "n half-hours": flat.size,
                         "MAPE (%)": mp, "MAE (MW)": float(np.abs(pred[mm] - truth[mm]).mean()),
                         "normal_low": mp - half, "normal_high": mp + half, "boot_low": lo, "boot_high": hi,
                         "normal_width": 2 * half, "boot_width": hi - lo})

summ = pd.DataFrame(summ)
by_year = pd.DataFrame(by_year)
by_month = pd.DataFrame(by_month)
by_month["width_ratio"] = by_month.boot_width / by_month.normal_width
by_month["equiv_independent_obs"] = (by_month["n half-hours"] / by_month.width_ratio ** 2).round()
widths = by_month.groupby(["model", "strategy"], sort=False).agg(
    median_normal_width=("normal_width", "median"), median_boot_width=("boot_width", "median"),
    median_ratio=("width_ratio", "median"), min_ratio=("width_ratio", "min"), max_ratio=("width_ratio", "max"),
    median_equiv_independent_obs=("equiv_independent_obs", "median"))

summ.to_csv(os.path.join(OUT, "metrics_summary.csv"), index=False)
by_year.to_csv(os.path.join(OUT, "metrics_by_year.csv"), index=False)
by_month.to_csv(os.path.join(OUT, "metrics_by_month.csv"), index=False)
widths.to_csv(os.path.join(OUT, "monthly_ci_width_summary.csv"))

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
show = summ.set_index(["model", "strategy"])
print(show.round({c: (3 if "MAPE" in c else 0) for c in show.columns}).to_string())
print()
print(widths.round(2).to_string())
