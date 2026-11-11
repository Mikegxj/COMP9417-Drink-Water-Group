"""Degradation, drift detection and drift-triggered retraining for the selected demand model (MLP).

1. Monthly MAPE of the frozen models (MLP, Ridge, LightGBM) for 2015-01..2026-10 with two 95%
   intervals: (a) half-hours independent, normal approximation 1.96 s / sqrt(n); (b) day-block
   bootstrap (days resampled with their 48 half-hours, 1000 draws).
2. Detectors on the frozen MLP with references fixed on 2010-2014 (see drift.py):
   input  = daily-shape PCA novelty, fitted on 2010-2013, threshold = quantile of the 28-day
            statistic over 2014;
   error  = 28-day MLP/baseline MAPE ratio, threshold = quantile of the same ratio for the
            Stage 1 MLP (trained 2010-2013) over 2014.
   Main threshold q = 0.99 (fixed in advance); 0.95 and the 2014 maximum as sensitivity.
3. Concept-drift check: error of the frozen model on days whose inputs stay inside the 2014
   range of both input detectors (shape and weather), year by year.
4. Retraining strategies for the MLP, at most 6 retrains in 2015-01..2026-10, each refit on the
   last four years up to day D and used from D+1: frozen; periodic with 1..6 evenly spaced
   retrains; the teammate's schedule (1 Jan 2017/19/21/23/25); error-triggered and
   input-triggered retraining (180-day minimum gap; detectors restart after each retrain, the
   input reference is refitted on the first three years of the new window and its threshold set
   on the last year).
Outputs go to outputs/drift/.
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
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
SEEDS, EPOCHS, BUDGET, COOLDOWN = range(5), 55, 6, 180
QS = {"q95": 0.95, "q99": 0.99, "max": 1.0}

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
assert np.allclose(truth, z["truth"])
n = len(idx)
eligible = ~((cal["hol"].to_numpy()[idx] == 1) | (cal["xmas"].to_numpy()[idx] == 1))
WX = np.concatenate([arr[c][idx] for c in ("syd_temp", "syd_solar", "syd_cloud", "syd_rh")], axis=1)
T0 = int(np.argmax(yr >= 2015))
SIM, P1, P2 = yr >= 2015, (yr >= 2015) & (yr <= 2024), yr >= 2025
REF_FIT, REF_THR = (yr >= 2010) & (yr <= 2013), yr == 2014
dates = np.asarray(d_idx.date)


def ape_hh(pred):
    return np.abs(pred - truth) / truth * 100


def daily_ape(pred):
    return ape_hh(pred).mean(axis=1)


def mape(pred, mask):
    return float(ape_hh(pred)[mask].mean())


yday = np.vstack([np.full((1, F.P), np.nan), truth[:-1]])
b = daily_ape(yday)
frozen, stage1 = z["mlp|frozen"], z["mlp|stage1"]

# --- 1. monthly MAPE with two intervals -------------------------------------------------------
month = d_idx.to_period("M")
rng = np.random.default_rng(0)
rows = []
for name, pred in (("mlp", frozen), ("ridge_vicqld", z["ridge_vicqld|frozen"]), ("lgbm", z["lgbm|frozen"]),
                   ("baseline_yesterday", yday)):
    A = ape_hh(pred)
    for m in sorted(set(month[SIM])):
        sel = np.asarray(month == m)
        a = A[sel]
        flat, k = a.ravel(), a.shape[0]
        mp, half = flat.mean(), 1.96 * flat.std(ddof=1) / np.sqrt(flat.size)
        boots = a.mean(axis=1)[rng.integers(0, k, (1000, k))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(model=name, month=str(m), n=flat.size, mape=mp, normal_low=mp - half,
                         normal_high=mp + half, boot_low=lo, boot_high=hi))
monthly = pd.DataFrame(rows)
monthly["width_ratio"] = (monthly.boot_high - monthly.boot_low) / (monthly.normal_high - monthly.normal_low)
monthly.to_csv(os.path.join(OUT, "monthly_frozen_ci.csv"), index=False)
print("=== Monthly intervals, frozen models: bootstrap width / normal width ===")
print(monthly.groupby("model").width_ratio.describe()[["min", "50%", "max"]].round(2), "\n")

# --- 2. detectors on the frozen MLP, references fixed on 2010-2014 ---------------------------
shape = Dt.ShapeNovelty().fit(truth[REF_FIT & eligible])
s_daily = shape.score(truth)
weather = Dt.WeatherNovelty().fit(WX[REF_FIT & eligible])
w_daily = weather.score(WX)
s_ref = Dt.rolling_eligible(s_daily[REF_THR], eligible[REF_THR])
w_ref = Dt.rolling_eligible(w_daily[REF_THR], eligible[REF_THR])
r_ref = Dt.rolling_ratio(daily_ape(stage1)[REF_THR], b[REF_THR], eligible[REF_THR])
thr = pd.DataFrame({det: {k: float(np.nanquantile(ref, q)) for k, q in QS.items()}
                    for det, ref in (("input_shape", s_ref), ("weather", w_ref), ("error_ratio", r_ref))})
print(f"PCA components: shape {shape.pca.n_components_}, weather {weather.pca.n_components_}")
print("=== Thresholds (28-day statistics over 2014) ===")
print(thr.round(4), "\n")

seg = slice(T0, n)
mon = pd.DataFrame({
    "date": dates[seg], "eligible": eligible[seg],
    "shape_daily": s_daily[seg], "shape_28d": Dt.rolling_eligible(s_daily[seg], eligible[seg]),
    "weather_daily": w_daily[seg], "weather_28d": Dt.rolling_eligible(w_daily[seg], eligible[seg]),
    "mlp_frozen_daily_mape": daily_ape(frozen)[seg], "baseline_daily_mape": b[seg],
    "error_ratio_28d": Dt.rolling_ratio(daily_ape(frozen)[seg], b[seg], eligible[seg]),
})
mon.to_csv(os.path.join(OUT, "detector_daily_frozen.csv"), index=False)
ep_rows, first = [], {}
for det, col in (("input_shape", "shape_28d"), ("weather", "weather_28d"), ("error_ratio", "error_ratio_28d")):
    for k in QS:
        e = Dt.episodes(mon.date.to_numpy(), mon[col].to_numpy(), thr.loc[k, det])
        first[(det, k)] = e.start.iloc[0] if len(e) else None
        if k == "q99":
            ep_rows.append(e.assign(detector=det))
eps = pd.concat(ep_rows)
eps.to_csv(os.path.join(OUT, "episodes_frozen_q99.csv"), index=False)
print("=== First signal on the frozen MLP ===")
print(pd.Series(first).unstack(), "\n")
print("=== Episodes above the q99 threshold (frozen MLP), longest first per detector ===")
print(eps.sort_values(["detector", "days"], ascending=[True, False]).groupby("detector").head(6).to_string(index=False), "\n")
yearly_above = mon.assign(year=pd.DatetimeIndex(mon.date).year).groupby("year").apply(
    lambda g: pd.Series({det: float((g[col] > thr.loc["q99", det]).mean() * 100)
                         for det, col in (("input_shape", "shape_28d"), ("weather", "weather_28d"),
                                          ("error_ratio", "error_ratio_28d"))}), include_groups=False)
yearly_above["frozen MLP MAPE"] = [mape(frozen, yr == y) for y in yearly_above.index]
print("=== % of days with the 28-day statistic above q99, by year (frozen MLP) ===")
print(yearly_above.round(1), "\n")
yearly_above.to_csv(os.path.join(OUT, "yearly_above_threshold.csv"))

# --- 3. concept-drift check: in-distribution days only ---------------------------------------
s_q90 = np.quantile(s_daily[REF_THR & eligible], 0.9)
w_q90 = np.quantile(w_daily[REF_THR & eligible], 0.9)
s_input = np.r_[np.nan, s_daily[:-1]]                       # shape of day D feeds the forecast of D+1
in_dist = eligible & (s_input <= s_q90) & (w_daily <= w_q90)
mid = slice(23, 27)
crow = []
for y in range(2014, 2027):
    m = yr == y
    pred = stage1 if y == 2014 else frozen
    mi = m & in_dist
    crow.append({"year": y, "days": int(m.sum()), "in-dist days": int(mi.sum()),
                 "MAPE all days": mape(pred, m), "MAPE in-dist days": mape(pred, mi) if mi.any() else np.nan,
                 "baseline MAPE in-dist": mape(yday, mi) if mi.any() else np.nan,
                 "midday bias in-dist (%)": float(((pred[mi][:, mid] - truth[mi][:, mid]) / truth[mi][:, mid]).mean() * 100)
                 if mi.any() else np.nan})
concept = pd.DataFrame(crow).set_index("year")
concept.to_csv(os.path.join(OUT, "concept_check.csv"))
print("=== Frozen model on days whose inputs stay inside the 2014 range (2014 = Stage 1 model) ===")
print(concept.round(2), "\n")

mm = mon.assign(m=pd.DatetimeIndex(mon.date).to_period("M")).groupby("m")
corr_tab = pd.DataFrame({"shape": mm.shape_daily.mean(), "weather": mm.weather_daily.mean(),
                         "ratio": mm.mlp_frozen_daily_mape.sum() / mm.baseline_daily_mape.sum()})
t = np.arange(len(corr_tab))
resid = corr_tab.apply(lambda c: c - np.polyval(np.polyfit(t, c, 1), t))
corr = pd.DataFrame({"raw Spearman": corr_tab.corr("spearman").ratio,
                     "detrended Pearson": resid.corr().ratio}).drop("ratio")
print("=== Monthly correlation with the frozen MLP/baseline error ratio ===")
print(corr.round(2), "\n")
corr.to_csv(os.path.join(OUT, "input_error_correlation.csv"))

# --- 4. retraining strategies ------------------------------------------------------------------
cache = {}


def retrain(t_end):
    D = d_idx[t_end]
    if D not in cache:
        train = np.asarray((d_idx > D - pd.DateOffset(years=4)) & (d_idx <= D))
        res = Parallel(n_jobs=len(SEEDS))(delayed(M.mlp_fit_predict)(X, R, train, s, EPOCHS, 4) for s in SEEDS)
        cache[D] = np.mean(res, axis=0) * Lr
    return cache[D]


def recalibrate_input(t_end, q):
    D = d_idx[t_end]
    win = np.asarray((d_idx > D - pd.DateOffset(years=4)) & (d_idx <= D))
    fit = win & np.asarray(d_idx <= D - pd.DateOffset(years=1))
    held = win & ~fit
    det = Dt.ShapeNovelty().fit(truth[fit & eligible])
    sc = det.score(truth)
    return sc, float(np.nanquantile(Dt.rolling_eligible(sc[held], eligible[held]), q))


def simulate(kind, q=None, eff_dates=(), cooldown=COOLDOWN):
    current, pred, retr, last = frozen, frozen.copy(), [], T0
    win = Dt.OnlineWindow()
    eff = {pd.Timestamp(x) for x in eff_dates}
    sc = s_daily
    th = (np.nanquantile(r_ref, q) if kind == "error" else np.nanquantile(s_ref, q)) if q else None
    for t in range(T0, n):
        pred[t] = current[t]
        if kind == "periodic":
            trigger = t + 1 < n and d_idx[t + 1] in eff
        elif kind == "frozen":
            trigger = False
        else:
            if eligible[t]:
                if kind == "error":
                    win.push(float(np.mean(np.abs(current[t] - truth[t]) / truth[t]) * 100), b[t])
                else:
                    win.push(sc[t])
            stat = (win.ratio() if kind == "error" else win.mean()) if win.full else -np.inf
            trigger = stat > th and (t + 1 - last) >= cooldown
        if trigger and len(retr) < BUDGET and t + 1 < n:
            current = retrain(t)
            retr.append(d_idx[t + 1].date())
            last = t + 1
            win.clear()
            if kind == "input":
                sc, th = recalibrate_input(t, q)
    return pred, retr


def even_dates(k):
    s, e = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-11-01")
    return [(s + (e - s) * j / (k + 1)).normalize() for j in range(1, k + 1)]


plans = [("frozen", "frozen", None, ())]
plans += [(f"periodic, {k} evenly spaced", "periodic", None, even_dates(k)) for k in range(1, 7)]
plans += [("periodic, 1 Jan 2017/19/21/23/25 (teammate)", "periodic", None,
           [pd.Timestamp(f"{y}-01-01") for y in (2017, 2019, 2021, 2023, 2025)])]
plans += [(f"error-triggered {k}", "error", q, ()) for k, q in QS.items()]
plans += [(f"input-triggered {k}", "input", q, ()) for k, q in QS.items() if k != "max"]

srows, smonthly = [], {}
for name, kind, q, eff in plans:
    t0 = time.time()
    pred, retr = simulate(kind, q, eff)
    srows.append({"strategy": name, "retrains": len(retr), "dates": " ".join(str(x) for x in retr),
                  "MAPE 2015-24": mape(pred, P1), "MAPE 2025-26": mape(pred, P2), "MAPE all": mape(pred, SIM)})
    smonthly[name] = [mape(pred, np.asarray(month == m)) for m in sorted(set(month[SIM]))]
    print(f"{name:<46} retrains={len(retr)} all={srows[-1]['MAPE all']:.3f}  {time.time() - t0:.0f}s", flush=True)
strat = pd.DataFrame(srows).set_index("strategy")
strat.to_csv(os.path.join(OUT, "strategies.csv"))
pd.DataFrame(smonthly, index=[str(m) for m in sorted(set(month[SIM]))]).to_csv(os.path.join(OUT, "strategies_monthly.csv"))
print("\n=== Retraining strategies for the MLP (budget 6, 4-year windows) ===")
print(strat.round(3).to_string())
thr.to_csv(os.path.join(OUT, "thresholds.csv"))
print(f"\nsaved to {OUT}")
