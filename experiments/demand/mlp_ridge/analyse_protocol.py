"""Compare single models and ensembles under the shared protocol (reads outputs/protocol/predictions.npz
from run_protocol.py and the teammate's exp2 tables copied next to it).

Ensemble weights are chosen once, on the 2014 validation predictions of the Stage 1 fits
(2010-2013 training), by minimising 2014 MAPE on a 0.05 grid over the simplex. They are then
fixed for the whole 2015-2026 simulation, so no simulation result is used to set them.
Paired day-block bootstrap intervals (days resampled, the 48 half-hours of a day kept together)
are given for MAPE differences; they treat days as independent, so they are optimistic.
"""
import itertools
import os

import numpy as np
import pandas as pd

import features as F

OUT = os.path.join(F.OUT_ROOT, "protocol")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

z = np.load(os.path.join(OUT, "predictions.npz"))
days = pd.DatetimeIndex(z["days"])
truth = z["truth"]
yr = days.year.to_numpy()
VAL, SIM = yr == 2014, yr >= 2015
P1, P2 = SIM & (yr <= 2024), yr >= 2025
RETRAINS = [2017, 2019, 2021, 2023, 2025]
SINGLE = ["ridge_vicqld", "lgbm", "mlp", "mlp_seed0", "mlp_nosolar"]
yday = np.vstack([np.full((1, F.P), np.nan), truth[:-1]])        # same half-hour yesterday


def strategy(model, kind):
    if kind == "stage1":
        return z[f"{model}|stage1"]
    pred = z[f"{model}|frozen"].copy()
    if kind == "periodic":
        for y0 in RETRAINS:
            pred[yr >= y0] = z[f"{model}|{y0}"][yr >= y0]
    return pred


def mape(pred, mask):
    return float(np.mean(np.abs(pred[mask] - truth[mask]) / truth[mask]) * 100)


def day_ape(pred):
    return np.mean(np.abs(pred - truth) / truth, axis=1) * 100


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)], [2.5, 97.5])


# --- ensembles: weights fixed on 2014 ------------------------------------------------------
COMBOS = {"mlp+lgbm": ["mlp", "lgbm"], "mlp+ridge": ["mlp", "ridge_vicqld"],
          "mlp+ridge+lgbm": ["mlp", "ridge_vicqld", "lgbm"]}
grid = np.round(np.arange(0, 1.0001, 0.05), 2)
weights, wrows = {}, []
for name, members in COMBOS.items():
    best = None
    for w in itertools.product(grid, repeat=len(members) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        ww = np.array(list(w) + [1 - sum(w)])
        v = mape(sum(wi * strategy(m, "stage1") for wi, m in zip(ww, members)), VAL)
        if best is None or v < best[0]:
            best = (v, ww)
    weights[f"{name} (2014 weights)"] = (members, best[1])
    weights[f"{name} (equal)"] = (members, np.full(len(members), 1 / len(members)))
    wrows.append({"ensemble": name, **{m: round(float(w), 2) for m, w in zip(members, best[1])},
                  "val 2014": best[0]})
wtab = pd.DataFrame(wrows).set_index("ensemble")
print("=== Ensemble weights chosen on 2014 validation ===")
print(wtab.round(3), "\n")


def ens(name, kind):
    members, ww = weights[name]
    return sum(wi * strategy(m, kind) for wi, m in zip(ww, members))


preds = {}
for m in SINGLE:
    preds[m] = {k: strategy(m, k) for k in ("stage1", "frozen", "periodic")}
for name in weights:
    preds[name] = {k: ens(name, k) for k in ("stage1", "frozen", "periodic")}

rows = []
for name, p in preds.items():
    rows.append({"model": name, "val 2014": mape(p["stage1"], VAL),
                 "frozen 2015-24": mape(p["frozen"], P1), "frozen 2025-26": mape(p["frozen"], P2),
                 "frozen all": mape(p["frozen"], SIM),
                 "periodic 2015-24": mape(p["periodic"], P1), "periodic 2025-26": mape(p["periodic"], P2),
                 "periodic all": mape(p["periodic"], SIM)})
rows.append({"model": "baseline yesterday", "val 2014": mape(yday, VAL),
             **{f"{s} {per}": mape(yday, m) for s in ("frozen", "periodic")
                for per, m in (("2015-24", P1), ("2025-26", P2), ("all", SIM))}})

# teammate's exp2 results (their own runs), for side-by-side reading
tm = pd.read_csv(os.path.join(OUT, "teammate_exp2_comparison.csv"), encoding="utf-8")
tm["per"] = tm.period.map(lambda s: "all" if s == "overall" else ("2015-24" if s.startswith("2015") else "2025-26"))
val_tm = {"nbeats": 1.887, "tft": 1.914, "gam": 2.283, "lightgbm": 2.294}   # docs/demand/experiment_log.md §10
for model, g in tm.groupby("model"):
    r = {"model": f"[teammate] {model}", "val 2014": val_tm.get(model, np.nan)}
    for _, x in g.iterrows():
        r[f"{x.strategy} {x.per}"] = x.mape
    rows.append(r)
summary = pd.DataFrame(rows).set_index("model")
print("=== MAPE (%) under the shared protocol ===")
print(summary.round(3), "\n")

# --- reproduction check for LightGBM --------------------------------------------------------
ta = pd.read_csv(os.path.join(OUT, "teammate_exp2_annual.csv"))
chk = []
for kind in ("frozen", "periodic"):
    t = ta[(ta.model == "lightgbm") & (ta.strategy == kind)].set_index("year").mape
    mine = pd.Series({y: mape(preds["lgbm"][kind], yr == y) for y in t.index})
    chk.append(pd.DataFrame({"teammate": t, "reproduction": mine, "diff": mine - t}).assign(strategy=kind))
chk = pd.concat(chk)
print("=== LightGBM reproduction vs teammate (annual MAPE) ===")
print(chk.round(3).to_string(), "\n")

# --- annual, periodic strategy ---------------------------------------------------------------
years = sorted(set(yr[SIM]))
annual = pd.DataFrame({m: [mape(preds[m]["periodic"], yr == y) for y in years]
                       for m in ["ridge_vicqld", "lgbm", "mlp", "mlp_nosolar", "mlp+lgbm (2014 weights)",
                                 "mlp+ridge+lgbm (2014 weights)"]}, index=years)
tp = ta[ta.strategy == "periodic"].pivot(index="year", columns="model", values="mape")
annual = annual.join(tp.add_prefix("[tm] "))
annual["baseline yesterday"] = [mape(yday, yr == y) for y in years]
print("=== Annual MAPE (%), periodic retraining ===")
print(annual.round(2), "\n")

# --- paired differences ----------------------------------------------------------------------
pairs = [("mlp+lgbm (2014 weights)", "mlp"), ("mlp+ridge+lgbm (2014 weights)", "mlp"),
         ("lgbm", "mlp"), ("mlp_nosolar", "mlp"), ("mlp", "mlp_seed0")]
brows = []
for a, b in pairs:
    for per, m in (("2015-26", SIM), ("2025-26", P2)):
        d = (day_ape(preds[a]["periodic"]) - day_ape(preds[b]["periodic"]))[m]
        lo, hi = boot(d)
        brows.append({"A": a, "B": b, "period": per, "MAPE A - B (pp)": d.mean(), "95% low": lo, "95% high": hi})
btab = pd.DataFrame(brows)
print("=== Paired differences, periodic strategy (negative = A better) ===")
print(btab.round(3).to_string(index=False), "\n")

# --- time of day and weekly improvement in the recent period ---------------------------------
SLOTS = {"midday 11:30-13:30": slice(23, 27), "evening 18:30-20:00": slice(37, 40)}


def slot_mape(pred, mask, sl):
    return float(np.mean(np.abs(pred[mask][:, sl] - truth[mask][:, sl]) / truth[mask][:, sl]) * 100)


tod = pd.DataFrame({m: {k: slot_mape(preds[m]["periodic"], P2, sl) for k, sl in SLOTS.items()}
                    for m in ["ridge_vicqld", "lgbm", "mlp", "mlp_nosolar", "mlp+lgbm (2014 weights)"]}).T
tod.loc["baseline yesterday"] = [slot_mape(yday, P2, sl) for sl in SLOTS.values()]
print("=== 2025-26 periodic, by time of day (MAPE %) ===")
print(tod.round(2), "\n")

dow = days.dayofweek.to_numpy()
week = np.asarray(days - pd.to_timedelta(dow, unit="D"))
wk = []
for w in sorted(set(week[P2])):
    m = P2 & (week == w)
    if m.sum() < 7:
        continue
    b = mape(yday, m)
    wk.append({"week": pd.Timestamp(w).date(), "baseline": b,
               **{k: 100 * (1 - mape(preds[k]["periodic"], m) / b)
                  for k in ["lgbm", "mlp", "mlp+lgbm (2014 weights)", "mlp+ridge+lgbm (2014 weights)"]}})
wk = pd.DataFrame(wk).set_index("week")
q = wk.drop(columns="baseline").describe(percentiles=[.1, .5]).T[["10%", "50%", "min"]]
q["share I>=40 (%)"] = (wk.drop(columns="baseline") >= 40).mean() * 100
print(f"=== Weekly improvement I vs yesterday, {len(wk)} Mon-Sun weeks of 2025-26, periodic ===")
print(q.round(1))
print("\nDecember 2025 weeks:")
print(wk[[pd.Timestamp(w).month == 12 for w in wk.index]].round(1))

summary.to_csv(os.path.join(OUT, "summary.csv"))
wtab.to_csv(os.path.join(OUT, "ensemble_weights.csv"))
annual.to_csv(os.path.join(OUT, "annual_periodic.csv"))
btab.to_csv(os.path.join(OUT, "paired_differences.csv"), index=False)
chk.to_csv(os.path.join(OUT, "lgbm_reproduction_check.csv"))
wk.to_csv(os.path.join(OUT, "weekly_improvement.csv"))
print(f"\nsaved tables to {OUT}")
