"""Shared feature construction for the NSW day-ahead demand task.

Everything is built on a "daily" layout: one row per market day, 48 columns (half-hours).
For a target day i (= D+1), demand features only use days <= i-1 (= D); weather and calendar
features may use day i itself, as allowed by the spec.
"""
import os
import sys

import holidays
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
if os.path.exists(os.path.join(_REPO, "harness", "dayahead.py")):
    # group repository: this folder is experiments/demand/mlp_ridge/
    HARNESS = os.path.join(_REPO, "harness")
    DATA = os.path.join(_REPO, "data", "nem_weather_halfhourly_2010_2026.csv")
    OUT_ROOT = os.path.join(_REPO, "results", "demand", "mlp_ridge")
    PACKAGE = os.path.join(_REPO, "submission_mlp")
else:
    # local project: project/experiments/demand/ next to project/comp9417_project_kit/
    _KIT = os.path.normpath(os.path.join(HERE, "..", "..", "comp9417_project_kit"))
    HARNESS = os.path.join(_KIT, "harness")
    DATA = os.path.join(_KIT, "data", "nem_weather_halfhourly_2010_2026.csv")
    OUT_ROOT = os.path.join(HERE, "outputs")
    PACKAGE = os.path.normpath(os.path.join(HERE, "..", "..", "package_mlp"))
sys.path.insert(0, HARNESS)
import dayahead as da  # noqa: E402

WX = ["syd_temp", "syd_rh", "syd_cloud", "syd_solar"]
P = 48


def load():
    return da.load_dataset(DATA)


def to_daily(df):
    """Reshape the half-hourly frame to (n_days, 48) arrays, one per column."""
    day = da.market_day(df.index)
    days = pd.DatetimeIndex(day.unique())
    n = len(days)
    assert len(df) == n * P, "expected complete days only"
    assert (np.asarray(day).reshape(n, P) == np.asarray(days)[:, None]).all()
    arr = {c: df[c].to_numpy(float).reshape(n, P) for c in da.MARKET_COLS + da.WEATHER_COLS}
    return days, arr


def calendar(days):
    years = range(days.min().year - 1, days.max().year + 2)
    hol = holidays.country_holidays("AU", subdiv="NSW", years=years)
    is_hol = np.array([d.date() in hol for d in days], dtype=float)
    dow = days.dayofweek.to_numpy()
    m, dd = days.month.to_numpy(), days.day.to_numpy()
    xmas = (((m == 12) & (dd >= 22)) | ((m == 1) & (dd <= 4))).astype(float)
    doy = days.dayofyear.to_numpy() / 365.25
    return pd.DataFrame({
        "dow": dow, "hol": is_hol, "xmas": xmas,
        "workday": ((dow < 5) & (is_hol == 0)).astype(float),
        "s1": np.sin(2 * np.pi * doy), "c1": np.cos(2 * np.pi * doy),
        "s2": np.sin(4 * np.pi * doy), "c2": np.cos(4 * np.pi * doy),
    })


def level(arr, normalize=True, const=None, col="nsw_demand"):
    """Scale L[i] used to normalise demand: mean demand of the 7 days before day i
    (ratio mode), or a constant (raw mode). Uses only days <= i-1."""
    Y = arr[col]
    n = len(Y)
    if not normalize:
        return np.full(n, float(const))
    cs = np.concatenate([[0.0], np.cumsum(Y.mean(1))])
    L = np.full(n, np.nan)
    L[7:] = (cs[7:n] - cs[0:n - 7]) / 7.0
    return L


def _weather_stats(arr):
    T = arr["syd_temp"]
    tmean = T.mean(1)
    t3 = np.full(len(T), np.nan)
    t3[2:] = (tmean[2:] + tmean[1:-1] + tmean[:-2]) / 3.0   # days i-2..i (weather, allowed)
    return {"tmax": T.max(1), "tmin": T.min(1), "tmean": tmean, "t3": t3,
            "solar_mean": arr["syd_solar"].mean(1)}


def ridge_frame(arr, cal, L, idx, p, opts=None):
    """Design matrix for the per-half-hour linear model at period p, target days idx.

    opts (all optional, default = the full Sydney-only feature set):
      cities: extra weather cities, e.g. ["mel", "bne"] (temperature of D+1 and its daily max)
      regions: other states' demand, e.g. ["vic", "qld"] (day D, scaled by their own 7-day mean)
      no_solar_cloud: drop cloud and solar features
      no_holidays: drop the NSW public-holiday flags (weekday/weekend logic only)
      no_xmas: drop the Christmas-New Year period flag
      minimal: a lean set like the teammate's 11 features (lags 1/2/7, temperature of the
               half-hour on D+1 and D, D's min/max temperature, weekday, season)
    """
    opts = opts or {}
    if opts.get("minimal"):
        return _ridge_frame_minimal(arr, cal, L, idx, p)
    Y, T = arr["nsw_demand"], arr["syd_temp"]
    rh, cl, so = arr["syd_rh"], arr["syd_cloud"], arr["syd_solar"]
    ws = _weather_stats(arr)
    i, y1, y7 = idx, idx - 1, idx - 7
    r1, r7, rlast = Y[y1, p] / L[i], Y[y7, p] / L[i], Y[y1, P - 1] / L[i]
    dow = cal["dow"].to_numpy()[i]
    hol, wd = cal["hol"].to_numpy(), cal["workday"].to_numpy()
    f = {"r1": r1, "r7": r7, "rlast": rlast}
    for k in range(7):
        oh = (dow == k).astype(float)
        f[f"dow{k}"] = oh
        f[f"r1_dow{k}"] = r1 * oh
        f[f"r7_dow{k}"] = r7 * oh
    f.update({
        "hol_t": hol[i], "hol_y": hol[y1], "hol_7": hol[y7],
        "r1_hol_t": r1 * hol[i], "r1_hol_y": r1 * hol[y1],
        "r7_hol_t": r7 * hol[i], "r7_hol_7": r7 * hol[y7],
        "r1_wdchange": r1 * (wd[i] != wd[y1]),
        "T": T[i, p], "Tmax": ws["tmax"][i], "T3": ws["t3"][i],
        "dT": T[i, p] - T[y1, p], "dTmax": ws["tmax"][i] - ws["tmax"][y1],
        "hum_heat": rh[i, p] * np.maximum(T[i, p] - 22.0, 0.0),
        "cloud": cl[i, p], "solar": so[i, p], "dsolar": so[i, p] - so[y1, p],
        "solar_mean": ws["solar_mean"][i],
        "xmas": cal["xmas"].to_numpy()[i],
    })
    for c in ("s1", "c1", "s2", "c2"):
        f[c] = cal[c].to_numpy()[i]
    if opts.get("no_solar_cloud"):
        for k in ("cloud", "solar", "dsolar", "solar_mean"):
            f.pop(k)
    if opts.get("no_holidays"):
        for k in ("hol_t", "hol_y", "hol_7", "r1_hol_t", "r1_hol_y", "r7_hol_t", "r7_hol_7"):
            f.pop(k)
        wd_dow = (cal["dow"].to_numpy() < 5).astype(float)    # weekday/weekend only
        f["r1_wdchange"] = r1 * (wd_dow[i] != wd_dow[y1])
    if opts.get("no_xmas"):
        f.pop("xmas")
    for city in opts.get("cities", ()):
        Tc = arr[f"{city}_temp"]
        f[f"{city}_T"] = Tc[i, p]
        f[f"{city}_Tmax"] = Tc.max(1)[i]
    for reg in opts.get("regions", ()):
        Dg, Lg = arr[f"{reg}_demand"], level(arr, col=f"{reg}_demand")
        f[f"{reg}_r1"] = Dg[y1, p] / Lg[i]
        f[f"{reg}_daylevel"] = Dg[y1].mean(1) / Lg[i]      # day D relative to its own week
    return pd.DataFrame(f)


def _ridge_frame_minimal(arr, cal, L, idx, p):
    Y, T = arr["nsw_demand"], arr["syd_temp"]
    i = idx
    f = {"r1": Y[i - 1, p] / L[i], "r2": Y[i - 2, p] / L[i], "r7": Y[i - 7, p] / L[i],
         "T": T[i, p], "Tprior": T[i - 1, p],
         "Tprior_min": T.min(1)[i - 1], "Tprior_max": T.max(1)[i - 1]}
    dow = cal["dow"].to_numpy()[i]
    for k in range(7):
        f[f"dow{k}"] = (dow == k).astype(float)
    f["s1"], f["c1"] = cal["s1"].to_numpy()[i], cal["c1"].to_numpy()[i]
    return pd.DataFrame(f)


def lgbm_frame(arr, cal, days, idx):
    """Long-format inputs (one row per target half-hour, day-major) reproducing the teammate's
    LightGBM: 'cloud' variant with holiday level 2, same column order as src/demand/tabular.py."""
    Y, T, cl = arr["nsw_demand"], arr["syd_temp"], arr["syd_cloud"]
    i = np.repeat(idx, P)
    p = np.tile(np.arange(P), len(idx))
    hol = cal["hol"].to_numpy()
    angle = 2 * np.pi * (days.dayofyear.to_numpy()[i] - 1) / 365.25
    dow = cal["dow"].to_numpy()[i]
    f = {"slot": p.astype(float), "season_sin": np.sin(angle), "season_cos": np.cos(angle),
         "temp": T[i, p], "prior_temp": T[i - 1, p],
         "lag1": Y[i - 1, p], "lag2": Y[i - 2, p], "lag7": Y[i - 7, p], "cloud": cl[i, p],
         "hol_today": hol[i], "hol_lag1": hol[i - 1], "hol_lag7": hol[i - 7]}
    for k in range(7):
        f[f"weekday_{k}"] = (dow == k).astype(float)
    return pd.DataFrame(f)


_TREE_DROP = ("r1_dow", "r7_dow", "r1_hol", "r7_hol", "r1_wdchange", "dow")


def lgbm_rich_frame(arr, cal, L, idx, opts=None):
    """Long-format (day-major) version of the Ridge/MLP information set for tree models: the
    per-half-hour features of ridge_frame (lags scaled by the 7-day level, weather incl. solar,
    holidays, Christmas flag, season) without the hand-made interactions, plus slot and weekday.
    Pair it with the ratio target R = demand / L."""
    frames = []
    for p in range(P):
        fr = ridge_frame(arr, cal, L, idx, p, opts)
        fr = fr[[c for c in fr.columns if not c.startswith(_TREE_DROP)]]
        fr.insert(0, "slot", float(p))
        fr.insert(1, "weekday", cal["dow"].to_numpy()[idx].astype(float))
        frames.append(fr)
    long = pd.concat(frames, ignore_index=True)            # period-major
    order = np.arange(P * len(idx)).reshape(P, len(idx)).T.ravel()
    return long.iloc[order].reset_index(drop=True)          # day-major, like Y[idx].ravel()


# columns that get a spline basis in the Ridge pipeline (whichever are present)
SPLINE_COLS = ["T", "Tmax", "T3", "Tprior", "Tprior_min", "Tprior_max",
               "mel_T", "mel_Tmax", "bne_T", "bne_Tmax"]


def mlp_matrix(arr, cal, L, idx, opts=None):
    """Wide input matrix for the MLP: one row per target day. opts as in ridge_frame."""
    opts = opts or {}
    Y, T = arr["nsw_demand"], arr["syd_temp"]
    i, y1, y7 = idx, idx - 1, idx - 7
    dow = cal["dow"].to_numpy()[i]
    if opts.get("minimal"):
        return np.concatenate([Y[y1] / L[i, None], Y[i - 2] / L[i, None], Y[y7] / L[i, None],
                               T[i], T[y1], np.eye(7)[dow], cal[["s1", "c1"]].to_numpy()[i],
                               T.min(1)[y1, None], T.max(1)[y1, None]], axis=1)
    ws = _weather_stats(arr)
    blocks = [Y[y1] / L[i, None], Y[y7] / L[i, None], T[i], T[y1], arr["syd_rh"][i]]
    if not opts.get("no_solar_cloud"):
        blocks += [arr["syd_cloud"][i], arr["syd_solar"][i], arr["syd_solar"][y1]]
    hol = cal["hol"].to_numpy()
    scal = [np.eye(7)[dow]]
    if not opts.get("no_holidays"):
        scal += [hol[i, None], hol[y1, None], hol[y7, None]]
    cal_cols = ["s1", "c1", "s2", "c2"] if opts.get("no_xmas") else ["xmas", "s1", "c1", "s2", "c2"]
    scal.append(cal[cal_cols].to_numpy()[i])
    stats = np.stack([ws["tmax"][i], ws["tmin"][i], ws["t3"][i]], 1)
    extra = [arr[f"{city}_temp"][i] for city in opts.get("cities", ())]
    for reg in opts.get("regions", ()):
        Lg = level(arr, col=f"{reg}_demand")
        extra.append(arr[f"{reg}_demand"][y1] / Lg[i, None])
    return np.concatenate(blocks + scal + [stats] + extra, axis=1)
