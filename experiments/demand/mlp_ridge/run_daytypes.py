"""Day-type analysis of NSW demand (spec: Day-Type Analysis).

Each market day is its 48 half-hourly NSW demand values. PCA (components explaining 95% of the
variance) and k-means are fitted on 2010-2013 only and then kept fixed; every later day is
projected with the same PCA and assigned to the nearest centre.

Representation: shape = profile / its daily mean (level removed) vs raw MW. Both are clustered at
the same k and compared on how well clusters line up with day type (workday vs weekend/holiday)
and season, and how much the cluster mix of later years is driven by the falling demand level.
k: scanned 2..10 on 2010-2013 (silhouette, k-means inertia, Gaussian-mixture BIC); chosen with
--k (default 4, see the printed scan for the justification).
Outputs: outputs/daytypes/*.csv and figures.
Usage: python run_daytypes.py [--k K]
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

import features as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=4)
K = ap.parse_args().k
OUT = os.path.join(F.OUT_ROOT, "daytypes")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

df = F.load()
days, arr = F.to_daily(df)
cal = F.calendar(days)
Y = arr["nsw_demand"]
yr = days.year.to_numpy()
TRAIN = (yr >= 2010) & (yr <= 2013)
shape = Y / Y.mean(axis=1, keepdims=True)
workday = cal["workday"].to_numpy().astype(int)
season = ((days.month.to_numpy() % 12) // 3)                    # 0 summer, 1 autumn, 2 winter, 3 spring


def fit_space(V):
    pca = PCA(n_components=0.95, svd_solver="full").fit(V[TRAIN])
    return pca, pca.transform(V)


# --- k scan on the training years ----------------------------------------------------------------
pca_s, Zs = fit_space(shape)
scan = []
for k in range(2, 11):
    km = KMeans(k, n_init=20, random_state=0).fit(Zs[TRAIN])
    gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=0).fit(Zs[TRAIN])
    scan.append({"k": k, "silhouette": silhouette_score(Zs[TRAIN], km.labels_),
                 "inertia": km.inertia_, "GMM BIC": gm.bic(Zs[TRAIN]),
                 "ARI vs workday": adjusted_rand_score(workday[TRAIN], km.labels_),
                 "ARI vs season": adjusted_rand_score(season[TRAIN], km.labels_)})
scan = pd.DataFrame(scan).set_index("k")
print(f"shape PCA: {pca_s.n_components_} components for 95% variance (fitted 2010-2013)")
print("=== k scan on 2010-2013 (normalised profiles) ===")
print(scan.round(3), "\n")
scan.to_csv(os.path.join(OUT, "k_scan.csv"))

# --- raw vs normalised at the chosen k --------------------------------------------------------------
pca_r, Zr = fit_space(Y)
km_s = KMeans(K, n_init=20, random_state=0).fit(Zs[TRAIN])
km_r = KMeans(K, n_init=20, random_state=0).fit(Zr[TRAIN])
lab_s, lab_r = km_s.predict(Zs), km_r.predict(Zr)
cmp_rows = []
for name, lab in (("normalised (shape)", lab_s), ("raw MW", lab_r)):
    late = yr >= 2023
    share_late = np.bincount(lab[late], minlength=K) / late.sum()
    cmp_rows.append({"representation": name,
                     "ARI vs workday (2010-13)": adjusted_rand_score(workday[TRAIN], lab[TRAIN]),
                     "ARI vs season (2010-13)": adjusted_rand_score(season[TRAIN], lab[TRAIN]),
                     "largest cluster share 2023-26": share_late.max()})
cmp = pd.DataFrame(cmp_rows).set_index("representation")
print(f"=== Representation comparison at k={K} ===")
print(cmp.round(3), "\n")
cmp.to_csv(os.path.join(OUT, "representation_comparison.csv"))

# --- interpret the normalised clusters ----------------------------------------------------------------
lab = lab_s
order = np.argsort([-np.mean(workday[TRAIN][lab[TRAIN] == c]) for c in range(K)])   # workday-heavy first
rename = {old: new for new, old in enumerate(order)}
lab = np.vectorize(rename.get)(lab)
centres = km_s.cluster_centers_[order]
dist_all = np.min(np.linalg.norm(Zs[:, None, :] - centres[None], axis=2), axis=1)
# reconstruction residual (what PCA cannot represent) adds to the distance from the training space
resid = np.linalg.norm(shape - pca_s.inverse_transform(Zs), axis=1)
novelty = np.sqrt(dist_all ** 2 + resid ** 2)
q95 = np.quantile(novelty[TRAIN], 0.95)

comp = []
months = days.month.to_numpy()
for c in range(K):
    m = TRAIN & (lab == c)
    comp.append({"cluster": c, "days 2010-13": int(m.sum()),
                 "workday %": 100 * workday[m].mean(), "Sat %": 100 * (days.dayofweek.to_numpy()[m] == 5).mean(),
                 "Sun/holiday %": 100 * ((days.dayofweek.to_numpy()[m] == 6) | (cal['hol'].to_numpy()[m] == 1)).mean(),
                 "summer (DJF) %": 100 * np.isin(months[m], [12, 1, 2]).mean(),
                 "winter (JJA) %": 100 * np.isin(months[m], [6, 7, 8]).mean(),
                 "mean demand MW": Y[m].mean(), "peak half-hour": int(np.argmax(shape[m].mean(0))),
                 "min/max of mean shape": shape[m].mean(0).min() / shape[m].mean(0).max()})
comp = pd.DataFrame(comp).set_index("cluster")
print("=== Clusters (normalised, fitted 2010-2013) ===")
print(comp.round(2), "\n")
comp.to_csv(os.path.join(OUT, "cluster_composition.csv"))
profiles = pd.DataFrame({f"cluster {c}": shape[TRAIN & (lab == c)].mean(0) for c in range(K)})
profiles.to_csv(os.path.join(OUT, "cluster_profiles.csv"), index_label="half_hour")

# --- yearly shares, novelty and forecast error -------------------------------------------------------
z = np.load(os.path.join(F.OUT_ROOT, "protocol", "predictions.npz"))
pdays = pd.DatetimeIndex(z["days"])
pos = np.searchsorted(days, pdays)
frozen = z["mlp|frozen"]
truth = z["truth"]
ape_day = np.full(len(days), np.nan)
ape_day[pos] = np.mean(np.abs(frozen - truth) / truth, axis=1) * 100
rows = []
for y_ in range(2010, 2027):
    m = yr == y_
    r = {"year": y_, **{f"share c{c} %": 100 * np.mean(lab[m] == c) for c in range(K)},
         "mean novelty": novelty[m].mean(), "% days above 2010-13 q95": 100 * np.mean(novelty[m] > q95),
         "frozen MLP MAPE": np.nanmean(ape_day[m]) if y_ >= 2015 else np.nan}
    for c in range(K):
        mc = m & (lab == c)
        r[f"MAPE c{c}"] = np.nanmean(ape_day[mc]) if (y_ >= 2015 and mc.any()) else np.nan
    rows.append(r)
yearly = pd.DataFrame(rows).set_index("year")
print("=== Yearly day-type shares, novelty and frozen-MLP error ===")
print(yearly.round(2), "\n")
yearly.to_csv(os.path.join(OUT, "yearly_shares_error.csv"))
sim = yearly.loc[2015:2026]
print("Spearman correlation with frozen MLP MAPE (2015-2026, yearly):")
print(sim.drop(columns=[c for c in sim.columns if c.startswith("MAPE")]).corr("spearman")["frozen MLP MAPE"].round(2), "\n")
daily = pd.DataFrame({"date": days.date, "cluster": lab, "novelty": novelty, "frozen_mlp_daily_mape": ape_day})
daily.to_csv(os.path.join(OUT, "daily_daytypes.csv"), index=False)

# --- figures -------------------------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
names = [f"C{c}: {comp.loc[c, 'workday %']:.0f}% workdays, {comp.loc[c, 'summer (DJF) %']:.0f}% summer, "
         f"{comp.loc[c, 'winter (JJA) %']:.0f}% winter" for c in range(K)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.25})
fig.patch.set_facecolor(SURFACE)
for ax in (a1, a2):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
hours = (np.arange(48) + 0.5) / 2
for c in range(K):
    a1.plot(hours, profiles[f"cluster {c}"], color=PAL[c], lw=1.8, label=names[c])
a1.set_xlabel("Time of day (hour, interval start)")
a1.set_ylabel("Demand / daily mean")
a1.set_xticks(range(0, 25, 3))
a1.set_title(f"Typical daily shapes, k = {K} (PCA + k-means fitted on 2010–2013)", loc="left", fontsize=10.2)
a1.legend(frameon=False, fontsize=7.8, loc="lower right")
bottom = np.zeros(len(yearly))
years = yearly.index.to_numpy()
for c in range(K):
    v = yearly[f"share c{c} %"].to_numpy()
    a2.bar(years, v, bottom=bottom, color=PAL[c], width=0.8, edgecolor=SURFACE, lw=1.2, label=f"C{c}")
    bottom += v
a2.set_ylabel("Share of days (%)")
a2.set_xticks(years[::2])
a2.set_title("Share of each day type by year (2026 = Jan–Oct); line: % of days unlike any 2010–13 type",
             loc="left", fontsize=10.2)
a2.plot(years, yearly["% days above 2010-13 q95"], color=INK, lw=1.6, marker="o", ms=4, label="% novel days (> q95)")
a2.legend(frameon=False, fontsize=8, ncol=K + 1, loc="upper center", bbox_to_anchor=(0.5, -0.08))
a2.set_ylim(0, 100)
path = os.path.join(OUT, "fig_daytypes.png")
fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
print("saved", path)
