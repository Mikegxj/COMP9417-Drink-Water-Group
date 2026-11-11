"""Annual MAPE of the four main demand models, frozen vs periodic retraining (shared protocol).

All four from our own runs on the same days and truth: MLP, Ridge and LightGBM (exact reproduction of
the teammate's configuration) from outputs/protocol/predictions.npz (run_protocol.py); N-BEATSx
(teammate's configuration, mean of seeds 42/43/44) from predictions_nbeats.npz (run_nbeats.py).
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import features as F  # noqa: E402

OUT = os.path.join(F.OUT_ROOT, "protocol")
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
SERIES = [("MLP (selected)", "#2a78d6"), ("Ridge + splines", "#eb6834"),
          ("LightGBM", "#1baf7a"), ("N-BEATSx", "#eda100")]           # reference palette slots 1-4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

z = dict(np.load(os.path.join(OUT, "predictions.npz")))
z.update(np.load(os.path.join(OUT, "predictions_nbeats.npz")))
days = pd.DatetimeIndex(z["days"])
truth = z["truth"]
yr = days.year.to_numpy()
years = list(range(2015, 2027))


def strategy(model, kind):
    pred = z[f"{model}|frozen"].copy()
    if kind == "periodic":
        for y0 in (2017, 2019, 2021, 2023, 2025):
            pred[yr >= y0] = z[f"{model}|{y0}"][yr >= y0]
    return pred


def annual(pred):
    return [float(np.mean(np.abs(pred[yr == y] - truth[yr == y]) / truth[yr == y]) * 100) for y in years]


yday = np.vstack([np.full((1, 48), np.nan), truth[:-1]])
lweek = np.vstack([np.full((7, 48), np.nan), truth[:-7]])        # same half-hour last week (day D-6)
data = {}
for kind in ("frozen", "periodic"):
    data[kind] = pd.DataFrame({"MLP (selected)": annual(strategy("mlp", kind)),
                               "Ridge + splines": annual(strategy("ridge_vicqld", kind)),
                               "LightGBM": annual(strategy("lgbm", kind)), "N-BEATSx": annual(strategy("nbeats", kind)),
                               "yesterday baseline": annual(yday), "last-week baseline": annual(lweek)}, index=years)
pd.concat(data, axis=1).round(3).to_csv(os.path.join(OUT, "annual_four_models.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True, gridspec_kw={"wspace": 0.28})
fig.patch.set_facecolor(SURFACE)
titles = {"frozen": "Frozen (trained 2010–2014, never updated)",
          "periodic": "Periodic retraining (4-year window, 1 Jan 2017/19/21/23/25)"}
for ax, kind in zip(axes, ("frozen", "periodic")):
    d = data[kind]
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot(years, d["last-week baseline"], color=REF, lw=1.1, ls=(0, (1, 1.5)))
    ax.plot(years, d["yesterday baseline"], color=REF, lw=1.2, ls=(0, (4, 2)))
    for name, c in SERIES:
        ax.plot(years, d[name], "-o", color=c, lw=1.7, ms=4.5, mec=SURFACE, mew=0.8)
    # direct labels at the line ends, nudged apart so they do not collide
    ends = sorted([(d[name].iloc[-1], name, c) for name, c in SERIES] +
                  [(d[b].iloc[-1], b, REF) for b in ("yesterday baseline", "last-week baseline")])
    placed = []
    for v, name, c in ends:
        y = v if not placed else max(v, placed[-1] + 0.48)
        placed.append(y)
        ax.annotate(name, (years[-1], v), xytext=(years[-1] + 0.35, y), textcoords="data",
                    va="center", fontsize=8.3, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=REF, lw=0.9, shrinkA=0, shrinkB=3) if abs(y - v) > 0.05 else None)
    ax.set_title(titles[kind], loc="left", fontsize=10.2)
    ax.set_xticks(years[::2])
    ax.set_xlim(2014.6, 2029.6)
    ax.set_xlabel("Year (2026 = Jan–Oct)")
axes[0].set_ylabel("Annual MAPE (%)")
fig.suptitle("NSW day-ahead demand: four main models under the same protocol", x=0.01, y=1.0, ha="left",
             fontsize=11.5, fontweight="bold")
path = os.path.join(OUT, "fig_four_models_annual.png")
fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
print("saved", path)
