"""Predicted vs observed NSW demand for selected weeks (spec: Model Assessment).

Weeks (Monday-Sunday):
  A  1-7 Jun 2015  winter, frozen model five months after training (frozen = periodic here)
  B  15-21 Apr 2019  Easter: Good Friday and Easter Monday are public holidays
  C  8-14 Dec 2025  closest analogue of the live week: periodic MLP (retrained Jan 2025) vs
                    the frozen MLP (trained 2010-2014) and the yesterday baseline
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import features as F  # noqa: E402

OUT = os.path.join(F.OUT_ROOT, "protocol")
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

z = np.load(os.path.join(OUT, "predictions.npz"))
days = pd.DatetimeIndex(z["days"])
yr = days.year.to_numpy()
truth = z["truth"]


def periodic(model):
    pred = z[f"{model}|frozen"].copy()
    for y0 in (2017, 2019, 2021, 2023, 2025):
        pred[yr >= y0] = z[f"{model}|{y0}"][yr >= y0]
    return pred


per, fro = periodic("mlp"), z["mlp|frozen"]
yday = np.vstack([np.full((1, 48), np.nan), truth[:-1]])
WEEKS = [("A", "2015-06-01", "Winter week, Jun 2015 (model 5 months old)"),
         ("B", "2019-04-15", "Easter week, Apr 2019 (Good Friday, Easter Monday)"),
         ("C", "2025-12-08", "8–14 Dec 2025, closest analogue of the live week")]
fig, axes = plt.subplots(3, 1, figsize=(12.5, 10), gridspec_kw={"hspace": 0.45})
fig.patch.set_facecolor(SURFACE)
for ax, (tag, start, title) in zip(axes, WEEKS):
    m = np.asarray((days >= start) & (days <= pd.Timestamp(start) + pd.Timedelta(days=6)))
    t = pd.date_range(pd.Timestamp(start) + pd.Timedelta(minutes=30), periods=m.sum() * 48, freq="30min")
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot(t, truth[m].ravel(), color=INK, lw=1.6, label="observed")
    ax.plot(t, yday[m].ravel(), color=REF, lw=1.1, ls=(0, (4, 2)), label="same half-hour yesterday")
    ax.plot(t, per[m].ravel(), color=BLUE, lw=1.4, label="MLP, periodic retraining")
    mp = lambda p: np.mean(np.abs(p[m] - truth[m]) / truth[m]) * 100  # noqa: E731
    txt = f"MAPE: MLP {mp(per):.1f}%, baseline {mp(yday):.1f}%"
    if tag == "C":
        ax.plot(t, fro[m].ravel(), color=ORANGE, lw=1.3, label="MLP, frozen (trained 2010–2014)")
        txt += f", frozen MLP {mp(fro):.1f}%"
    ax.set_title(f"{tag}. {title}   —   {txt}", loc="left", fontsize=10)
    ax.set_ylabel("NSW demand (MW)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.legend(frameon=False, fontsize=8.3, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
path = os.path.join(OUT, "fig_pred_vs_obs.png")
fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
print("saved", path)
