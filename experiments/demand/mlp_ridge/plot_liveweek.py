"""Live-week evidence figure from outputs/liveweek/weeks.csv (run_liveweek.py).

Left: weekly MAPE of December weeks for the MLP trained up to 31 Oct of the same year, and the
same-half-hour-yesterday baseline; the analogue week (Monday 7-13 Dec) is filled, other weeks
hollow; linear trend of the analogue weeks extrapolated to 2026 with +/- 2 residual s.d.
Right: improvement I = 100 (1 - MAPE_model / MAPE_baseline) of the same weeks.
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import features as F  # noqa: E402

OUT = os.path.join(F.OUT_ROOT, "liveweek")
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE = "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

wk = pd.read_csv(os.path.join(OUT, "weeks.csv"))
an = wk["analogue (Mon 7-13 Dec)"].astype(bool)
a = wk[an]
coef = np.polyfit(a.year, a["MLP MAPE"], 1)
band = 2 * (a["MLP MAPE"] - np.polyval(coef, a.year)).std(ddof=2)
jit = np.where(an, 0.0, np.where(wk["week from"].str[-2:].astype(int) < 7, -0.18, 0.18))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.22, "width_ratios": [1.35, 1]})
fig.patch.set_facecolor(SURFACE)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(range(2015, 2027))
    ax.set_xticklabels([str(y)[2:] if y < 2026 else "2026\nlive" for y in range(2015, 2027)])

x = wk.year + jit
ax1.scatter(x[~an], wk["baseline MAPE"][~an], s=26, facecolors="none", edgecolors=REF, lw=1.1)
ax1.scatter(x[an], wk["baseline MAPE"][an], s=30, color=REF, label="same half-hour yesterday")
ax1.scatter(x[~an], wk["MLP MAPE"][~an], s=26, facecolors="none", edgecolors=BLUE, lw=1.1)
ax1.scatter(x[an], wk["MLP MAPE"][an], s=30, color=BLUE, label="MLP, trained to 31 Oct")
yy = np.arange(2015, 2027)
ax1.plot(yy, np.polyval(coef, yy), color=BLUE, lw=1.2, ls=(0, (4, 2)))
p26 = np.polyval(coef, 2026)
ax1.errorbar(2026, p26, yerr=band, fmt="D", ms=6, color=BLUE, capsize=4, lw=1.4)
ax1.annotate(f"analogue-week trend {coef[0]:+.2f} pp/yr\n2026 extrapolation {p26:.1f} ± {band:.1f}%",
             (2026, p26 - band), xytext=(2025.9, 0.45), ha="right", va="bottom", fontsize=8.5, color=BLUE,
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
ax1.set_ylabel("weekly MAPE (%)")
ax1.set_title("Weekly MAPE in December, model trained up to 31 Oct", loc="left", fontsize=9.5)
h = [Line2D([], [], ls="", marker="o", color=BLUE, label="MLP"),
     Line2D([], [], ls="", marker="o", color=REF, label="same half-hour yesterday"),
     Line2D([], [], ls="", marker="o", color=INK2, label="filled: week from Mon 7–13 Dec (analogue)"),
     Line2D([], [], ls="", marker="o", mfc="none", mec=INK2, label="hollow: other Dec week ending by 21 Dec")]
ax1.legend(handles=h, frameon=False, fontsize=8.3, loc="upper left", ncol=2)
ax1.set_ylim(0, 13.5)

ax2.scatter(x[~an], wk["I (%)"][~an], s=26, facecolors="none", edgecolors=BLUE, lw=1.1)
ax2.scatter(x[an], wk["I (%)"][an], s=30, color=BLUE)
ax2.axhline(0, color=INK2, lw=0.8)
ax2.set_ylim(0, 100)
ax2.set_xlim(2014.5, 2025.5)
ax2.set_xticks(range(2015, 2026))
ax2.set_xticklabels([str(y)[2:] for y in range(2015, 2026)])
ax2.set_ylabel("I = 100 (1 − MAPE model / MAPE baseline)")
ax2.set_title("Improvement over the yesterday baseline, same weeks", loc="left", fontsize=9.5)
ax1.set_xlabel("year (20xx)")
ax2.set_xlabel("year (20xx)")
path = os.path.join(OUT, "fig_liveweek.png")
fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
print("saved", path)
