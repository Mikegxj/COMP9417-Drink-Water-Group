"""Figure for the feature ablation (reads outputs/ablation_features.csv from run_ablation.py).

Left: change in 2014 validation MAPE against the base feature set, with 95% day-block
bootstrap intervals. Right: MAPE of the same models kept frozen, 2023 to Oct 2026, against
the same-half-hour-yesterday baseline.
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import features as F  # noqa: E402
import models as M  # noqa: E402

OUT = F.OUT_ROOT
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Ridge": "#2a78d6", "MLP": "#eb6834"}          # reference palette slots 1-2
ROWS = {
    "base (Sydney only)": "Base: Sydney weather only",
    "+ Mel/Bne weather": "+ Melbourne & Brisbane weather",
    "+ VIC/QLD demand": "+ VIC & QLD demand (day D)",
    "+ Mel/Bne + VIC/QLD": "+ both of the above",
    "- solar & cloud": "− Sydney solar & cloud",
    "minimal (teammate-like)": "Lean set (teammate's 11 features)",
}

t = pd.read_csv(os.path.join(OUT, "ablation_features.csv")).set_index(["model", "features"])

# yesterday baseline over the same frozen window, for the reference line
df = F.load()
days, arr = F.to_daily(df)
Y, idx = arr["nsw_demand"], np.arange(7, len(days))
late = days[idx].year.to_numpy() >= 2023
base_late = M.ratio_mape(Y[idx - 1][late], Y[idx][late])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                     "text.color": INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True,
                               gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.06})
fig.subplots_adjust(left=0.215, right=0.985, top=0.80, bottom=0.17)
fig.patch.set_facecolor(SURFACE)
ypos = -np.arange(len(ROWS))
offset = {"Ridge": 0.14, "MLP": -0.14}

for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for y in ypos[1::2]:                                   # light banding for row reading
        ax.axhspan(y - 0.5, y + 0.5, color="#f3f2ee", zorder=0, lw=0)

# left panel: change vs base with 95% interval
ax1.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)))
for model, c in COLORS.items():
    for y, key in zip(ypos, ROWS):
        r = t.loc[(model, key)]
        if key.startswith("base"):
            continue
        d, lo, hi = r["diff vs base (pp)"], r["95% CI low"], r["95% CI high"]
        ax1.errorbar(d, y + offset[model], xerr=[[d - lo], [hi - d]], fmt="o", color=c,
                     ms=6.5, mec=SURFACE, mew=1.2, elinewidth=1.6, capsize=0, zorder=3)
ax1.text(0.012, ypos[0], "reference (0)", va="center", ha="left", color=INK2, fontsize=8.5)
ax1.set_xlabel("Change in 2014 validation MAPE vs base (percentage points)\n"
               "← better            worse →")
ax1.set_title("Validation 2014: effect of each feature change\n(dot = mean, bar = 95% day-block bootstrap)",
              loc="left", fontsize=10, color=INK)

# right panel: frozen robustness
ax2.axvline(base_late, color=INK2, lw=1, ls=(0, (3, 3)))
ax2.text(base_late - 0.08, ypos[0] + 0.45, f"yesterday\nbaseline\n{base_late:.2f}%", ha="right",
         va="top", color=INK2, fontsize=8.5)
for model, c in COLORS.items():
    vals = [t.loc[(model, k), "frozen 2023-26"] for k in ROWS]
    ax2.plot(vals, ypos + offset[model], "o", color=c, ms=6.5, mec=SURFACE, mew=1.2, zorder=3)
    lean = vals[-1]                                     # selective direct labels: lean-set row
    ax2.text(lean + 0.12, ypos[-1] + offset[model], f"{lean:.1f}", va="center", color=INK2, fontsize=8.5)
    ax2.text(vals[0] + 0.12, ypos[0] + offset[model], f"{vals[0]:.1f}", va="center", color=INK2, fontsize=8.5)
ax2.set_xlim(5.5, 13.5)
ax2.set_xlabel("MAPE of frozen model, Jan 2023 – Oct 2026 (%)\n(trained on 2010–2013, never retrained)")
ax2.set_title("Robustness: error when kept frozen", loc="left", fontsize=10, color=INK)

ax1.set_yticks(ypos)
ax1.set_yticklabels(list(ROWS.values()))
ax1.set_ylim(ypos[-1] - 0.6, ypos[0] + 0.6)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7, label=m) for m, c in COLORS.items()]
fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.985, 0.985))
fig.suptitle("NSW day-ahead demand: feature ablation", x=0.012, y=0.975, ha="left", fontsize=12,
             fontweight="bold", color=INK)
path = os.path.join(OUT, "fig_ablation_features.png")
fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
print("saved", path)
