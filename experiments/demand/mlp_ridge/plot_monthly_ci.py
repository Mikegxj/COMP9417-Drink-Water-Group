"""Monthly MAPE of the selected model (MLP), frozen and adapted (periodic retraining), with both
95% intervals, and both baselines (reads outputs/protocol/metrics_by_month.csv, run_metrics_all.py).

(a) frozen MLP, (b) MLP with periodic retraining (retrain dates marked), (c) width of the
day-block bootstrap interval divided by the half-hours-independent interval, for both.
"""
import os

import matplotlib
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

m = pd.read_csv(os.path.join(OUT, "metrics_by_month.csv"))
m["date"] = pd.PeriodIndex(m.month, freq="M").to_timestamp() + pd.Timedelta(days=14)
yd = m[m.model == "baseline: same half-hour yesterday"]
lw = m[m.model == "baseline: same half-hour last week"]
RETRAINS = ["2017-01-01", "2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, axes = plt.subplots(3, 1, figsize=(12, 9.6), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.3], "hspace": 0.28})
fig.patch.set_facecolor(SURFACE)
for ax in axes:
    style(ax)
panels = [("frozen", ORANGE, "(a) Frozen MLP (trained 2010–2014, never updated)"),
          ("periodic", BLUE, "(b) MLP with periodic retraining (4-year window; dotted lines: retrain dates)")]
for ax, (kind, c, title) in zip(axes[:2], panels):
    s = m[(m.model == "MLP") & (m.strategy == kind)]
    ax.plot(lw.date, lw["MAPE (%)"], color=REF, lw=1.0, ls=(0, (1, 1.5)), label="same half-hour last week")
    ax.plot(yd.date, yd["MAPE (%)"], color=REF, lw=1.2, ls=(0, (4, 2)), label="same half-hour yesterday")
    ax.fill_between(s.date, s.boot_low, s.boot_high, color=c, alpha=0.18, lw=0, label="95% day-block bootstrap")
    ax.fill_between(s.date, s.normal_low, s.normal_high, color=c, alpha=0.45, lw=0,
                    label="95% normal approx. (half-hours independent)")
    ax.plot(s.date, s["MAPE (%)"], color=c, lw=1.6, label=f"MLP, {kind}")
    ax.set_ylabel("Monthly MAPE (%)")
    ax.set_ylim(0, 20)
    for _, r in lw[lw["MAPE (%)"] > 20].iterrows():          # clipped peaks of the weaker baseline
        ax.annotate(f"last week {r['MAPE (%)']:.1f}% ({r.month})", (r.date, 20), xytext=(-4, -2),
                    textcoords="offset points", ha="right", va="top", fontsize=8, color=INK2)
    ax.set_title(title, loc="left", fontsize=10.2)
    h, lab = ax.get_legend_handles_labels()
    order = [4, 3, 2, 1, 0]
    ax.legend([h[i] for i in order], [lab[i] for i in order], loc="upper left", fontsize=8.5, ncol=2,
              frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1)
for d in RETRAINS:
    axes[1].axvline(pd.Timestamp(d), color=BLUE, lw=0.9, ls=(0, (1, 2)))
for kind, c in (("frozen", ORANGE), ("periodic", BLUE)):
    s = m[(m.model == "MLP") & (m.strategy == kind)]
    axes[2].plot(s.date, s.width_ratio, color=c, lw=1.2, label=f"MLP, {kind}")
axes[2].axhline(1, color=INK2, lw=0.8)
axes[2].set_ylim(0, None)
axes[2].set_ylabel("bootstrap width /\nnormal width")
axes[2].set_title("(c) Ratio of interval widths (1 = equal)", loc="left", fontsize=10.2)
axes[2].legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2, bbox_to_anchor=(1, 1.0))
axes[2].xaxis.set_major_locator(mdates.YearLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[2].set_xlabel("Month (Jan 2015 – Oct 2026)")
path = os.path.join(OUT, "fig_monthly_ci_mlp.png")
fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
print("saved", path)
