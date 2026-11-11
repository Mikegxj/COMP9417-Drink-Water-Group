"""Figures for the degradation / drift section (reads outputs/drift/*.csv from run_drift.py)."""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import features as F  # noqa: E402

D = os.path.join(F.OUT_ROOT, "drift")
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # reference palette slots 1-3
EVENTS = [("2020-03-31", "2020-05-15", "NSW COVID-19\nrestrictions"),
          ("2021-06-26", "2021-10-11", "Greater Sydney\nlockdown")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    path = os.path.join(D, name)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("saved", path)


# --- 1. monthly MAPE of the frozen MLP with both intervals ------------------------------------
m = pd.read_csv(os.path.join(D, "monthly_frozen_ci.csv"))
m["date"] = pd.PeriodIndex(m.month, freq="M").to_timestamp() + pd.Timedelta(days=14)
mlp, base = m[m.model == "mlp"], m[m.model == "baseline_yesterday"]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
fig.patch.set_facecolor(SURFACE)
for ax in (a1, a2):
    style(ax)
a1.fill_between(mlp.date, mlp.boot_low, mlp.boot_high, color=BLUE, alpha=0.18, lw=0,
                label="95% day-block bootstrap")
a1.fill_between(mlp.date, mlp.normal_low, mlp.normal_high, color=BLUE, alpha=0.45, lw=0,
                label="95% normal approx. (half-hours independent)")
a1.plot(mlp.date, mlp.mape, color=BLUE, lw=1.6, label="frozen MLP (trained 2010–2014)")
a1.plot(base.date, base.mape, color=REF, lw=1.2, ls=(0, (4, 2)), label="same half-hour yesterday")
a1.set_ylabel("Monthly MAPE (%)")
a1.legend(frameon=False, loc="upper left", fontsize=8.8)
a1.set_title("Frozen MLP: monthly error, Jan 2015 – Oct 2026", loc="left", fontsize=10.5)
a2.plot(mlp.date, mlp.width_ratio, color=INK2, lw=1.2)
a2.axhline(1, color=GRID, lw=1)
a2.set_ylabel("bootstrap width /\nnormal width")
a2.xaxis.set_major_locator(mdates.YearLocator())
a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
save(fig, "fig_monthly_frozen_ci.png")

# --- 2. detector statistics on the frozen MLP --------------------------------------------------
d = pd.read_csv(os.path.join(D, "detector_daily_frozen.csv"), parse_dates=["date"])
thr = pd.read_csv(os.path.join(D, "thresholds.csv"), index_col=0)
panels = [("shape_28d", "input_shape", "Input detector: daily-shape novelty\n(28-day mean, PCA fitted 2010–2013)", BLUE),
          ("weather_28d", "weather", "Diagnostic: weather novelty\n(28-day mean)", AQUA),
          ("error_ratio_28d", "error_ratio", "Error detector: MLP / yesterday-baseline\nMAPE ratio (28 days)", ORANGE)]
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.patch.set_facecolor(SURFACE)
for ax, (col, det, label, c) in zip(axes, panels):
    style(ax)
    for s, e, txt in EVENTS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#efeee9", lw=0, zorder=0)
    ax.plot(d.date, d[col], color=c, lw=1.2)
    t99 = thr.loc["q99", det]
    ax.axhline(t99, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(d.date.iloc[-1], t99, " q99 threshold", va="bottom", ha="right", color=INK2, fontsize=8.5)
    above = d[col] > t99
    if above.any():
        f = d.date[above.idxmax()]
        ax.axvline(f, color=c, lw=0.9, alpha=0.7)
        ax.text(f, ax.get_ylim()[1], f" first signal {f.date()}", va="top", ha="left", color=INK2, fontsize=8.5)
    ax.set_ylabel(label, fontsize=8.8)
for s, e, txt in EVENTS:
    axes[0].text(pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2, axes[0].get_ylim()[1] * 0.98, txt,
                 ha="center", va="top", fontsize=7.5, color=INK2)
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0].set_title("Online detectors on the frozen MLP (references fixed on 2010–2014; shaded: lockdowns)",
                  loc="left", fontsize=10.5)
save(fig, "fig_detectors_frozen.png")

# --- 3. retraining trade-off and monthly error of selected strategies --------------------------
s = pd.read_csv(os.path.join(D, "strategies.csv"), index_col=0)
sm = pd.read_csv(os.path.join(D, "strategies_monthly.csv"), index_col=0)
sm.index = pd.PeriodIndex(sm.index, freq="M").to_timestamp() + pd.Timedelta(days=14)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.2})
fig.patch.set_facecolor(SURFACE)
for ax in (b1, b2):
    style(ax)
per = s[s.index.str.startswith("periodic,") & s.index.str.contains("evenly")]
xs = np.r_[0, per.retrains.to_numpy()]
ys = np.r_[s.loc["frozen", "MAPE all"], per["MAPE all"].to_numpy()]
b1.plot(xs, ys, "-o", color=BLUE, ms=6, lw=1.6, mec=SURFACE, label="periodic, evenly spaced")
tm = s.loc["periodic, 1 Jan 2017/19/21/23/25 (teammate)"]
b1.plot(tm.retrains, tm["MAPE all"], "s", color=BLUE, ms=7, mfc=SURFACE, mew=1.6, label="periodic, 1 Jan every 2 years")
for prefix, c, mk in (("error-triggered", ORANGE, "D"), ("input-triggered", AQUA, "^")):
    g = s[s.index.str.startswith(prefix)]
    b1.plot(g.retrains, g["MAPE all"], mk, color=c, ms=7, mec=SURFACE, label=prefix)
    order = g["MAPE all"].rank(method="first").astype(int)       # stack labels by value
    for name, r in g.iterrows():
        dy = {1: -11, 2: 0, 3: 11}[order[name]] if len(g) == 3 else {1: -7, 2: 7}[order[name]]
        b1.annotate(name.split()[-1], (r.retrains, r["MAPE all"]), xytext=(9, dy), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
b1.set_xlabel("Number of retrains, Jan 2015 – Oct 2026")
b1.set_ylabel("MAPE 2015–2026 (%)")
b1.set_title("Retrains vs error (MLP, 4-year windows)", loc="left", fontsize=10.5)
b1.legend(frameon=False, fontsize=8.5)
lines = [("frozen", REF, "frozen"), ("periodic, 1 Jan 2017/19/21/23/25 (teammate)", BLUE, "periodic (2 yr)"),
         ("error-triggered q99", ORANGE, "error-triggered q99"), ("input-triggered q99", AQUA, "input-triggered q99")]
for col, c, lab in lines:
    b2.plot(sm.index, sm[col], color=c, lw=1.4 if col != "frozen" else 1.1, label=lab)
    dates = s.loc[col, "dates"]
    if isinstance(dates, str):
        for x in dates.split():
            b2.axvline(pd.Timestamp(x), color=c, lw=0.8, alpha=0.5, ymax=0.06)
b2.set_ylabel("Monthly MAPE (%)")
b2.set_title("Monthly error by strategy (ticks at bottom: retrain dates)", loc="left", fontsize=10.5)
b2.legend(frameon=False, fontsize=8.5, loc="upper left")
b2.xaxis.set_major_locator(mdates.YearLocator(2))
b2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
save(fig, "fig_retraining_strategies.png")

# --- 4. first vs second iteration of drift-triggered retraining --------------------------------
v2_path = os.path.join(D, "strategies_v2.csv")
if os.path.exists(v2_path):
    v2 = pd.read_csv(v2_path, index_col=0)
    rows = [("frozen", "Frozen", REF, s.loc["frozen"]),
            ("p6", "Periodic, 6 evenly spaced", BLUE, s.loc["periodic, 6 evenly spaced"]),
            ("p5", "Periodic, 1 Jan every 2 years", BLUE, s.loc["periodic, 1 Jan 2017/19/21/23/25 (teammate)"]),
            ("e1", "Error-triggered v1 (2014 threshold, 180-day gap)", ORANGE, s.loc["error-triggered q99"]),
            ("e2", "Error-triggered v2: recalibrated, 180-day gap", ORANGE, v2.loc["error: recalibrated, 180-day gap"]),
            ("e3", "Error-triggered v2: 2014 threshold, paced", ORANGE, v2.loc["error: fixed 2014 threshold + paced"]),
            ("e4", "Error-triggered v2: recalibrated + paced", ORANGE, v2.loc["error: recalibrated + paced"]),
            ("i1", "Input-triggered v1 (180-day gap)", AQUA, s.loc["input-triggered q99"]),
            ("i2", "Input-triggered v2: paced", AQUA, v2.loc["input: re-fitted reference + paced"])]
    rows.sort(key=lambda r: -r[3]["MAPE all"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    for k, (_, label, c, r) in enumerate(rows):
        ax.plot([2.3, r["MAPE all"]], [k, k], color=GRID, lw=1, zorder=1)
        ax.plot(r["MAPE all"], k, "o", color=c, ms=8, mec=SURFACE, mew=1, zorder=3)
        ax.text(r["MAPE all"] + 0.05, k, f"{r['MAPE all']:.2f}%  ({int(r.retrains)} retrains)", va="center",
                fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows])
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(2.3, 5.6)
    ax.set_xlabel("MAPE 2015–2026 (%) of the MLP under each retraining strategy (budget: 6 retrains)")
    ax.set_title("Pacing the budget, not the detector, closes the gap to periodic retraining", loc="left",
                 fontsize=10.5)
    save(fig, "fig_detector_v2.png")
