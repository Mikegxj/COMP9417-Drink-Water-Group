"""Data analysis for the demand task (report section 2): summary statistics and demand patterns.

Descriptive only (no model is fitted or selected here). Market days as in harness/dayahead.py
(D 00:30 ... D+1 00:00); slot k ends at 00:30 + 30k min, so midday 11:30-14:00 = slots 22-27 and
evening 17:30-20:00 = slots 34-39.

Outputs (outputs/eda/):
  describe.csv          distribution of NSW demand and Sydney weather over the whole file
  summary_by_year.csv   yearly level, peaks, midday vs evening, timing of the daily minimum,
                        day-to-day persistence (why the yesterday baseline is hard to beat)
  fig_eda_demand.png    (a) monthly mean demand: all half-hours, midday, evening
                        (b) mean workday profile in five years
                        (c) daily mean demand vs daily mean Sydney temperature, early vs recent years
                        (d) mean profile by day type in 2014
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import features as F  # noqa: E402

OUT = os.path.join(F.OUT_ROOT, "eda")
os.makedirs(OUT, exist_ok=True)
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
MID, EVE = slice(22, 28), slice(34, 40)

df = F.load()
days, arr = F.to_daily(df)
cal = F.calendar(days)
Y, T = arr["nsw_demand"], arr["syd_temp"]
year = days.year.to_numpy()
work = cal.workday.to_numpy() == 1

# --- summary statistics ----------------------------------------------------------------------
cols = ["nsw_demand", "syd_temp", "syd_rh", "syd_cloud", "syd_solar"]
desc = df[cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
desc.to_csv(os.path.join(OUT, "describe.csv"))

rows = []
dmean = Y.mean(axis=1)
for y in sorted(set(year)):
    m = year == y
    a = Y[m]
    nxt = np.flatnonzero(m)[:-1]
    rows.append({
        "year": y if y < 2026 else "2026 (Jan-Oct)", "days": int(m.sum()),
        "mean (MW)": a.mean(), "sd (MW)": a.std(), "min (MW)": a.min(), "max (MW)": a.max(),
        "mean daily peak (MW)": a.max(axis=1).mean(), "mean daily min (MW)": a.min(axis=1).mean(),
        "midday 11:30-14:00 (MW)": a[:, MID].mean(), "evening 17:30-20:00 (MW)": a[:, EVE].mean(),
        "midday / evening": a[:, MID].mean() / a[:, EVE].mean(),
        "share of days with minimum in daytime, half-hours ending 09:00-16:30 (%)": 100 * np.mean((a.argmin(axis=1) >= 17) & (a.argmin(axis=1) <= 32)),
        "corr of daily mean, day D vs D+1": np.corrcoef(dmean[nxt], dmean[nxt + 1])[0, 1],
        "mean |change| of daily mean, D to D+1 (%)": 100 * np.mean(np.abs(dmean[nxt + 1] / dmean[nxt] - 1)),
    })
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "summary_by_year.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(desc.round(1).to_string())
print()
print(summ.round(3).to_string(index=False))


# --- figure ----------------------------------------------------------------------------------
def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), gridspec_kw={"hspace": 0.42, "wspace": 0.22})
fig.patch.set_facecolor(SURFACE)
for ax in axes.ravel():
    style(ax)
hours = (np.arange(48) + 1) / 2                      # end of interval, hours after midnight

# (a) monthly means
a = axes[0, 0]
mon = days.to_period("M")
mm = pd.DataFrame({"all": Y.mean(axis=1), "midday": Y[:, MID].mean(axis=1), "evening": Y[:, EVE].mean(axis=1)},
                  index=days).groupby(mon).mean()
x = mm.index.to_timestamp() + pd.Timedelta(days=14)
for col, c, lab in (("evening", ORANGE, "evening 17:30–20:00"), ("all", INK2, "all half-hours"),
                    ("midday", YELLOW, "midday 11:30–14:00")):
    a.plot(x, mm[col], color=c, lw=1.0, alpha=0.45)
    a.plot(x, mm[col].rolling(12, center=True).mean(), color=c, lw=2.0, label=lab)
a.set_ylabel("NSW demand (MW)")
a.set_title("(a) Monthly mean demand (thin) and 12-month mean (thick)", loc="left", fontsize=10)
a.set_ylim(None, 12900)
a.legend(frameon=False, fontsize=8.3, loc="upper center", ncol=3, handlelength=1.6, columnspacing=1.0)
a.xaxis.set_major_locator(mdates.YearLocator(2))
a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
a.set_xlabel("Month")

# (b) mean workday profile in selected years (ordered years -> one hue, light to dark)
b = axes[0, 1]
YEARS = [2010, 2014, 2018, 2022, 2025]
shades = ["#a9c9f0", "#7aaae6", "#4d8bdc", "#2a6cc0", "#173f78"]
for y, c in zip(YEARS, shades):
    m = (year == y) & work
    b.plot(hours, Y[m].mean(axis=0), color=c, lw=1.8, label=str(y))
b.set_xticks(range(0, 25, 3))
b.set_xlim(0, 24)
b.set_xlabel("Time of day (end of half-hour, h)")
b.set_ylabel("NSW demand (MW)")
b.set_title("(b) Mean workday profile (Mon–Fri, not a public holiday)", loc="left", fontsize=10)
b.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=1, title="year", title_fontsize=8.5)

# (c) daily demand vs temperature, early vs recent, with binned means
c_ = axes[1, 0]
tm = T.mean(axis=1)
for (y0, y1), col in (((2010, 2013), BLUE), ((2023, 2026), ORANGE)):
    m = (year >= y0) & (year <= y1) & work
    c_.scatter(tm[m], dmean[m], s=5, color=col, alpha=0.25, lw=0)
    bins = np.arange(np.floor(tm[m].min()), np.ceil(tm[m].max()) + 1, 1.0)
    k = np.digitize(tm[m], bins)
    bx = [tm[m][k == i].mean() for i in np.unique(k) if (k == i).sum() >= 10]
    by = [dmean[m][k == i].mean() for i in np.unique(k) if (k == i).sum() >= 10]
    c_.plot(bx, by, color=col, lw=2.2, label=f"{y0}–{y1 if y1 < 2026 else '26 (Oct)'} (1 °C bin means)")
c_.set_xlabel("Daily mean Sydney temperature (°C)")
c_.set_ylabel("Daily mean NSW demand (MW)")
c_.set_title("(c) Demand vs temperature, workdays", loc="left", fontsize=10)
c_.legend(frameon=False, fontsize=8.5, loc="upper center")

# (d) mean profile by day type in 2014
d = axes[1, 1]
m14 = year == 2014
dow = cal.dow.to_numpy()
hol = cal.hol.to_numpy() == 1
types = [("workday", m14 & work, BLUE), ("Saturday", m14 & (dow == 5) & ~hol, AQUA),
         ("Sunday", m14 & (dow == 6) & ~hol, YELLOW), ("weekday public holiday", m14 & (dow < 5) & hol, ORANGE)]
for lab, m, col in types:
    d.plot(hours, Y[m].mean(axis=0), color=col, lw=1.8, label=f"{lab} (n = {m.sum()})")
d.set_xticks(range(0, 25, 3))
d.set_xlim(0, 24)
d.set_xlabel("Time of day (end of half-hour, h)")
d.set_ylabel("NSW demand (MW)")
d.set_title("(d) Mean profile by day type, 2014", loc="left", fontsize=10)
d.legend(frameon=False, fontsize=8.5, loc="lower right")

path = os.path.join(OUT, "fig_eda_demand.png")
fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
print("saved", path)
