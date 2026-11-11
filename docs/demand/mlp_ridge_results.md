# Demand part — results pack (facts and numbers for writing the report)

This file collects the numbers, tables, figures and caveats behind the demand part. It is a
working reference, not report text: the spec requires the report to be written by the group.
All MAPE values are in %, computed over half-hours. "Full period" = 2015-01 to 2026-10.
"Recent" = 2025-01 to 2026-10 (already looked at several times, so treat as exploratory).

Code: `experiments/demand/mlp_ridge/` (see `docs/demand/mlp_ridge.md` for the run order).
Package: `submission_mlp/` (MLP demand model; price direction still a placeholder; proposed to replace `submission/`).

---

## 1. Setup facts

- Day D = intervals stamped D 00:30 … D+1 00:00 (`harness/dayahead.py`). Forecast made at the end of D for the 48 half-hours of D+1.
- Our baselines and targets were checked against `dayahead.baseline` / `dayahead.targets` (identical); both baselines (yesterday = day D, last week = day D−6) are re-checked in `run_metrics_all.py`.
- Both metrics (MAPE and MAE in MW, the harness implementations) and both baselines are reported for every period: `results/demand/mlp_ridge/protocol/metrics_summary.csv`, `metrics_by_year.csv`, `metrics_by_month.csv`.

## 1b. Data analysis (report section 2)
Files: `results/demand/mlp_ridge/eda/describe.csv`, `summary_by_year.csv`; figure `results/demand/mlp_ridge/eda/fig_eda_demand.png` (script `run_eda.py`, descriptive only).

- Whole file: 295,104 half-hours; NSW demand mean 7,918 MW, s.d. 1,361, min 2,582 (2025), max 14,580 (2011); 1–99 % range 5,054–11,396.
- Level: yearly mean 8,807 MW (2010) → 7,918 (2014) → 7,497 (2025). The fall is concentrated at midday: midday 11:30–14:00 mean 9,561 → 8,390 → 6,186 MW, while the evening 17:30–20:00 mean is flat (9,816 → 8,914 → 9,362, rising since 2021). Midday / evening ratio 0.97 (2010) → 0.94 (2014) → 0.66 (2025).
- Timing of the daily minimum: share of days whose minimum falls in daytime (half-hours ending 09:00–16:30) 0 % until 2018 → 3.6 % (2019) → 27 % (2021) → 60 % (2023) → 72 % (2025): the overnight trough has been replaced by a midday trough (rooftop PV, which the data do not contain directly).
- Workday profile (panel b): the 2010–2014 daytime plateau turns into a midday valley (2025: ~6,400 MW at 12:30 vs ~9,700 MW at 18:30); the evening peak is the stable part.
- Demand vs temperature (panel c): U-shape with minimum near 18–19 °C daily mean (heating below, cooling above); 2023–26 curve is ~1,000–1,300 MW lower at the same temperature than 2010–13: the same weather now comes with less demand, so the demand–temperature relationship itself shifted, not only the temperatures.
- Calendar (panel d, 2014): weekday public holidays look like Sundays (≈1,600 MW below a workday at midday); Saturday/Sunday differ mainly in the morning.
- Why the yesterday baseline is hard to beat: daily mean demand is persistent (correlation day D vs D+1: 0.68–0.85 per year; mean |change| of the daily mean 4.2–5.8 %), so copying yesterday is already within ~5 % on most days; it fails on weekday↔weekend transitions, holidays and weather changes, which is exactly what calendar and weather features fix. The last-week baseline handles the weekly cycle but is further away in time and weather, so it is worse (7.42 vs 6.15 % over 2015–26).
- Inputs only use demand up to the end of D; weather of D+1 (reanalysis, treated as known); calendar incl. NSW public holidays from the `holidays` package.
- Scalers / PCA / thresholds are fitted on training data only; a refit of any of them happens only on a retraining occasion.
- Stage 1: train on target days in 2010–2013, select on 2014. Neural early stopping uses an inner split (train 2010–2012, stop on 2013), then refit on 2010–2013 with that number of epochs, so 2014 is used only for selection.
- Stage 2 protocol (same as the teammate's formal backtest): frozen = refit on 2010–2014; periodic = refits on 1 Jan 2017/2019/2021/2023/2025, each on the previous four years (5 retrains); hyperparameters and epochs fixed at Stage 1 values.

## 2. Models

| Model | Family | Owner | Features | Target |
|---|---|---|---|---|
| MLP (selected) | feed-forward NN, 402-256-256-48, 5 seeds averaged | us | demand of D and D-6 (scaled by 7-day mean), Sydney temp/rh/cloud/solar of D+1, temp and solar of D, weekday, holiday flags D+1/D/D-6, Christmas flag, season sin/cos ×2, Tmax/Tmin/3-day mean temp | demand / 7-day mean, MAPE loss |
| Ridge + splines | linear/additive, one model per half-hour | us | same information + weekday×lag and holiday×lag interactions, temperature splines, VIC/QLD demand of D | demand / 7-day mean |
| LightGBM | gradient-boosted trees | teammate (we reproduce it exactly) | slot, weekday, season, temp D+1, temp D same slot, lags 1/2/7 (MW), cloud, holiday flags | MW |
| N-BEATSx | deep sequence NN (pure PyTorch) | teammate | 480 half-hours of demand + Sydney temp, calendar sin/cos, holiday flags | scaled MW, L1 loss |
| Appendix: GAM, TFT, TCN, LSTM | | teammate | see docs/demand in the group repo | |

## 3. Stage 1 — validation 2014 (fit 2010–2013)

| Model | 2014 MAPE |
|---|---|
| MLP (5-seed average; per-seed mean 1.67) | **1.53** |
| Ridge + VIC/QLD | 1.88 |
| N-BEATSx (teammate's configuration, 3-seed mean; teammate's run 1.89, our run 1.86, our seed 42 alone 2.13) | 1.86–1.89 |
| TFT (teammate) | 1.91 |
| Ridge, Sydney only | 2.00 |
| GAM (teammate) | 2.28 |
| LightGBM (teammate) | 2.29 |
| TCN / LSTM (teammate, old features) | 2.42 / 2.33 |
| Same half-hour yesterday / last week | 4.93 / 4.93 |

MAE in 2014 (MW): MLP 124, Ridge 152, N-BEATSx 152, LightGBM 183, yesterday 394, last week 401 (same ranking as MAPE).

Selection rule (Stage 1): lowest 2014 MAPE → **MLP**.
Weekday detail (2014): yesterday baseline is worst on Mon (9.4) and Sat (8.6); the models are flat across weekdays (1.3–2.2).

### Feature ablations (2014, change vs full features, 95% day-block bootstrap)
File: `results/demand/mlp_ridge/ablation_features.csv`, `results/demand/mlp_ridge/ablation_holidays*.csv`; figure `results/demand/mlp_ridge/fig_ablation_features.png`.

| Change | Ridge | MLP |
|---|---|---|
| + Melbourne & Brisbane weather | −0.01 [−0.07, +0.06] | +0.04 [−0.00, +0.07] |
| + VIC & QLD demand of D | **−0.12 [−0.15, −0.09]** | +0.03 [−0.02, +0.07] |
| − Sydney solar & cloud | +0.04 [+0.02, +0.06] | +0.06 [−0.00, +0.11] |
| − public-holiday flags | **+0.21 [+0.10, +0.36]** | **+0.24 [+0.14, +0.37]** |
| − holidays and Christmas flag | +0.19 [+0.07, +0.35] | +0.32 [+0.21, +0.45] |
| teammate's 11 features only | +0.45 [+0.30, +0.61] | +0.36 [+0.24, +0.50] |

- Holidays: on weekday public holidays (2014–2018, 44 days) MLP MAPE 2.67 with flags vs 9.03 without (yesterday baseline 7.18). The teammate's own ablation agrees for all their models (−6 to −11%).
- Solar matters little in 2014 but a lot later: without solar & cloud the periodic MLP is +0.20 overall and **+0.39 [+0.29, +0.50] in 2025–26** (midday 6.7 → 8.1). Feature importance itself drifts with rooftop PV.
- Decision: Sydney weather only for MLP; VIC/QLD only helps the linear model.

## 4. Stage 2 — four main models under the same protocol
Files: `results/demand/mlp_ridge/protocol/metrics_summary.csv` (MAPE + MAE, both baselines), `metrics_by_year.csv`, `metrics_by_month.csv`; older `summary.csv`, `summary_nbeats.csv`, `annual_four_models.csv`; figure `results/demand/mlp_ridge/protocol/fig_four_models_annual.png` (all four models from our own runs on the same days, both baselines).

MAPE (%):

| Model | Frozen 2015–24 | Frozen recent | Frozen full | Periodic 2015–24 | Periodic recent | **Periodic full** |
|---|---|---|---|---|---|---|
| MLP | 4.17 | 9.19 | 4.94 | 2.41 | 3.05 | **2.51** |
| Ridge + VIC/QLD | 4.25 | 8.78 | 4.95 | 2.99 | 3.91 | 3.13 |
| LightGBM | 3.73 | 6.71 | 4.19 | 3.22 | 4.44 | 3.41 |
| N-BEATSx (our run, 3-seed mean) | 4.79 | 11.05 | 5.76 | 2.90 | 4.39 | 3.13 |
| N-BEATSx (teammate's run, seed 42) | 4.78 | 11.07 | 5.75 | 3.06 | 4.52 | 3.29 |
| GAM (teammate) | 3.71 | 5.90 | 4.05 | 3.44 | 5.08 | 3.70 |
| TFT (teammate, frozen only) | 5.89 | 13.68 | 7.10 | – | – | – |
| Yesterday baseline | 5.90 | 7.53 | 6.15 | 5.90 | 7.53 | 6.15 |
| Last-week baseline | 6.95 | 10.00 | 7.42 | 6.95 | 10.00 | 7.42 |

MAE (MW):

| Model | Frozen 2015–24 | Frozen recent | Frozen full | Periodic 2015–24 | Periodic recent | Periodic full |
|---|---|---|---|---|---|---|
| MLP | 298 | 578 | 341 | 184 | 216 | **189** |
| Ridge + VIC/QLD | 305 | 549 | 343 | 226 | 269 | 233 |
| LightGBM | 278 | 424 | 300 | 244 | 304 | 254 |
| N-BEATSx (our run, 3-seed mean) | 346 | 707 | 402 | 223 | 302 | 235 |
| Yesterday baseline | 459 | 532 | 471 | 459 | 532 | 471 |
| Last-week baseline | 545 | 704 | 570 | 545 | 704 | 570 |

- MAE ranks the four models exactly as MAPE in every period above, so the conclusions do not depend on the metric.
- Frozen models cross the yesterday baseline: Ridge 2021, MLP 2022, N-BEATSx/TFT/TCN 2021; LightGBM, GAM, LSTM never (but 2025: 7.7/6.5/7.7 vs 8.1).
- Against the weaker last-week baseline: frozen MLP, Ridge and LightGBM stay below it in every year (MLP 2025: 9.81 vs 10.55); frozen N-BEATSx crosses it in 2023 (8.60 vs 7.90).
- Last-week baseline is worst around Christmas (Dec 2025 monthly MAPE 23.2 %: the week before is a normal working week).
- LightGBM reproduction check: periodic annual MAPE identical to the teammate's every year 2017–2026; frozen within 0.05.
- N-BEATSx reproduction check (`nbeats_reproduction_check.csv`): our seed 42 vs the teammate's seed 42 differ by −0.20…+0.17 per year (not bit-identical: training is not deterministic across machines/thread counts); full-period periodic 3.31 vs 3.29. Averaging three seeds helps N-BEATSx a lot (periodic full 3.31 → 3.13; 2014 2.13 → 1.86), as five seeds help the MLP (2.70 → 2.51).
- Paired day-block bootstrap vs MLP (periodic, `paired_differences_nbeats.csv`): N-BEATSx +0.62 [+0.58, +0.67] (2015–26), +1.34 [+1.19, +1.50] (2025–26); Ridge +0.63 [+0.59, +0.67] / +0.86 [+0.74, +0.99].
- MLP is not best in every year: periodic 2017 N-BEATSx 2.23 vs MLP 2.25; 2018 all three others 2.60–2.74 vs MLP 2.75. From 2019 on the MLP is best every year.
- Recent period by time of day (periodic): MLP midday 6.73 / evening 2.21; LightGBM 10.65 / 2.62; baseline 16.24 / 5.22.
- Weekly improvement I vs yesterday (94 Mon–Sun weeks of 2025–26, periodic; `weekly_improvement_nbeats.csv`):

| | median I | 10th pct | min | weeks with I ≥ 40 % |
|---|---|---|---|---|
| MLP | 60.6 | 43.9 | 32.2 | 93.6 % |
| Ridge | 48.3 | 34.5 | 6.0 | 74.5 % |
| N-BEATSx | 44.2 | 23.1 | −3.8 | 61.7 % |
| LightGBM | 42.5 | 25.3 | −21.1 | 61.7 % |

### N-BEATSx with cloud & solar (fairness check)
Files: `results/demand/mlp_ridge/protocol/nbeats_wx_summary.csv`, `nbeats_wx_by_year.csv`, `nbeats_wx_paired.csv` (script `run_nbeats_wx.py`).
The teammate's N-BEATSx configuration knows only Sydney temperature, calendar sin/cos and holiday flags; the MLP also uses humidity, cloud and solar. Here `syd_cloud` and `syd_solar` were added to N-BEATSx's known covariates (standardised on the training rows; same seeds 42/43/44 and protocol).

| MAPE % (MAE MW) | 2014 | periodic 2015–24 | periodic 2025–26 | periodic full | midday 2025–26 | weekly I 2025–26: median / weeks ≥ 40 % |
|---|---|---|---|---|---|---|
| MLP | 1.53 | 2.41 | 3.05 (216) | 2.51 (189) | 6.59 | 60.6 / 93.6 % |
| MLP without solar & cloud | 1.59 | 2.58 | 3.43 (239) | 2.71 (203) | 7.97 | 56.0 / 89.4 % |
| N-BEATSx + cloud & solar | 1.84 | 2.76 | 3.80 (268) | 2.92 (221) | 7.77 | 48.7 / 76.6 % |
| N-BEATSx (teammate's inputs) | 1.86 | 2.90 | 4.39 (302) | 3.13 (235) | 10.57 | 44.2 / 61.7 % |

- Cloud & solar help N-BEATSx a lot in recent years: −0.21 [−0.25, −0.18] over 2015–26 and −0.59 [−0.71, −0.46] in 2025–26 (midday 10.6 → 7.8 %), but not at all in 2014 (−0.01 [−0.08, +0.06]). The value of solar information appears only as rooftop PV grows, so a Stage 1 selection on 2014 cannot see it — the same pattern as the MLP ablation (without solar & cloud: +0.06 in 2014, +0.39 in 2025–26). Frozen N-BEATSx is slightly worse with them (5.87 vs 5.76).
- It is still worse than the MLP (paired day-block bootstrap): +0.32 [+0.23, +0.40] in 2014, +0.41 [+0.37, +0.45] over 2015–26, +0.76 [+0.64, +0.87] in 2025–26, and worse in every single year 2015–2026. It is even worse than the MLP *without* solar & cloud (+0.25 / +0.21 / +0.37). The gap is the model, not the weather inputs; the choice of the MLP does not change.

### Cross experiment: model family × information set (periodic full)
File: `results/demand/mlp_ridge/protocol/cross_summary.csv`.

| | Lean (teammate's 11) | Rich (ours) |
|---|---|---|
| LightGBM | 3.41 (2014: 2.29) | 3.01 (2014: 1.95) |
| MLP | 2.96 (2014: 1.89) | 2.51 (2014: 1.53) |

- Feature effect ≈ −0.40 (LightGBM) / −0.46 (MLP); model effect ≈ −0.45 (lean) / −0.51 (rich): about equal and roughly additive.
- Robustness when frozen is a model-family property: LightGBM frozen 3.95–4.19 vs MLP 4.94–5.87 for either feature set; richer features made both more robust, not less (the Ridge exception: its weekday×lag interactions tie it to the training-era shape).

### Ensembles (weights fixed on 2014 only)
File: `results/demand/mlp_ridge/protocol/ensemble_weights.csv`, `paired_differences.csv`.
- MLP+Ridge+LightGBM (0.75/0.15/0.10): periodic full 2.470, −0.039 [−0.049, −0.029] vs MLP; MLP+LightGBM (0.9/0.1): −0.028.
- Equal weights are worse than MLP alone (2.66).
- Decision: submit the 5-seed MLP alone (gain ≈ 1.5% relative is not worth two extra models; in the live-week analogue week MLP alone was better).

## 5. Degradation and drift detection
Files: `results/demand/mlp_ridge/drift/*`; figures `results/demand/mlp_ridge/protocol/fig_monthly_ci_mlp.png` (main figure: frozen and periodic MLP with both intervals and both baselines), `fig_detectors_frozen.png`, `fig_retraining_strategies.png`, `fig_detector_v2.png` (older frozen-only version: `fig_monthly_frozen_ci.png`).

### Monthly intervals (frozen and adapted models)
Files: `results/demand/mlp_ridge/protocol/metrics_by_month.csv` (every model × strategy × month: MAPE, MAE, both intervals), summary `monthly_ci_width_summary.csv`.

| Model | Strategy | median width, half-hours independent | median width, day-block bootstrap | ratio (median, range) | equivalent independent obs. per month |
|---|---|---|---|---|---|
| MLP | frozen | 0.45 | 1.18 | 2.6 (1.2–4.2) | 210 |
| MLP | periodic | 0.24 | 0.66 | 2.8 (1.7–4.9) | 183 |
| Ridge | frozen | 0.48 | 1.13 | 2.6 (1.5–4.3) | 220 |
| Ridge | periodic | 0.31 | 0.94 | 2.7 (1.8–4.3) | 196 |
| LightGBM | frozen | 0.38 | 1.16 | 3.1 (2.0–4.7) | 154 |
| LightGBM | periodic | 0.32 | 0.94 | 3.0 (1.9–4.4) | 164 |
| N-BEATSx | frozen | 0.53 | 1.10 | 2.4 (1.2–4.2) | 264 |
| N-BEATSx | periodic | 0.31 | 0.90 | 2.7 (1.8–4.5) | 201 |
| Yesterday baseline | – | 0.61 | 2.44 | 4.1 (2.3–5.2) | 87 |
| Last-week baseline | – | 0.69 | 2.55 | 4.1 (2.3–5.7) | 88 |

- Equivalent independent observations ≈ n / ratio² ≈ 150–260 per month (instead of ≈1440): errors within a day are strongly correlated, so the naive interval is 2.4–3.1 times too narrow for every model, frozen or retrained. The baselines are even more correlated within a day (≈ 90): a wrong day (e.g. Monday copied from Sunday) is wrong in all 48 half-hours.
- For the MLP, periodic retraining roughly halves the interval widths as well as the MAPE (monthly errors are smaller and less variable; Ridge/LightGBM/N-BEATSx shrink less), but the bootstrap/normal ratio stays about the same (2.6 vs 2.8): the within-day dependence is a property of day-ahead errors, not of the model's age.
- Both intervals still treat days as independent; consecutive days are correlated (heatwaves, holidays), and week blocks give only 4–5 blocks per month. Interval values differ slightly from earlier versions of this table because the bootstrap draws changed (seeded, 1000 draws).

### Detectors (online; day D observed at end of D; retrain effective D+1)

| | Input detector | Error detector |
|---|---|---|
| Statistic | PCA reconstruction error of day D's normalised demand shape (the lag-1 input), 28-day mean over eligible days | sum of MLP daily MAPE / sum of yesterday-baseline daily MAPE over the last 28 eligible days |
| Reference | PCA fitted on 2010–2013 (5 components, 95% variance) | Stage 1 MLP (2010–2013) on 2014 |
| Threshold | q99 of the statistic over 2014 (0.0665) | q99 of the ratio over 2014 (0.371) |
| Eligible days | public holidays and 22 Dec–4 Jan skipped | same |

- Frozen MLP, first q99 signal: input 2015-06-06, error 2015-10-03 (weather diagnostic 2015-12-11). Frozen MAPE then was 1.57 (2014: 1.53) → early signals, gradual PV drift; from 2018 (error) / Oct 2019 (input) the statistics stay above the threshold permanently (saturation).
- Concept-drift check (days whose inputs stay inside the 2014 range of both input detectors): frozen MAPE 1.43 (2014) → 2.33 (2018) → 3.59 (2020) while the baseline on the same days stays 4.4–5.6; midday bias +1.2% → +6.8% (2020) → +10.9% (2021). Same-looking inputs, lower demand → P(Y|X) changed (unobserved rooftop PV). Few such days after 2021.
- Input change without harm: detrended monthly correlation with the error ratio only 0.23 (shape) / 0.25 (weather); weather novelty peaked Nov–Dec 2019 (heat/bushfire season) while the error ratio was low (0.47–0.63).
- Events (dates to verify against official sources before citing): NSW COVID-19 restrictions 31 Mar–15 May 2020 → error ratio Apr 0.89 vs Mar/May 0.60, shape detector flat (supporting evidence, error detector only); Greater Sydney lockdown 26 Jun–11 Oct 2021 → ratio 1.14 in Sep but every spring rises and the shape detector fell (coincidence in time, inconclusive).

## 6. Adaptation (selected model = MLP, budget 6 retrains, 4-year windows)
File: `results/demand/mlp_ridge/drift/strategies.csv`, `strategies_v2.csv`.

| Strategy | Retrains | Full MAPE |
|---|---|---|
| Frozen | 0 | 4.94 |
| Periodic, evenly spaced, k = 1/2/3/4/5/6 | k | 3.12 / 2.82 / 2.66 / 2.58 / 2.49 / 2.48 |
| Periodic, 1 Jan 2017/19/21/23/25 | 5 | 2.51 |
| Error-triggered v1 (2014 threshold, 180-day gap), q95/q99/max | 6 | 4.02 / 4.04 / 3.94 (budget gone by Sep 2018) |
| Input-triggered v1, q95/q99 | 6 | 3.16 / 3.17 (budget gone by Feb 2020) |
| v2 error: recalibrated threshold, 180-day gap | 6 | 3.07 |
| v2 error: 2014 threshold, paced | 6 | 2.50 |
| v2 error: recalibrated + paced | 6 | 2.48 |
| v2 input: re-fitted reference + paced | 6 | 2.45 |

- v2 is a second iteration designed after v1 failed (say so in the report). Pacing: minimum gap = days left / (retrains left + 1).
- Recalibrated error thresholds rose 0.371 → 0.527: even a freshly retrained model's error relative to the baseline grows over time (the task got harder), so a fixed threshold cannot tell "model aged" from "conditions harder".
- With pacing, retrains happen within weeks of the evenly spaced dates (e.g. 2016-10-02 vs 2016-09-09): under continuous drift the detector is almost always on.
- Final strategy: periodic retraining on the last four years (simple, equal to the best detector-based variant).

## 7. Day types
Files: `results/demand/mlp_ridge/daytypes/*`; figure `results/demand/mlp_ridge/daytypes/fig_daytypes.png`.
- Representation: normalised profiles (shape). At k = 4, ARI with workday/non-workday 0.255 (raw MW: 0.105); raw clusters mostly encode level, which drifts down over the years.
- k scan on 2010–2013: silhouette 0.41 (k=2), 0.34 (3), 0.31 (4), 0.34 (5); GMM BIC improves strongly to k=4 (−15,702) and little after (k=5: −15,812); ARI with workday peaks at k=4. k = 4 chosen (elbow + interpretability).
- PCA: 5 components (95% variance), fitted on 2010–2013 only, then fixed.
- Clusters (2010–2013): C0 winter workday (99.6% workdays, 59% winter, evening peak 18:00); C1 mild workday with daytime plateau (80% workdays, 0% winter); C2 hot-day workday with afternoon peak (76% workdays, 64% summer, peak 15:30); C3 weekend/holiday shape (4% workdays, midday dip, evening peak).
- Yearly shares: C3 18–24% (2010–14) → 46% (2020) → 66% (2025–26); C1 and C2 shrink to ~0–2%. **Share of workdays assigned to the weekend-shaped C3: 2% (2014), 10% (2019), 25% (2020), 50% (2025)** — rooftop PV gives workdays a midday dip.
- Days unlike any 2010–13 type (distance > 2010–13 q95): 7% (2014) → 50% (2018) → 98% (2023).
- Yearly Spearman correlation with frozen MLP MAPE (2015–2026): C3 share 0.97, novelty 0.99 (shared trend, not causal proof). Frozen error grows most on C3 days (1.6% in 2015 → 11.8% in 2025).
- Same representation and reference period as the input detector (both are distances from the 2010–13 shape space).

## 8. Stage 3 and the package
- Package `submission_mlp/`: 5-seed MLP trained on 2022-11-01 … 2026-10-31 (1461 days), 55 epochs; artifacts 3.6 MB; `train.py` 7.8 min on this machine (while other jobs ran).
- Feature code shared by train.py and model.py; equal to the experiment features to 5e-13.
- `check_submission.py`: 14/14 days ok, init 0.1–0.3 s, predict 1.1–2.5 s; in-sample MAPE 1.81 (baseline 7.12).
- Out-of-sample end-to-end check (train to 2024-12-31 in Python 3.11, predict 134 days of 2025–26 one at a time): 3.316% vs 3.284% for the experiment (Python 3.13); forecasts differ by 0.33% on average.
- Fallback: any failure in predict returns the yesterday baseline for that output instead of a failed day.
- Price direction is a placeholder (yesterday baseline) until the price model is merged.

## 9. Live-week forecast (7–13 Dec 2026)
Files: `results/demand/mlp_ridge/liveweek/weeks.csv`; figure `results/demand/mlp_ridge/liveweek/fig_liveweek.png` (scripts `run_liveweek.py`, `plot_liveweek.py`).

- Replay of the live situation for every year Y = 2015…2025: MLP (5 seeds, 55 epochs, same as the package) trained on the four years up to 31 Oct Y, then the Monday–Sunday weeks of December Y ending by 21 Dec are predicted. Distance from training end: 31–45 days (live week: 37 days, 7 Dec 2026 − 31 Oct 2026).
- Analogue week = the week starting on the Monday between 7 and 13 Dec (one per year):

| Year | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP MAPE | 1.55 | 2.25 | 1.98 | 1.96 | 2.42 | 2.44 | 2.90 | 3.52 | 3.15 | 2.81 | 3.26 |
| Baseline MAPE | 7.82 | 10.11 | 8.14 | 4.16 | 6.21 | 5.68 | 7.19 | 6.94 | 8.38 | 6.87 | 8.24 |
| I (%) | 80.2 | 77.8 | 75.7 | 53.0 | 61.1 | 57.1 | 59.7 | 49.2 | 62.4 | 59.1 | 60.5 |

- All 23 December weeks: MLP MAPE median 2.42 [1.41, 4.77]; I median 62.4 [49.2, 80.2]. Analogue weeks 2021–2025: MAPE median 3.15 [2.81, 3.52], I median 59.7 [49.2, 62.4].
- Trend of analogue-week MAPE: +0.16 pp per year; linear extrapolation to 2026: **3.5 ± 0.6 %** (± 2 residual s.d., i.e. roughly 2.9–4.2 %).
- I has no trend after 2018 (49–62 %): the baseline gets harder at about the same rate as the model (more PV variability from day to day). Expected live I ≈ 50–60 %.
- Worst December week: 1–7 Dec 2025, MLP 4.77 but baseline 11.52 (I 58.6): a hard week for the model is also hard for the baseline, so I is more stable than MAPE.
- 7–13 Dec 2026 contains no NSW public holiday and is before the 22 Dec Christmas flag, so the holiday and Christmas features stay off.
- Caveats: the extrapolation assumes the drift continues at the same pace; the replay uses reanalysis weather (see section 10); 11 analogue weeks only.

## 10. Caveats to state in the report
- Weather is reanalysis treated as known: operational errors would be higher (weather-forecast error, especially temperature and cloud/solar).
- Recent-period numbers were viewed repeatedly; ensemble weights and Stage 1 choices used 2014 only.
- v2 detectors are a second iteration designed after seeing v1.
- MLP, Ridge, LightGBM and N-BEATSx are all from our own runs under one protocol (LightGBM identical to the teammate's; N-BEATSx within ±0.2 per year of the teammate's seed 42). GAM/TFT numbers come from the teammate's runs; TCN/LSTM use the older features.
- Seed averaging differs (MLP 5 seeds, N-BEATSx 3 seeds, Ridge/LightGBM deterministic); single-seed numbers are in `summary.csv` / `summary_nbeats.csv`.
- Bootstrap intervals assume independent days.
- Lockdown dates must be checked against official sources before citing.
