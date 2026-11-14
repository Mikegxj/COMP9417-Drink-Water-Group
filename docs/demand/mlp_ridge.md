# Demand: MLP (selected model) and Ridge — code, results and package

This adds the selected demand model (5-seed MLP), the Ridge + spline model, re-runs of the
teammate's LightGBM and N-BEATSx under one shared protocol, drift detection, adaptation,
day types, data analysis and the live-week evidence. All numbers, tables and caveats for
writing the report are in [`mlp_ridge_results.md`](mlp_ridge_results.md).

| What | Where |
|---|---|
| Code (flat scripts, run from this folder) | `experiments/demand/mlp_ridge/` |
| Result tables (CSV) and figures (PNG) | `results/demand/mlp_ridge/` |
| Model package (MLP demand; price still a placeholder) | `submission_mlp/` (proposed to replace `submission/` once the group agrees) |
| Numbers for the report | `docs/demand/mlp_ridge_results.md` |

## Protocol (same as `docs/demand/formal_experiment.md`)
- Day D = intervals D 00:30 … D+1 00:00; forecast at the end of D for D+1; demand features use days ≤ D only, weather of D+1 and calendar (NSW holidays from `holidays`).
- Stage 1: fit on 2010–2013, select on 2014 (neural early stopping on an inner 2010–2012 / 2013 split, then refit on 2010–2013).
- Stage 2: frozen = refit on 2010–2014; periodic = refits on 1 Jan 2017/2019/2021/2023/2025, each on the previous four years (5 retrains) — the shared protocol used to compare all models.
- Window length (`run_window_length.py`, same design and rule as `docs/demand/experiment_log.md` §20): for the MLP a 6-year window is significantly better on 2017–24 and not worse in 2025–26, so the **selected MLP strategy is periodic retraining on the previous six years** (37 epochs = the 4-year gradient-update count).
- Stage 3 (package): the last six years, 2020-11-01 … 2026-10-31, 37 epochs (6th retrain).
- Both baselines (`harness/dayahead.baseline`, lag 1 and lag 7) and both metrics (MAPE, MAE) for every period.

## Environment
Everything runs in the course environment (`requirements.txt`, Python 3.11, PyTorch CPU).
The experiments were produced with Python 3.13 (scikit-learn 1.8, PyTorch 2.14); re-running
`run_metrics_all.py` and `run_eda.py` in the 3.11 environment gives identical tables, all modules
import in 3.11, and the package trained in 3.11 (to 2024-12-31, 6-year window)
gives 3.254 % vs 3.241 % MAPE on 134 out-of-sample days of 2025–26 (`check_package_oos.py`).

Paths are set in `features.py` (`HARNESS`, `DATA`, `OUT_ROOT`, `PACKAGE`): inside this repository
they point to `harness/`, `data/`, `results/demand/mlp_ridge/` and `submission_mlp/`. Docstrings that
mention `outputs/...` mean `results/demand/mlp_ridge/...` here.

## How to run (from `experiments/demand/mlp_ridge/`)
Library modules: `features.py` (data, calendar, feature matrices), `models.py` (Ridge, MLP,
LightGBM), `nbeats.py` (teammate's N-BEATSx configuration in plain PyTorch), `drift.py` (detectors).

The prediction caches (`*.npz`, up to ~100 MB) are not committed (`results/demand/mlp_ridge/.gitignore`); steps 1–3 and 7b rebuild them. The selected strategy's forecasts are committed as a plain CSV (step 7c).

| Step | Script | Writes (under `results/demand/mlp_ridge/`) |
|---|---|---|
| 1 | `run_protocol.py` | `protocol/predictions.npz` — Stage 1 + frozen + periodic fits of MLP, Ridge, LightGBM and MLP variants |
| 2 | `run_nbeats.py` | `protocol/predictions_nbeats.npz`, `nbeats_reproduction_check.csv` (≈ 11 min, 4 workers) |
| 3 | `run_cross.py` | `protocol/predictions_cross.npz`, `cross_summary.csv` (features × model family) |
| 3b | `run_nbeats_wx.py` | `protocol/predictions_nbeats_wx.npz`, `nbeats_wx_summary.csv`, `nbeats_wx_by_year.csv`, `nbeats_wx_paired.csv` (N-BEATSx with cloud & solar added, ≈ 6 min) |
| 4 | `analyse_protocol.py`, `analyse_nbeats.py` | `protocol/summary*.csv`, ensembles, paired bootstrap differences, weekly improvement I |
| 5 | `run_metrics_all.py` | `protocol/metrics_summary.csv`, `metrics_by_year.csv`, `metrics_by_month.csv` (MAPE + MAE, both baselines, both monthly 95 % intervals), `monthly_ci_width_summary.csv` |
| 6 | `plot_compare.py`, `plot_pred_obs.py`, `plot_monthly_ci.py` | `protocol/fig_four_models_annual.png`, `fig_pred_vs_obs.png`, `fig_monthly_ci_mlp.png` |
| 7 | `run_drift.py`, `run_drift_v2.py`, `plot_drift.py` | `drift/*.csv`, `drift/fig_*.png` (detectors, triggered vs periodic retraining) |
| 8 | `run_daytypes.py --k 4` | `daytypes/*.csv`, `daytypes/fig_daytypes.png` |
| 7b | `run_window_length.py` | `window_length/comparison.csv`, `bootstrap_vs_4years.csv`, `by_year_vs_4years.csv` (2/3/4/6 years and all history; ≈ 5 min, 8 workers) — rerun `run_metrics_all.py` and `plot_monthly_ci.py` afterwards |
| 7c | `export_selected_predictions.py` | `window_length/predictions_mlp_6years.csv` — half-hourly forecasts of the selected strategy (frozen 2010–14 fit for 2015–16, 6-year refits from 2017) with observed demand, 2015-01-01 … 2026-10-31 (MAPE 2.401 %, as in `metrics_summary.csv`) |
| 9 | `run_liveweek.py 6` (`run_liveweek.py 4` for the old window), `plot_liveweek.py` | `liveweek/weeks_6years.csv` (`weeks_4years.csv`), `liveweek/fig_liveweek.png` (≈ 2 min, 6 workers) |
| 10 | `run_eda.py` | `eda/describe.csv`, `eda/summary_by_year.csv`, `eda/fig_eda_demand.png` |
| – | `run_trial.py`, `run_recent.py`, `run_ablation.py features\|holidays`, `plot_ablation.py` | early Stage 1 runs and feature/holiday ablations: `val2014*.csv`, `recent_*.csv`, `ablation_*.csv`, `fig_ablation_features.png` |
| – | `check_package_oos.py WORK_DIR` | end-to-end check of `submission_mlp/` in the 3.11 environment (printed) |

`protocol/teammate_exp2_*.csv` are copies of the teammate's own N-BEATSx/LightGBM/GAM/TFT results
(`results/demand/formal_exp2`), used by `analyse_protocol.py` and `run_nbeats.py` to check the reproductions.

## Package (`submission_mlp/`)
`submission/` (the N-BEATSx package) is left unchanged; `submission_mlp/` is the proposed replacement.
- `python submission_mlp/train.py data/nem_weather_halfhourly_2010_2026.csv` rebuilds `submission_mlp/artifacts/` from the last six years (2020-11-01 … 2026-10-31, 5 seeds × 37 epochs, ≈ 1–2 min on 4 CPU threads, fixed seeds).
- `check_submission.py` (project kit): all 14 days ok, init ≤ 0.1 s, predict ≈ 1 s, peak memory ≈ 0.3 GB, artifacts 3.6 MB.
- `predict` falls back to the same-half-hour-yesterday baseline if anything fails, so a bug cannot cost a whole day.
- **Price direction is a placeholder** (yesterday baseline) until the price classifier is merged into `model.py` / `train.py`.
