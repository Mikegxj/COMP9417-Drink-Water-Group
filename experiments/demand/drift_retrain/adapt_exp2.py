"""Stage 2 drift-triggered retraining for the selected demand model (N-BEATSx, exp2 features).

Detector: rolling 28-day mean of daily MAPE, computed only from days already observed.
Threshold: quantile q of that rolling mean on 2014, from a model trained on 2010-2013 with
the same refit epochs and seed (no data after 2014). The main setting q=0.95 was fixed
before running; other q values are a sensitivity analysis of the retrain/error trade-off.

Rules (spec Stage 2): a signal at the end of day D retrains with data up to D only and the new
model is used from D+1. Retraining uses the most recent four years, like the periodic
strategy. At most six retrains from 2015-01 to 2026-10; at least 180 days between retrains so a
single bad spell cannot use up the budget. The initial 2010-2014 fit is shared with the frozen
backtest (same checkpoint), so all strategies agree until their first retrain.
"""
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.demand.backtest.formal_backtest import neural_refit
from harness import dayahead

from experiments.demand import ROOT  # noqa: E402
FIRST, LAST = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-10-31")
MAX_RETRAINS, MIN_GAP_DAYS, WINDOW = 6, 180, 28


def daily_mape(actual, forecast):
    days = dayahead.market_day(actual.index)
    return (100 * (actual - forecast).abs() / actual).groupby(days).mean()


def calibrate(df, selected, folder):
    folder.mkdir(parents=True, exist_ok=True)
    pred = neural_refit("nbeats", selected["config"], selected["refit_epochs"], df,
                        pd.Timestamp("2010-01-01"), pd.Timestamp("2013-12-31"),
                        pd.Timestamp("2014-12-31"), folder, seed=selected["refit_seed"])
    return daily_mape(df.nsw_demand.loc[pred.index], pred)


def simulate(df, selected, q, calibration, initial_folder, folder):
    threshold = float(calibration.rolling(WINDOW).mean().dropna().quantile(q))
    errors = deque(calibration.to_numpy()[-(WINDOW - 1):], maxlen=WINDOW)
    origin, retrains, segments, events = FIRST, 0, [], []
    while origin <= LAST:
        if origin == FIRST:
            start, fit_folder = pd.Timestamp("2010-01-01"), initial_folder
        else:
            start = origin - pd.DateOffset(years=4)
            fit_folder = folder / f"retrain_{retrains}_{origin.date()}"
        fit_folder.mkdir(parents=True, exist_ok=True)
        pred = neural_refit("nbeats", selected["config"], selected["refit_epochs"], df, start,
                            origin - pd.Timedelta(days=1), LAST, fit_folder, seed=selected["refit_seed"])
        # Bulk prediction is only for speed: each day's forecast uses its own inputs, and
        # each day's error is read only after that day has been observed (loop order).
        daily = daily_mape(df.nsw_demand.loc[pred.index], pred)
        keep_until = LAST
        for day, error in daily.items():
            errors.append(error)
            rolling = float(np.mean(errors))
            effective = day + pd.Timedelta(days=1)
            if (rolling > threshold and (effective - origin).days >= MIN_GAP_DAYS
                    and retrains < MAX_RETRAINS and effective <= LAST):
                keep_until = day
                events.append(dict(signal_day=str(day.date()), effective=str(effective.date()),
                                   rolling_mape=rolling, threshold=threshold,
                                   train_start=str((effective - pd.DateOffset(years=4)).date()),
                                   train_end=str(day.date())))
                break
        days = dayahead.market_day(pred.index)
        segments.append(pred[days <= keep_until])
        if keep_until == LAST:
            break
        origin, retrains = keep_until + pd.Timedelta(days=1), retrains + 1
    forecast = pd.concat(segments)
    expected = df.index[(dayahead.market_day(df.index) >= FIRST) & (dayahead.market_day(df.index) <= LAST)]
    if not forecast.index.equals(expected) or not np.isfinite(forecast.to_numpy()).all():
        raise ValueError("Drift simulation does not cover every forecast half-hour exactly once")
    return forecast, events, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/demand/formal_exp2")
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.95, 0.90, 0.99])
    args = parser.parse_args()
    selected = json.loads((args.root / "nbeats/selected.json").read_text())
    df = dayahead.load_dataset(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    out = args.root / "adaptation" / "nbeats"
    out.mkdir(parents=True, exist_ok=True)
    calibration = calibrate(df, selected, out / "calibration_2010_2013")
    calibration.rename("daily_mape").to_csv(out / "calibration_2014_daily_mape.csv")
    summary = []
    for q in args.quantiles:
        folder = out / f"error_drift_q{q:.2f}"
        forecast, events, threshold = simulate(df, selected, q, calibration,
                                               args.root / "backtest/nbeats/2015", folder)
        frame = pd.DataFrame(dict(actual=df.nsw_demand.loc[forecast.index], forecast=forecast))
        frame.index.name = "timestamp"
        frame.to_csv(folder / "predictions.csv.gz")
        pd.DataFrame(events, columns=["signal_day", "effective", "rolling_mape", "threshold",
                                      "train_start", "train_end"]).to_csv(folder / "retrains.csv", index=False)
        summary.append(dict(quantile=q, threshold=threshold, retrains=len(events)))
        print("q", q, "threshold", round(threshold, 3), "retrains", len(events),
              "mape", round(dayahead.mape(frame.actual, frame.forecast), 4), flush=True)
    pd.DataFrame(summary).to_csv(out / "drift_settings.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(dict(
        detector=f"rolling {WINDOW}-day mean daily MAPE > quantile q of its 2014 values",
        calibration_model="trained 2010-2013, same config/epochs/seed", main_quantile=0.95,
        sensitivity_quantiles=args.quantiles, max_retrains=MAX_RETRAINS, min_gap_days=MIN_GAP_DAYS,
        retrain_window="last 4 years up to the signal day", initial_fit="shared with frozen backtest"),
        indent=2))


if __name__ == "__main__":
    main()
