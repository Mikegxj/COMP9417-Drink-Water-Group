"""Online adaptation of the 2014-selected LightGBM with Sydney cloud cover."""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.demand import tabular as demand_tabular
from experiments.demand import ROOT
from harness import dayahead


def input_distances(df, days):
    # Relative day profiles separate changes of daily shape from demand level.
    profiles = df.nsw_demand.to_numpy().reshape(-1, 48)
    profiles = profiles / profiles.mean(axis=1, keepdims=True)
    reference = days.unique().year < 2014
    pca = PCA(n_components=0.95).fit(profiles[reference])
    coordinates = pca.transform(profiles)
    clusters = KMeans(n_clusters=4, random_state=42, n_init=10).fit(coordinates[reference])
    distances = clusters.transform(coordinates)
    return pd.DataFrame({"cluster": distances.argmin(1), "distance": distances.min(1)}, index=days.unique())


def simulate(df, days, x, schedule, threshold, calibration_errors):
    first = pd.Timestamp("2015-01-01")
    last = days[-1]
    output = []
    signals = []
    errors = deque(calibration_errors[-28:], maxlen=28)
    origin = first
    updates = 0
    while origin <= last:
        training = days < origin
        if updates:
            training &= days >= origin - pd.DateOffset(years=4)
        # Predictions are computed in bulk until a replacement is due. Every
        # row uses only its own available inputs. Future errors are never consulted.
        remaining = days >= origin
        prediction = demand_tabular.forecast("lightgbm", x, df.nsw_demand, training, remaining)
        segment = pd.Series(prediction, index=df.index[remaining])
        dates = days[remaining].unique()
        accepted_days = len(dates)
        observed = df.nsw_demand[remaining].to_numpy().reshape(-1, 48)
        daily_errors = (100 * abs(prediction.reshape(-1, 48) - observed) / observed).mean(axis=1)
        for i, day in enumerate(dates):
            errors.append(daily_errors[i])
            rolling = float(np.mean(errors))
            next_day = day + pd.Timedelta(days=1)
            trigger = (next_day == origin + pd.DateOffset(years=2) if schedule == "periodic"
                       else rolling > threshold and (next_day - origin).days >= 180)
            if trigger and updates < 6 and next_day <= last:
                accepted_days = i + 1
                signals.append(dict(observed=day, effective=next_day, rolling_mape=rolling))
                break
        output.append(segment.iloc[:48 * accepted_days])
        origin = dates[accepted_days - 1] + pd.Timedelta(days=1)
        updates += 1
    return pd.concat(output), signals


def main(output):
    df = dayahead.load_dataset(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    days = dayahead.market_day(df.index)
    x = demand_tabular.features(df, "cloud")
    calibration = days.year == 2014
    pred = demand_tabular.forecast("lightgbm", x, df.nsw_demand, days.year < 2014, calibration)
    errors = (100 * abs(df.nsw_demand[calibration] - pred) / df.nsw_demand[calibration])
    daily = errors.groupby(days[calibration]).mean()
    threshold = daily.rolling(28).mean().quantile(0.95)
    folder = output / "adaptation"
    folder.mkdir(parents=True, exist_ok=True)
    clusters = input_distances(df, days)
    distances = clusters.distance.rolling(28).mean()
    input_threshold = distances.loc["2014"].quantile(0.95)
    clusters["input_signal"] = distances > input_threshold
    clusters.to_csv(folder / "day_types.csv")
    summaries = []
    for schedule in ("periodic", "error_drift"):
        forecast, signals = simulate(df, days, x, schedule, threshold, daily.to_numpy())
        frame = pd.DataFrame({"actual": df.nsw_demand.loc[forecast.index], "forecast": forecast})
        market_days = dayahead.market_day(frame.index)
        frame.to_csv(folder / f"{schedule}_predictions.csv.gz")
        pd.DataFrame(signals, columns=["observed", "effective", "rolling_mape"]).to_csv(
            folder / f"{schedule}_retrains.csv", index=False)
        daily_ape = (100 * abs(frame.actual - frame.forecast) / frame.actual).groupby(market_days).mean()
        rolling = pd.concat([daily.tail(27), daily_ape]).rolling(28).mean().loc[daily_ape.index]
        pd.DataFrame({"rolling_mape": rolling, "error_signal": rolling > threshold,
                      "error_threshold": threshold, "input_threshold": input_threshold}).to_csv(
                          folder / f"{schedule}_signals.csv")
        for name, selection in [("selection", market_days.year < 2025),
                                ("recent_check", market_days.year >= 2025)]:
            selected = frame.loc[selection]
            summaries.append(dict(strategy=schedule, period=name, retrains_total=len(signals),
                                  mape=dayahead.mape(selected.actual, selected.forecast),
                                  mae=dayahead.mae(selected.actual, selected.forecast)))
    pd.DataFrame(summaries).to_csv(folder / "strategy_scores.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/demand")
    main(parser.parse_args().output)
