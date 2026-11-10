"""Repeat the original 2014 pilot. Run from the repo root: python -m experiments.demand.pilot.pilot_gam"""
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.demand import ROOT
from src.demand import tabular as demand_tabular


def main():
    path = ROOT / "data/nem_weather_halfhourly_2010_2026.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    years = (df.index - pd.Timedelta(minutes=30)).year
    x = demand_tabular.features(df)
    test = years == 2014
    predictions = {"Yesterday": x.loc[test, "lag1"], "Last week": x.loc[test, "lag7"]}
    for name in ["gam"]:
        predictions[name] = demand_tabular.forecast(name, x, df.nsw_demand, years < 2014, test)
    for name, forecast in predictions.items():
        error = np.abs(df.nsw_demand[test] - forecast)
        print(f"{name:16s} MAPE {100 * (error / df.nsw_demand[test]).mean():.3f}%  MAE {error.mean():.1f} MW")


if __name__ == "__main__":
    main()
