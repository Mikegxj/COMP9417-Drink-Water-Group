"""Rebuild and use the candidate demand component pending the team comparison."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.demand.tabular import features, lightgbm_model, tree_inputs
from experiments.demand import ROOT
from harness import dayahead


def train(df, artifact):
    days = dayahead.market_day(df.index)
    end = days.max() + pd.DateOffset(days=1)
    x = features(df, "cloud")
    rows = (days >= end - pd.DateOffset(years=4)) & x.notna().all(axis=1)
    model = lightgbm_model().fit(tree_inputs(x.loc[rows]), df.nsw_demand.loc[rows])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact)


class DemandForecaster:
    def __init__(self, artifact):
        self.model = joblib.load(artifact)

    def predict(self, history, weather_next):
        tail = history.iloc[-336:]
        index = pd.date_range(tail.index[-1] + pd.Timedelta(minutes=30), periods=48, freq="30min")
        if len(tail) != 336 or not weather_next.index.equals(index):
            raise ValueError("Demand forecast requires a complete week followed by the next 48 slots")
        inputs = pd.concat([tail, weather_next])
        x = tree_inputs(features(inputs, "cloud").iloc[-48:])
        forecast = self.model.predict(x)
        if not np.isfinite(forecast).all():
            raise ValueError("Non-finite demand prediction")
        return pd.Series(np.clip(forecast, 0, 20000), index=index, name="nsw_demand_mw")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    parser.add_argument("--artifact", type=Path, default=ROOT / "results/demand/artifacts/demand.joblib")
    args = parser.parse_args()
    train(dayahead.load_dataset(args.data), args.artifact)
