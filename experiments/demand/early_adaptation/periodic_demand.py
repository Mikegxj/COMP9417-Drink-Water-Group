"""Repeat the same two-year update schedule for all six Stage 1 configurations."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import tabular as demand_tabular
from experiments.demand import ROOT
from harness import dayahead

CONFIGS = {"lightgbm": "cloud", "gam": "prior_extrema",
           "nbeats": "calendar_temp", "tft": "base", "tcn": "base", "lstm": "base"}


def compare(output):
    data = dayahead.load_dataset(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    days = dayahead.market_day(data.index)
    folder = output / "periodic"
    folder.mkdir(parents=True, exist_ok=True)
    scores = []
    for name, variant in CONFIGS.items():
        pieces = []
        tabular = name in ("lightgbm", "gam")
        if tabular:
            x = demand_tabular.features(data, variant)
        for year in range(2015, 2026, 2):
            start = pd.Timestamp("2010-01-01" if year == 2015 else f"{year-4}-01-01")
            train = (days >= start) & (days.year < year)
            test = (days.year >= year) & (days.year < year + 2)
            if tabular:
                pred = demand_tabular.forecast(name, x, data.nsw_demand, train, test)
            else:
                from src.demand.compact_neural import forecast
                # Retain pre-window history for the first training target's lag inputs.
                use = (days >= start - pd.Timedelta(days=7)) & (days.year < year + 2)
                pred = forecast(name, data.loc[use], days[use], year, 16, "cpu", variant, start)
            pieces.append(pd.Series(pred, index=data.index[test]))
        predicted = pd.concat(pieces)
        actual = data.nsw_demand[days.year >= 2015]
        if not predicted.index.equals(actual.index) or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid periodic forecasts for {name}")
        frame = pd.DataFrame({"actual": actual, "forecast": predicted})
        frame.to_csv(folder / f"{name}_predictions.csv.gz")
        years = dayahead.market_day(frame.index).year
        for period, mask in [("selection_2015_2024", years < 2025),
                             ("check_2025_2026_10", years >= 2025),
                             ("overall", years >= 2015)]:
            subset = frame.loc[mask]
            scores.append(dict(model=name, variant=variant, period=period, n=len(subset),
                               mape=dayahead.mape(subset.actual, subset.forecast),
                               mae=dayahead.mae(subset.actual, subset.forecast)))
        print(name, [round(row["mape"], 3) for row in scores[-3:]], flush=True)
    pd.DataFrame(scores).to_csv(folder / "comparison.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/demand")
    compare(parser.parse_args().output)
