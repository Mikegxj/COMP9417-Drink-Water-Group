from pathlib import Path
import sys

import holidays
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = PROJECT_ROOT / "harness"

if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import dayahead as da


MARKET_COLUMNS = [
    "nsw_demand",
    "nsw_price",
    "vic_demand",
    "vic_price",
    "qld_demand",
    "qld_price",
]


def build_dayahead_table(csv_path):
    """
    Build one row for each half-hour being predicted.

    A row belonging to day T may use:
    - market data from T-1 and earlier
    - weather for day T
    - calendar information for day T

    It must not use market data from day T as model inputs.
    """
    df = da.load_dataset(str(csv_path)).copy()

    # Each market day runs from 00:30 to the following 00:00.
    df["day"] = da.market_day(df.index)

    interval_start = df.index - pd.Timedelta(minutes=30)
    df["period"] = (
        interval_start.hour * 2
        + interval_start.minute // 30
    )

    table = pd.DataFrame(index=df.index)
    table["day"] = df["day"]
    table["period"] = df["period"]
    table["year"] = pd.DatetimeIndex(table["day"]).year

    # Targets
    previous_day_mean_price = (
        df.groupby("day")["nsw_price"]
        .transform("mean")
        .shift(48)
    )

    table["y_demand"] = df["nsw_demand"]
    table["y_up"] = (
        df["nsw_price"] > previous_day_mean_price
    ).astype("int8")

    # Market data from yesterday and last week.
    for column in MARKET_COLUMNS:
        table[f"{column}_lag1"] = df[column].shift(48)
        table[f"{column}_lag7"] = df[column].shift(48 * 7)

    # The two official price-direction baselines.
    table["up_lag1"] = (
        df["nsw_price"].shift(48) > previous_day_mean_price
    ).astype("int8")

    table["up_lag7"] = (
        df["nsw_price"].shift(48 * 7) > previous_day_mean_price
    ).astype("int8")

    # Previous-day summaries, known at forecast time.
    daily_summary = df.groupby("day").agg(
        previous_day_mean_price=("nsw_price", "mean"),
        previous_day_std_price=("nsw_price", "std"),
        previous_day_min_price=("nsw_price", "min"),
        previous_day_max_price=("nsw_price", "max"),
        previous_day_mean_demand=("nsw_demand", "mean"),
        previous_day_max_demand=("nsw_demand", "max"),
    )

    previous_summary = daily_summary.shift(1)

    for column in previous_summary.columns:
        table[column] = table["day"].map(previous_summary[column])

    # Forecast-day weather is allowed by the specification.
    for column in da.WEATHER_COLS:
        table[column] = df[column]

    # Calendar features for the forecast day.
    forecast_day = pd.to_datetime(table["day"])

    table["day_of_week"] = forecast_day.dt.dayofweek
    table["month"] = forecast_day.dt.month
    table["day_of_year"] = forecast_day.dt.dayofyear
    table["is_weekend"] = (
        table["day_of_week"] >= 5
    ).astype("int8")

    nsw_holidays = holidays.Australia(
        subdiv="NSW",
        years=range(
            int(table["year"].min()),
            int(table["year"].max()) + 1,
        ),
    )

    table["is_nsw_holiday"] = forecast_day.dt.date.map(
        lambda date: int(date in nsw_holidays)
    )

    # Cyclic time features.
    table["period_sin"] = np.sin(
        2 * np.pi * table["period"] / 48
    )
    table["period_cos"] = np.cos(
        2 * np.pi * table["period"] / 48
    )
    table["day_of_week_sin"] = np.sin(
        2 * np.pi * table["day_of_week"] / 7
    )
    table["day_of_week_cos"] = np.cos(
        2 * np.pi * table["day_of_week"] / 7
    )
    table["day_of_year_sin"] = np.sin(
        2 * np.pi * table["day_of_year"] / 365.25
    )
    table["day_of_year_cos"] = np.cos(
        2 * np.pi * table["day_of_year"] / 365.25
    )

    # The first seven days do not have all lag features.
    required_lags = [
        f"{column}_lag7" for column in MARKET_COLUMNS
    ]

    table = table.dropna(subset=required_lags).copy()

    return table


def chronological_split(table):
    """Return the three periods required by the project."""
    train = table.loc[table["year"].between(2010, 2013)].copy()
    validation = table.loc[table["year"] == 2014].copy()
    simulation = table.loc[table["year"].between(2015, 2026)].copy()

    return train, validation, simulation