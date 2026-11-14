"""COMP9417 price-direction Logistic Regression component.

Prediction for each half-hour of day T uses market data through T-1
and the weather of day T (explicitly permitted by the specification).
"""
import numpy as np
import pandas as pd

WEATHER_COLS = [f"{city}_{var}" for city in ("syd", "mel", "bne")
                for var in ("temp", "rh", "cloud", "solar")]
MARKET_COLS = ["nsw_price", "nsw_demand", "vic_price", "qld_price"]
FEATURE_NAMES = [
    "sin_slot", "cos_slot", "sin_weekday", "cos_weekday", "is_weekend",
    "yesterday_nsw_price_log", "last_week_nsw_price_log",
    "yesterday_nsw_price_minus_yesterday_mean_log",
    "last_week_nsw_price_minus_yesterday_mean_log",
    "yesterday_nsw_demand", "last_week_nsw_demand",
    "yesterday_mean_nsw_price_log", "last_week_mean_nsw_price_log",
    "last_7_days_same_slot_price_mean_log",
    "yesterday_vic_price_log", "yesterday_qld_price_log",
] + WEATHER_COLS


def signed_log(x):
    """Preserve the sign of negative electricity prices while taming spikes."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def make_price_features(market_last_7_days, weather_next, target_day):
    """Return (48, n_features); input is exactly the 7 COMPLETE previous days.

    market_last_7_days: numpy array (7, 48, 4) oldest to newest, columns MARKET_COLS.
    weather_next: numpy array (48, 12), columns WEATHER_COLS.
    target_day: market day of the forecast, not raw end-of-interval timestamp.
    """
    market = np.asarray(market_last_7_days, dtype=float)
    weather = np.asarray(weather_next, dtype=float)
    if market.shape != (7, 48, 4) or weather.shape != (48, 12):
        raise ValueError(f"Expected (7,48,4) and (48,12), got {market.shape}, {weather.shape}")
    price7 = market[:, :, 0]
    prev_price = price7[-1]
    week_price = price7[0]
    prev_mean = prev_price.mean()
    week_mean = week_price.mean()
    slot = np.arange(48)
    weekday = pd.Timestamp(target_day).dayofweek

    columns = [
        np.sin(2 * np.pi * slot / 48),
        np.cos(2 * np.pi * slot / 48),
        np.full(48, np.sin(2 * np.pi * weekday / 7)),
        np.full(48, np.cos(2 * np.pi * weekday / 7)),
        np.full(48, float(weekday >= 5)),
        signed_log(prev_price),
        signed_log(week_price),
        signed_log(prev_price - prev_mean),
        signed_log(week_price - prev_mean),
        market[-1, :, 1],
        market[0, :, 1],
        np.full(48, signed_log(prev_mean)),
        np.full(48, signed_log(week_mean)),
        signed_log(price7.mean(axis=0)),
        signed_log(market[-1, :, 2]),
        signed_log(market[-1, :, 3]),
    ]
    columns.extend([weather[:, i] for i in range(weather.shape[1])])
    features = np.column_stack(columns)
    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite price-direction features")
    return pd.DataFrame(features, columns=FEATURE_NAMES)


def features_from_live_inputs(history, weather_next):
    """Create features for Forecaster.predict without touching future prices."""
    recent = history.iloc[-7 * 48:]
    if len(recent) != 7 * 48 or len(weather_next) != 48:
        raise ValueError("Need 7 complete history days and 48 forecast weather rows")
    market = recent[MARKET_COLS].to_numpy(dtype=float).reshape(7, 48, 4)
    weather = weather_next[WEATHER_COLS].to_numpy(dtype=float)
    target_day = pd.Timestamp(weather_next.index[0]) - pd.Timedelta(minutes=30)
    return make_price_features(market, weather, target_day.normalize())


def make_historical_table(df, start='2010-01-01', end='2014-12-31'):
    """Build leakage-free samples for training and validation.

    Only target days between start/end are returned. Seven preceding days are
    needed to construct each target day's features.
    """
    df = df.sort_index()
    market_days = (df.index - pd.Timedelta(minutes=30)).normalize()
    market_col_idx = MARKET_COLS
    day_data = {}
    for day, group in df.groupby(market_days, sort=True):
        if len(group) == 48:
            day_data[pd.Timestamp(day)] = group
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    X_blocks, y_blocks, date_blocks = [], [], []
    for day in sorted(day_data):
        if day < start or day > end:
            continue
        previous_days = [day - pd.Timedelta(days=k) for k in range(7, 0, -1)]
        if any(d not in day_data for d in previous_days):
            continue
        prev = [day_data[d] for d in previous_days]
        market = np.stack([p[market_col_idx].to_numpy(dtype=float) for p in prev])
        weather = day_data[day][WEATHER_COLS].to_numpy(dtype=float)
        features = make_price_features(market, weather, day)
        prev_mean = prev[-1]['nsw_price'].mean()
        labels = (day_data[day]['nsw_price'].to_numpy() > prev_mean).astype(int)
        X_blocks.append(features)
        y_blocks.append(labels)
        date_blocks.extend([day] * 48)
    if not X_blocks:
        raise ValueError('No complete training examples in given date range')
    return (pd.concat(X_blocks, ignore_index=True),
            np.concatenate(y_blocks),
            pd.DatetimeIndex(date_blocks))
