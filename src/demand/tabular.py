"""Half-hour features shared by the pilot and the full time splits."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def features(df, variant="base"):
    start = df.index - pd.Timedelta(minutes=30)
    x = pd.DataFrame(index=df.index)
    x["slot"] = start.hour * 2 + start.minute // 30
    x["weekday"] = start.dayofweek
    angle = 2 * np.pi * (start.dayofyear - 1) / 365.25
    x["season_sin"], x["season_cos"] = np.sin(angle), np.cos(angle)
    x["temp"] = df.syd_temp
    x["prior_temp"] = df.syd_temp.shift(48)
    for days in (1, 2, 7):
        x[f"lag{days}"] = df.nsw_demand.shift(48 * days)
    if variant == "no_target_temp":
        x = x.drop(columns="temp")
    if variant == "seven_lags":
        for days in (3, 4, 5, 6):
            x[f"lag{days}"] = df.nsw_demand.shift(48 * days)
    if variant == "prior_extrema":
        market_days = start.normalize()
        prior = df.syd_temp.groupby(market_days).agg(["min", "max"]).shift()
        for stat in ("min", "max"):
            x[f"prior_{stat}"] = market_days.map(prior[stat])
    if variant in ("humidity", "cloud", "solar", "cloud_solar"):
        cols = {"humidity": ["rh"], "cloud": ["cloud"], "solar": ["solar"],
                "cloud_solar": ["cloud", "solar"]}[variant]
        for variable in cols:
            x[variable] = df[f"syd_{variable}"]
    if variant in ("mel_temp", "bne_temp"):
        x[variant] = df[variant]
    return x


def nsw_holidays(years):
    """NSW public holidays from the course-provided holidays package.

    "Bank Holiday" applies to banks only, not a general day off, so it is excluded."""
    import holidays
    table = holidays.Australia(subdiv="NSW", years=years)
    return {pd.Timestamp(day) for day, name in table.items() if "Bank Holiday" not in name}


def add_holidays(x, level):
    """Calendar flags added on top of a model's feature variant.

    level 1: hol_today; level 2: + hol_lag1, hol_lag7; level 3: + xmas_period
    (24 Dec - 2 Jan days that are not public holidays). Only the calendar is used."""
    if level == 0:
        return x
    x = x.copy()
    day = (x.index - pd.Timedelta(minutes=30)).normalize()
    known = nsw_holidays(range(day.year.min() - 1, day.year.max() + 2))
    def flag(offset_days):
        shifted = day - pd.Timedelta(days=offset_days)
        return np.fromiter((d in known for d in shifted), bool, len(shifted)).astype(float)
    x["hol_today"] = flag(0)
    if level >= 2:
        x["hol_lag1"], x["hol_lag7"] = flag(1), flag(7)
    if level >= 3:
        in_window = ((day.month == 12) & (day.day >= 24)) | ((day.month == 1) & (day.day <= 2))
        x["xmas_period"] = (in_window & (x["hol_today"] == 0)).astype(float)
    return x


BINARY_PREFIXES = ("hol_", "xmas_")


def tree_inputs(x):
    # Fixed calendar vocabulary also supports inference on a single weekday.
    return pd.get_dummies(x.assign(weekday=pd.Categorical(x.weekday, categories=range(7))),
                          columns=["weekday"], dtype=float)


def lightgbm_model(params=None):
    from lightgbm import LGBMRegressor
    options = dict(n_estimators=350, learning_rate=0.04, num_leaves=31,
                   min_child_samples=48, random_state=42, n_jobs=2, verbosity=-1)
    options.update(params or {})
    return LGBMRegressor(**options)


def forecast(name, x, y, train, test, params=None):
    params = params or {}
    train = train & x.notna().all(axis=1)
    if name == "gam":
        predictions = pd.Series(index=x.index[test], dtype=float)
        for slot in range(48):
            rows = x.slot == slot
            smooth = [col for col in x if col not in
                      {"slot", "weekday", "season_sin", "season_cos"}
                      and not col.startswith(("lag",) + BINARY_PREFIXES)]
            # Binary calendar flags enter linearly (a per-slot offset), not through splines.
            linear = [col for col in x if col.startswith(("lag", "season_") + BINARY_PREFIXES)]
            terms = ColumnTransformer([
                ("weather", SplineTransformer(n_knots=params.get("n_knots", 5), include_bias=False), smooth),
                ("weekday", OneHotEncoder(), ["weekday"]),
                ("linear", StandardScaler(), linear),
            ])
            gam = make_pipeline(terms, Ridge(alpha=params.get("alpha", 1)))
            gam.fit(x.loc[train & rows], y.loc[train & rows])
            predictions.loc[x.index[test & rows]] = gam.predict(x.loc[test & rows])
        return predictions.to_numpy()
    if name != "lightgbm":
        raise ValueError(name)
    model = lightgbm_model(params)
    inputs = tree_inputs(x)
    model.fit(inputs.loc[train], y.loc[train])
    return np.asarray(model.predict(inputs.loc[test])).ravel()
