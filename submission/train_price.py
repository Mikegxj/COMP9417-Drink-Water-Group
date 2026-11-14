import argparse
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from model import MARKET, WEATHER, MODEL_FILE, PER_DAY, market_days, price_features


def train_price(csv_path, artifact_dir):
    frame = pd.read_csv(csv_path, usecols=["timestamp", *MARKET, *WEATHER],
                        parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    days = market_days(frame.index)
    # Leave seven additional market days for lagged inputs at the window boundary.
    end = days.max() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=2)
    frame = frame.loc[days >= start - pd.Timedelta(days=7)]
    days = market_days(frame.index)
    X = price_features(frame)
    price = frame["nsw_price"]
    previous_mean = price.groupby(days).transform("mean").shift(PER_DAY)
    keep = days >= start
    if price[keep].isna().any() or previous_mean[keep].isna().any():
        raise ValueError("Missing NSW price or comparison mean in the training window.")
    y = (price > previous_mean).astype("int8")
    model = XGBClassifier(n_estimators=300, max_depth=3, min_child_weight=5,
                          learning_rate=0.05, objective="binary:logistic",
                          eval_metric="logloss", tree_method="hist",
                          random_state=42, n_jobs=2)
    model.fit(X.loc[keep], y.loc[keep])
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / MODEL_FILE
    model.save_model(target)
    print(f"{start.date()} to {days.max().date()}: {sum(keep)} rows, {X.shape[1]} features")
    print(f"Saved {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    train_price(args.csv_path, Path(__file__).resolve().parent / "artifacts")
