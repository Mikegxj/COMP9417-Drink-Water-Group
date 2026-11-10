"""INTEGRATION EXAMPLE ONLY: replace demand baseline with teammate's demand model.

This file gives a valid Forecaster interface for testing the price Logistic Regression.
For final submission, merge the classifier into your group's common model.py.
"""
import os
import joblib
import numpy as np
import pandas as pd
from price_logistic import features_from_live_inputs


class Forecaster:
    def __init__(self, artifact_dir: str):
        self.price_classifier = joblib.load(os.path.join(artifact_dir, 'price_lr.joblib'))

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        X = features_from_live_inputs(history, weather_next)
        price_up = self.price_classifier.predict(X).astype(int)
        # TEMPORARY demand baseline, NOT your team's final regression model.
        demand = history['nsw_demand'].iloc[-48:].to_numpy(dtype=float)
        return pd.DataFrame({'nsw_demand_mw': np.clip(demand, 0, 20000),
                             'nsw_price_up': price_up}, index=weather_next.index)
