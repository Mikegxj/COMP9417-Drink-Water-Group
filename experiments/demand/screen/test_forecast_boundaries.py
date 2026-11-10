"""Regression checks against future-target leakage and misaligned daily windows."""
import unittest

import numpy as np
import pandas as pd
import torch

from src.demand.compact_neural import CausalBlock, build_model, day_tensors
from src.demand.tabular import features
from experiments.demand import ROOT
from harness import dayahead


class ForecastBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = dayahead.load_dataset(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
        cls.days = dayahead.market_day(cls.df.index)

    def test_future_targets_do_not_change_inputs_or_scaling(self):
        original = self.df.loc[self.days.year <= 2014].copy()
        days = dayahead.market_day(original.index)
        changed = original.copy()
        changed.loc[days.year == 2014, "nsw_demand"] *= 4
        first = day_tensors(original, days, 2014)
        second = day_tensors(changed, days, 2014)
        count = first[3]
        self.assertEqual(first[4:], second[4:])
        torch.testing.assert_close(first[0][:count+1], second[0][:count+1])
        torch.testing.assert_close(first[1], second[1])
        # Later target days may use realized loads from earlier evaluation days.

    def test_leap_day_and_midnight_alignment(self):
        h, _, target, _, mean, scale = day_tensors(self.df, self.days, 2014)
        for date in ("2014-01-01", "2016-02-29", "2026-10-31"):
            history, weather = dayahead.make_inputs(self.df, date)
            i = int((self.days.unique() < pd.Timestamp(date)).sum()) - 7
            np.testing.assert_allclose(h[i].numpy()*scale+mean, history.nsw_demand.iloc[-336:], atol=.002)
            np.testing.assert_allclose(target[i].numpy()*scale+mean,
                                       self.df.loc[weather.index, "nsw_demand"], atol=.002)

    def test_tabular_features_match_daily_information(self):
        day = pd.Timestamp("2014-07-01")
        history, weather = dayahead.make_inputs(self.df, day)
        only_available = pd.concat([history, weather])
        for variant in ("base", "prior_extrema", "seven_lags", "cloud", "humidity"):
            full = features(self.df, variant).loc[weather.index]
            daily = features(only_available, variant).loc[weather.index]
            pd.testing.assert_frame_equal(full, daily)

    def test_rolling_window_scalers_use_only_training_period(self):
        start = pd.Timestamp("2021-01-01")
        mask = (self.days >= start-pd.Timedelta(days=7)) & (self.days.year < 2027)
        df, days = self.df.loc[mask].copy(), self.days[mask]
        before = day_tensors(df, days, 2025, start)
        df.loc[(days < start) | (days.year >= 2025), "nsw_demand"] *= 2
        after = day_tensors(df, days, 2025, start)
        self.assertEqual(before[4:], after[4:])

    def test_causal_convolution(self):
        torch.manual_seed(42)
        model = CausalBlock(4).eval()
        x = torch.randn(2, 16, 336)
        changed = x.clone()
        changed[:, :, 200:] = 100
        torch.testing.assert_close(model(x)[:, :, :200], model(changed)[:, :, :200])

    def test_network_shapes_and_gradients(self):
        torch.set_num_threads(2)
        for name, variant in [("nbeats","base"), ("nbeats","calendar_temp"),
                              ("tft","base"), ("tcn","base"), ("lstm","base")]:
            model = build_model(name, 7, variant)
            pred = model(torch.randn(2, 336), torch.randn(2, 48, 7))
            self.assertEqual(tuple(pred.shape), (2, 48))
            pred.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
