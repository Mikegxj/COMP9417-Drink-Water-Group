"""Availability and maintainer-forward checks for the formal experiment suite."""
import unittest

import numpy as np
import pandas as pd
import torch

from src.demand.neural import DailyWindows,make_model
from experiments.demand import ROOT
from harness import dayahead


class FormalModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=dayahead.load_dataset(ROOT/"data/nem_weather_halfhourly_2010_2026.csv")
        cls.start=pd.Timestamp("2010-01-01")
        cls.end=pd.Timestamp("2013-12-31")
        cls.eval_end=pd.Timestamp("2014-12-31")
        torch.set_num_threads(2)

    def test_masks_and_horizon(self):
        for window in (336,480):
            config=dict(window=window,width=16,weather=True)
            data=DailyWindows(self.df,config,self.start,self.end,self.eval_end)
            first=int(data.eval_indices[0])
            history,weather=dayahead.make_inputs(self.df,"2014-01-01")
            np.testing.assert_allclose(data.history[first,:,0]*data.scale+data.mean,
                                       history.nsw_demand.iloc[-window:],atol=.002)
            self.assertEqual(len(data.eval_indices),365)
            np.testing.assert_allclose(data.actual[first],
                                       self.df.nsw_demand.loc[weather.index],atol=.002)
            x=data.batch(torch.tensor([first]))
            self.assertEqual(tuple(x["futr_exog"].shape),(1,window+48,7))
            self.assertNotIn("outsample_y",x)

    def test_future_targets_do_not_change_first_validation_forecast(self):
        config=dict(window=336,width=16,weather=True)
        original=DailyWindows(self.df,config,self.start,self.end,self.eval_end)
        df=self.df.copy()
        df.loc[dayahead.market_day(df.index).year>=2014,"nsw_demand"]*=4
        altered=DailyWindows(df,config,self.start,self.end,self.eval_end)
        self.assertEqual(original.mean,altered.mean)
        self.assertEqual(original.scale,altered.scale)
        i=original.eval_indices[:1]
        for name in ("nbeats","tft"):
            model=make_model(name,config,42).eval()
            with torch.inference_mode():
                torch.testing.assert_close(model(original.batch(i)),model(altered.batch(i)))

    def test_maintainer_shapes_gradients_and_history(self):
        config=dict(window=336,width=16,weather=True)
        data=DailyWindows(self.df,config,self.start,self.end,self.eval_end)
        for name in ("nbeats","tft"):
            model=make_model(name,config,42)
            self.assertEqual(type(model).__module__,"src.demand.torch_models")
            pred=model(data.batch(data.train_indices[:2]))
            self.assertEqual(tuple(pred.shape),(2,48,1))
            pred.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=="__main__":
    unittest.main()
