"""Neural demand models, trained with project-controlled daily windows and scaling.

N-BEATSx and TFT come from src/demand/torch_models.py (plain PyTorch, matches neuralforecast 3.3.0
bit for bit; see experiments/demand/check_torch_models.py). neuralforecast is not allowed in the marking
environment. TCN and LSTM were only screened and still import neuralforecast lazily.
"""
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "nsw_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "nsw_cache"))

import numpy as np
import pandas as pd
import torch

from harness import dayahead
from src.demand.torch_models import NBEATSx, TFT

COVARIATES = ["temp", "slot_sin", "slot_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos"]
# Calendar-only flags (exp2): known for the forecast day as well as the history window.
HOLIDAY_COVARIATES = ["hol_today", "hol_lag1", "hol_lag7"]
# Optional extra Sydney weather (known for the forecast day under the course interface).
# Off by default so existing configs and checkpoints keep the same input width.
EXTRA_WEATHER = {"cloud": "syd_cloud", "solar": "syd_solar", "rh": "syd_rh"}
# Optional extras matching the MLP's inputs (section 18 of the experiment log), all off by default:
#   xmas: flag for market days 22 Dec - 4 Jan;
#   tempstats: Sydney Tmax, Tmin of the market day and the mean temperature of days d-2..d
#     (weather of the forecast day is known under the course interface);
#   level: demand scaled per sample by the mean of the 7 days before the forecast day
#     (input and target = demand / level - 1) instead of one train-period z-score;
#   loss="mape": train on absolute percentage error instead of MAE.
TEMPSTATS = ["tmax", "tmin", "t3"]
LEVEL_POINTS = 7 * 48


def extra_weather(config):
    return [key for key in EXTRA_WEATHER if config.get(key, False)]


def covariates(config):
    return (COVARIATES + (HOLIDAY_COVARIATES if config.get("holidays", False) else [])
            + extra_weather(config) + (["xmas"] if config.get("xmas", False) else [])
            + (TEMPSTATS if config.get("tempstats", False) else []))


def make_model(name, config, seed):
    torch.manual_seed(seed)
    size, width = config["window"], config["width"]
    if name == "nbeats":
        futr = len(covariates(config)) if config["weather"] else 0
        return NBEATSx(h=48, input_size=size, futr_size=futr, mlp_units=[[width, width]] * 3, seed=seed)
    if name == "tft":
        return TFT(h=48, input_size=size, futr_size=len(covariates(config)), hidden_size=width,
                   n_head=4, dropout=0.1, seed=seed)
    # Screened only (not selected): requires neuralforecast, not available in marking.
    from neuralforecast.models import TCN, LSTM
    from neuralforecast.losses.pytorch import MAE
    common = dict(h=48, input_size=size, loss=MAE(), scaler_type="identity", random_seed=seed,
                  futr_exog_list=covariates(config))
    if name == "lstm":
        return LSTM(**common, encoder_n_layers=1, encoder_hidden_size=width,
                    decoder_hidden_size=width, decoder_layers=2)
    # One kernel-3 causal convolution per dilation. Receptive field >= input window.
    layers = int(np.ceil(np.log2((size - 1) / 2 + 1)))
    return TCN(**common, encoder_hidden_size=width, decoder_hidden_size=width,
               kernel_size=3, dilations=[2**i for i in range(layers)])


class DailyWindows:
    def __init__(self, df, config, train_start, train_end, eval_end):
        days = dayahead.market_day(df.index)
        size = config["window"]
        keep = (days >= train_start-pd.Timedelta(days=size//48)) & (days <= eval_end)
        df, days = df.loc[keep], days[keep]
        expected = pd.date_range(df.index[0], df.index[-1], freq="30min")
        if not df.index.equals(expected) or not (days.value_counts()==48).all():
            raise ValueError("Expected complete contiguous market days")
        train = (days >= train_start) & (days <= train_end)
        y = df.nsw_demand.to_numpy(dtype=np.float32)
        self.mean, self.scale = float(y[train].mean()), float(y[train].std())
        self.level = None
        if config.get("level", False):
            if size < LEVEL_POINTS:
                raise ValueError("Level scaling needs a window of at least 7 days")
            raw = torch.tensor(y)
            history = raw.unfold(0,size,48)[:-1]
            self.level = history[:, -LEVEL_POINTS:].mean(1, keepdim=True)
            self.history = (history/self.level-1).unsqueeze(-1)
            self.target = raw[size:].reshape(-1,48)/self.level-1
        else:
            load = torch.tensor((y-self.mean)/self.scale)
            self.history = load.unfold(0,size,48)[:-1].unsqueeze(-1)
            self.target = load[size:].reshape(-1,48)
        start = df.index-pd.Timedelta(minutes=30)
        temp = df.syd_temp.to_numpy(dtype=np.float32)
        self.temp_mean, self.temp_scale = float(temp[train].mean()), float(temp[train].std())
        cov = [(temp-self.temp_mean)/self.temp_scale]
        for positions, period in [(start.hour*2+start.minute//30,48),(start.dayofweek,7),
                                  (start.dayofyear-1,365.25)]:
            cov.extend([np.sin(2*np.pi*positions/period), np.cos(2*np.pi*positions/period)])
        if config.get("holidays", False):
            from src.demand.tabular import add_holidays
            flags = add_holidays(pd.DataFrame(index=df.index), 2)
            cov.extend(flags[HOLIDAY_COVARIATES].to_numpy(dtype=np.float32).T)
        # Standardised with training-window statistics only.
        self.extra_scalers = {}
        for key in extra_weather(config):
            values = df[EXTRA_WEATHER[key]].to_numpy(dtype=np.float32)
            mean, scale = float(values[train].mean()), float(values[train].std())
            self.extra_scalers[key] = (mean, scale)
            cov.append((values-mean)/scale)
        if config.get("xmas", False):
            cov.append(np.asarray(((days.month == 12) & (days.day >= 22)) | ((days.month == 1) & (days.day <= 4)),
                                  dtype=np.float32))
        if config.get("tempstats", False):
            daily = pd.Series(temp, index=days).groupby(level=0).agg(["max", "min", "mean"])
            daily["t3"] = daily["mean"].rolling(3).mean()
            stats = daily.rename(columns={"max": "tmax", "min": "tmin"})[TEMPSTATS].reindex(days).to_numpy(np.float32)
            stats[np.isnan(stats)] = np.nanmean(stats[:, 2])  # first two days of the frame only
            for key, values in zip(TEMPSTATS, stats.T):
                mean, scale = float(values[train].mean()), float(values[train].std())
                self.extra_scalers[key] = (mean, scale)
                cov.append((values-mean)/scale)
        cov = torch.tensor(np.column_stack(cov),dtype=torch.float32)
        self.known = cov.unfold(0,size+48,48).permute(0,2,1)
        self.index = df.index[size:]
        sample_days = days[size:][::48]
        self.train_indices = torch.from_numpy(np.flatnonzero((sample_days >= train_start)&(sample_days <= train_end)))
        self.eval_indices = torch.from_numpy(np.flatnonzero(sample_days > train_end))
        self.actual = y[size:].reshape(-1,48)
        self.use_weather = config.get("weather",True)
        # Per-element loss weights: None = plain MAE on the scaled target (the original behaviour).
        # For MAPE, |error in MW| / actual = |error in model units| * d(MW)/d(unit) / actual, normalised
        # to mean 1 over the training samples so the gradient size stays comparable with MAE.
        self.loss_weight = None
        if config.get("loss", "mae") == "mape":
            unit = self.level.expand(-1,48) if self.level is not None else torch.full((len(self.target),48), self.scale)
            weight = unit/torch.tensor(self.actual)
            self.loss_weight = weight/weight[self.train_indices].mean()
        elif config.get("loss", "mae") != "mae":
            raise ValueError(f"Unknown loss {config['loss']}")

    def inverse(self, output, indices):
        if self.level is not None:
            return (output+1)*self.level[indices].numpy()
        return output*self.scale+self.mean

    def batch(self, indices):
        history = self.history[indices]
        return dict(insample_y=history, insample_mask=torch.ones_like(history),
                    futr_exog=self.known[indices] if self.use_weather else None,
                    hist_exog=None, stat_exog=None)


def device_batch(data, indices, device):
    return {key:value.to(device) if value is not None else None for key,value in data.batch(indices).items()}


def train_loss(data, output, batch, device):
    """Training loss for a batch of samples; plain MAE unless the config asks for MAPE."""
    from torch.nn import functional as F
    target = data.target[batch].to(device)
    if data.loss_weight is None:
        return F.l1_loss(output, target)
    return ((output-target).abs()*data.loss_weight[batch].to(device)).mean()


def predict(model, data, indices):
    model.eval()
    with torch.inference_mode():
        preds = [model(device_batch(data,batch,next(model.parameters()).device)).squeeze(-1).cpu() for batch in indices.split(32)]
    return data.inverse(torch.cat(preds).numpy(), indices)
