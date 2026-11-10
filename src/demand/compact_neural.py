"""Compact point-forecast networks. One sample per market day, 336 inputs to 48 outputs."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


class NBeats(nn.Module):
    """Generic basis, doubly residual backcast/forecast blocks."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(336, 128), nn.ReLU(), nn.Linear(128, 128),
                          nn.ReLU(), nn.Linear(128, 336 + 48)) for _ in range(3)
        ])

    def forward(self, history, future):
        prediction = history.new_zeros(len(history), 48)
        for block in self.blocks:
            backcast, adjustment = block(history).split([336, 48], dim=-1)
            history = history - backcast
            prediction = prediction + adjustment
        return prediction


class NBeatsX(nn.Module):
    """Generic N-BEATS blocks plus a forecast basis of known future covariates."""
    def __init__(self, covariates):
        super().__init__()
        self.generic = NBeats()
        self.coefficients = nn.Sequential(nn.Linear(336 + 48 * covariates, 96), nn.ReLU(),
                                          nn.Linear(96, covariates))

    def forward(self, history, future):
        weights = self.coefficients(torch.cat([history, future.flatten(1)], dim=1))
        return self.generic(history, future) + (weights[:, None] * future).sum(-1)


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(16, 16, 3, dilation=dilation)

    def forward(self, x):
        return x + F.relu(self.conv(F.pad(x, (self.padding, 0))))


class SequenceRegressor(nn.Module):
    def __init__(self, name, covariates):
        super().__init__()
        self.name = name
        if name == "lstm":
            self.encoder = nn.LSTM(1, 16, batch_first=True)
        else:
            self.encoder = nn.Sequential(nn.Conv1d(1, 16, 1),
                                         *[CausalBlock(2**i) for i in range(8)])
        self.head = nn.Sequential(nn.Linear(16 + covariates + 2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, history, future):
        if self.name == "lstm":
            _, (hidden, _) = self.encoder(history.unsqueeze(-1))
            context = hidden[-1]
        else:
            context = self.encoder(history.unsqueeze(1))[:, :, -1]
        # Same-slot daily/weekly lags are both inside the 336-point history.
        inputs = torch.cat([context[:, None].expand(-1, 48, -1), future,
                            history[:, -48:, None], history[:, :48, None]], dim=-1)
        return self.head(inputs).squeeze(-1)


class GateNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(width, 2 * width), nn.GLU())
        self.norm = nn.LayerNorm(width)

    def forward(self, x, skip):
        return self.norm(skip + self.gate(x))


class CompactTFT(nn.Module):
    """Single-series point TFT variant with variable selection and one attention head."""
    def __init__(self, covariates, width=16):
        super().__init__()
        self.history_embed = nn.Linear(1, width)
        self.embeddings = nn.ModuleList([nn.Linear(1, width) for _ in range(covariates)])
        self.weights = nn.Sequential(nn.Linear(covariates, width), nn.ELU(), nn.Linear(width, covariates))
        self.encoder = nn.LSTM(width, width, batch_first=True)
        self.decoder = nn.LSTM(width, width, batch_first=True)
        self.local_gate = GateNorm(width)
        self.attention = nn.MultiheadAttention(width, 1, batch_first=True)
        self.attention_gate = GateNorm(width)
        self.feed = nn.Sequential(nn.Linear(width, width), nn.ELU(), nn.Linear(width, width))
        self.output_gate = GateNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, history, future):
        past = self.history_embed(history.unsqueeze(-1))
        weights = self.weights(future).softmax(-1)
        variables = torch.stack([embed(future[:, :, i:i+1]) for i, embed in enumerate(self.embeddings)], dim=2)
        known = (variables * weights.unsqueeze(-1)).sum(2)
        encoded, state = self.encoder(past)
        decoded, _ = self.decoder(known, state)
        memory = self.local_gate(torch.cat([encoded, decoded], 1), torch.cat([past, known], 1))
        query = memory[:, -48:]
        mask = torch.ones(48, memory.size(1), device=memory.device, dtype=torch.bool).triu(history.size(1)+1)
        attended, _ = self.attention(query, memory, memory, attn_mask=mask, need_weights=False)
        attended = self.attention_gate(attended, query)
        return self.head(self.output_gate(self.feed(attended), attended)).squeeze(-1)


def day_tensors(df, days, cutoff, train_start=None):
    train_rows = days.year < cutoff
    if train_start is not None:
        train_rows &= days >= train_start
    demand = df.nsw_demand.to_numpy(dtype=np.float32)
    mean, scale = demand[train_rows].mean(), demand[train_rows].std()
    loads = torch.from_numpy((demand - mean) / scale)
    # Window starts every 48 rows. Last window has no following target day.
    history = loads.unfold(0, 336, 48)[:-1]
    targets = loads[336:].reshape(-1, 48)
    temp = df.syd_temp.to_numpy()
    temp = (temp - temp[train_rows].mean()) / temp[train_rows].std()
    start = df.index - pd.Timedelta(minutes=30)
    features = [temp]
    for positions, period in [(start.hour * 2 + start.minute // 30, 48),
                              (start.dayofweek, 7), (start.dayofyear - 1, 365.25)]:
        angle = 2 * np.pi * positions / period
        features.extend([np.sin(angle), np.cos(angle)])
    known = torch.tensor(np.column_stack(features)[336:], dtype=torch.float32).reshape(-1, 48, 7)
    count = int((days.year < cutoff).sum() // 48) - 7
    return history, known, targets, count, mean, scale


def build_model(name, covariates, variant="base"):
    if name == "nbeats" and variant in ("calendar_only", "calendar_temp"):
        model = NBeatsX(covariates)
    elif name == "nbeats":
        model = NBeats()
    elif name == "tft":
        model = CompactTFT(covariates)
    else:
        model = SequenceRegressor(name, covariates)
    return model


def forecast(name, df, days, cutoff, epochs, device, variant="base", train_start=None):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    history, future, targets, count, mean, scale = day_tensors(df, days, cutoff, train_start)
    first = 0 if train_start is None else max(0, int((days < train_start).sum() // 48) - 7)
    if variant == "calendar_only":
        future = future[:, :, 1:]
    model = build_model(name, future.size(-1), variant)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for _ in range(epochs):
        for batch in (first + torch.randperm(count - first)).split(32):
            optimizer.zero_grad()
            pred = model(history[batch].to(device), future[batch].to(device))
            loss = F.l1_loss(pred, targets[batch].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        values = []
        for batch in torch.arange(count, len(targets)).split(32):
            values.append(model(history[batch].to(device), future[batch].to(device)).cpu())
    return torch.cat(values).numpy().ravel() * scale + mean
