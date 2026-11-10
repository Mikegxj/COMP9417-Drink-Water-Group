"""Plain-PyTorch N-BEATSx and TFT, written to match neuralforecast 3.3.0.

The marking environment does not include neuralforecast, so these modules re-implement the two
architectures we use. They keep neuralforecast's layer order, initialisation order and parameter
names, so a state_dict trained with either implementation loads into the other, and the same seed
gives the same initial weights. Reference implementation: Nixtla neuralforecast 3.3.0
(neuralforecast/models/nbeatsx.py and neuralforecast/models/tft.py, Apache-2.0).

Only the parts our forecasts use are kept:
  * N-BEATSx: identity stacks, optional future exogenous inputs, no static/historic exogenous.
  * TFT: future exogenous inputs, LSTM encoder, no static/historic exogenous.

Both take the neuralforecast-style batch dict and return [batch, h, 1]:
  insample_y [B, L, 1], insample_mask [B, L, 1], futr_exog [B, L+h, C] or None.
"""
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed):
    """Same RNG state neuralforecast's BaseModel sets (pl.seed_everything) before building layers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- N-BEATSx

class IdentityBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size

    def forward(self, theta):
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, self.backcast_size:]
        return backcast, forecast.reshape(len(forecast), -1, 1)


class NBEATSBlock(nn.Module):
    def __init__(self, input_size, h, futr_input_size, mlp_units):
        super().__init__()
        self.futr_input_size = futr_input_size
        width = input_size + futr_input_size * (input_size + h)
        # neuralforecast's layer order: Linear(in, u0) then, for each unit pair, Linear + ReLU.
        layers = [nn.Linear(width, mlp_units[0][0])]
        for units in mlp_units:
            layers += [nn.Linear(units[0], units[1]), nn.ReLU()]
        layers.append(nn.Linear(mlp_units[-1][1], input_size + h))
        self.layers = nn.Sequential(*layers)
        self.basis = IdentityBasis(input_size, h)

    def forward(self, insample_y, futr_exog):
        if self.futr_input_size > 0:
            insample_y = torch.cat((insample_y, futr_exog.reshape(len(insample_y), -1)), dim=1)
        return self.basis(self.layers(insample_y))


class NBEATSx(nn.Module):
    """N-BEATSx with identity stacks (Olivares et al., 2023). futr_size=0 gives plain N-BEATS."""

    def __init__(self, h, input_size, futr_size=0, n_stacks=3, mlp_units=None, seed=1):
        super().__init__()
        seed_everything(seed)
        self.h = h
        mlp_units = mlp_units or 3 * [[512, 512]]
        self.blocks = nn.ModuleList([NBEATSBlock(input_size, h, futr_size, mlp_units)
                                     for _ in range(n_stacks)])

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"].squeeze(-1)
        insample_mask = windows_batch["insample_mask"].squeeze(-1)
        futr_exog = windows_batch["futr_exog"]
        residuals = insample_y.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))
        forecast = insample_y[:, -1:, None]  # level from the last observation
        for block in self.blocks:
            backcast, block_forecast = block(residuals, futr_exog)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
        return forecast


# ----------------------------------------------------------------------------- TFT

class MaybeLayerNorm(nn.Module):
    def __init__(self, output_size, hidden_size, eps):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = nn.LayerNorm(output_size if output_size else hidden_size, eps=eps)

    def forward(self, x):
        return self.ln(x)


class GLU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x):
        return F.glu(self.lin(x))


class GRN(nn.Module):
    """Gated residual network. Activation is ELU (neuralforecast's default)."""

    def __init__(self, input_size, hidden_size, output_size=None, context_hidden_size=None, dropout=0.):
        super().__init__()
        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=1e-3)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a, c=None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = self.glu(self.dropout(self.lin_i(F.elu(x))))
        y = a if not self.out_proj else self.out_proj(a)
        return self.layer_norm(x + y)


class TFTEmbedding(nn.Module):
    def __init__(self, hidden_size, futr_input_size, tgt_size=1):
        super().__init__()
        # Same creation order as neuralforecast (static and historic are unused here).
        for name, size in [("futr_exog_embedding", futr_input_size), ("tgt_embedding", tgt_size)]:
            vectors = nn.Parameter(torch.Tensor(size, hidden_size))
            bias = nn.Parameter(torch.zeros(size, hidden_size))
            nn.init.xavier_normal_(vectors)
            setattr(self, name + "_vectors", vectors)
            setattr(self, name + "_bias", bias)

    def forward(self, target_inp, futr_exog):
        k_inp = futr_exog.unsqueeze(-1) * self.futr_exog_embedding_vectors + self.futr_exog_embedding_bias
        target_inp = torch.matmul(target_inp.unsqueeze(3).unsqueeze(4),
                                  self.tgt_embedding_vectors.unsqueeze(1)).squeeze(3)
        return k_inp, target_inp + self.tgt_embedding_bias


class VariableSelectionNetwork(nn.Module):
    def __init__(self, hidden_size, num_inputs, dropout):
        super().__init__()
        self.joint_grn = GRN(hidden_size * num_inputs, hidden_size, output_size=num_inputs,
                             context_hidden_size=hidden_size)
        self.var_grns = nn.ModuleList([GRN(hidden_size, hidden_size, dropout=dropout)
                                       for _ in range(num_inputs)])

    def forward(self, x, context=None):
        weights = F.softmax(self.joint_grn(x.reshape(*x.shape[:-2], -1), c=context), dim=-1)
        embedded = torch.stack([m(x[..., i, :]) for i, m in enumerate(self.var_grns)], dim=-1)
        return torch.matmul(embedded, weights.unsqueeze(-1)).squeeze(-1), weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, hidden_size, example_length, attn_dropout, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_head = hidden_size // n_head
        self.qkv_linears = nn.Linear(hidden_size, (2 * n_head + 1) * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_head, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5
        self.register_buffer("_mask", torch.triu(
            torch.full((example_length, example_length), float("-inf")), 1).unsqueeze(0))

    def forward(self, x):
        bs, t, _ = x.shape
        q, k, v = self.qkv_linears(x).split(
            (self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), dim=-1)
        q = q.view(bs, t, self.n_head, self.d_head)
        k = k.view(bs, t, self.n_head, self.d_head)
        v = v.view(bs, t, self.d_head)
        score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        score.mul_(self.scale)
        prob = self.attn_dropout(F.softmax(score + self._mask, dim=3))
        out = torch.mean(torch.matmul(prob, v.unsqueeze(1)), dim=1)  # heads share V
        return self.out_dropout(self.out_proj(out)), prob


class TemporalCovariateEncoder(nn.Module):
    def __init__(self, hidden_size, num_historic_vars, num_future_vars, dropout):
        super().__init__()
        self.history_vsn = VariableSelectionNetwork(hidden_size, num_historic_vars, dropout)
        self.history_encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)
        self.future_encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)
        self.future_vsn = VariableSelectionNetwork(hidden_size, num_future_vars, dropout)
        self.input_gate = GLU(hidden_size, hidden_size)
        self.input_gate_ln = nn.LayerNorm(hidden_size, eps=1e-3)

    def forward(self, historical_inputs, future_inputs, cs, ch, cc):
        historical_features, _ = self.history_vsn(historical_inputs, cs)
        history, state = self.history_encoder(historical_features, (ch, cc))
        future_features, _ = self.future_vsn(future_inputs, cs)
        future, _ = self.future_encoder(future_features, state)
        embedding = torch.cat([historical_features, future_features], dim=1)
        features = self.input_gate(torch.cat([history, future], dim=1)) + embedding
        return self.input_gate_ln(features)


class TemporalFusionDecoder(nn.Module):
    def __init__(self, n_head, hidden_size, example_length, encoder_length, attn_dropout, dropout):
        super().__init__()
        self.encoder_length = encoder_length
        self.enrichment_grn = GRN(hidden_size, hidden_size, context_hidden_size=hidden_size, dropout=dropout)
        self.attention = InterpretableMultiHeadAttention(n_head, hidden_size, example_length,
                                                         attn_dropout, dropout)
        self.attention_gate = GLU(hidden_size, hidden_size)
        self.attention_ln = nn.LayerNorm(hidden_size, eps=1e-3)
        self.positionwise_grn = GRN(hidden_size, hidden_size, dropout=dropout)
        self.decoder_gate = GLU(hidden_size, hidden_size)
        self.decoder_ln = nn.LayerNorm(hidden_size, eps=1e-3)

    def forward(self, temporal_features, ce):
        enriched = self.enrichment_grn(temporal_features, c=ce)
        x, _ = self.attention(enriched)
        x = x[:, self.encoder_length:, :]
        temporal_features = temporal_features[:, self.encoder_length:, :]
        enriched = enriched[:, self.encoder_length:, :]
        x = self.attention_ln(self.attention_gate(x) + enriched)
        x = self.decoder_gate(self.positionwise_grn(x))
        return self.decoder_ln(x + temporal_features)


class TFT(nn.Module):
    """Temporal Fusion Transformer (Lim et al., 2021) with known future covariates only."""

    def __init__(self, h, input_size, futr_size, hidden_size=128, n_head=4,
                 attn_dropout=0., dropout=0.1, seed=1):
        super().__init__()
        if futr_size < 1:
            raise ValueError("This TFT needs at least one future covariate")
        seed_everything(seed)
        self.h, self.input_size = h, input_size
        self.example_length = input_size + h
        self.embedding = TFTEmbedding(hidden_size, futr_size)
        self.temporal_encoder = TemporalCovariateEncoder(hidden_size, futr_size + 1, futr_size, dropout)
        self.temporal_fusion_decoder = TemporalFusionDecoder(n_head, hidden_size, self.example_length,
                                                             input_size, attn_dropout, dropout)
        self.output_adapter = nn.Linear(hidden_size, 1)

    def forward(self, windows_batch):
        y_insample = windows_batch["insample_y"]
        futr_exog = windows_batch["futr_exog"]
        k_inp, target = self.embedding(y_insample, futr_exog)
        batch, _, _, hidden = target.shape
        # No static covariates: neuralforecast feeds zero contexts and zero LSTM states.
        cs = torch.zeros(batch, hidden, device=y_insample.device)
        ce = torch.zeros(batch, hidden, device=y_insample.device)
        ch = torch.zeros(1, batch, hidden, device=y_insample.device)
        cc = torch.zeros(1, batch, hidden, device=y_insample.device)
        historical = torch.cat([k_inp[:, :self.input_size, :], target[:, :self.input_size, :]], dim=-2)
        future = k_inp[:, self.input_size:]
        features = self.temporal_encoder(historical, future, cs, ch, cc)
        return self.output_adapter(self.temporal_fusion_decoder(features, ce))
