"""Model code shared by the demand experiments: per-half-hour Ridge and the daily MLP.

Both models predict the ratio target R = demand / L (see features.level); multiply by L to
get MW. MAPE in ratio space equals MAPE in MW because both are divided by the same L.
"""
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from torch import nn

import features as F


def ratio_mape(pred, R):
    return float(np.mean(np.abs(pred - R) / R) * 100)


# --- Ridge -------------------------------------------------------------------------------
def ridge_pipeline(alpha, cols):
    spl = [c for c in cols if c in F.SPLINE_COLS]
    other = [c for c in cols if c not in F.SPLINE_COLS]
    ct = ColumnTransformer([
        ("spl", SplineTransformer(n_knots=6, degree=3, extrapolation="linear"), spl),
        ("rest", "passthrough", other)])
    return make_pipeline(ct, StandardScaler(), Ridge(alpha=alpha))


def ridge_fit_predict(frames, R, train, alpha):
    """Fit one pipeline per half-hour on rows `train`; predict every row."""
    out = np.zeros((len(frames[0]), F.P))
    for p, fr in enumerate(frames):
        m = ridge_pipeline(alpha, list(fr.columns)).fit(fr[train], R[train, p])
        out[:, p] = m.predict(fr)
    return out


def ridge_select_and_fit(frames, R, inner_tr, inner_va, train,
                         alphas=(0.01, 0.03, 0.1, 0.3, 1, 3, 10)):
    """Choose alpha on the inner validation rows, then refit on all `train` rows."""
    scores = {a: ratio_mape(ridge_fit_predict(frames, R, inner_tr, a)[inner_va], R[inner_va])
              for a in alphas}
    alpha = min(scores, key=scores.get)
    return ridge_fit_predict(frames, R, train, alpha), alpha, scores


# --- MLP ---------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, h=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, F.P))

    def forward(self, x):
        return self.net(x) + 1.0             # targets are ratios around 1


def fit_mlp(X, R, seed, epochs, Xv=None, Rv=None, patience=40):
    """Train with a MAPE loss. With validation data, early-stop and return the best epoch."""
    torch.manual_seed(seed)
    sc = StandardScaler().fit(X)
    xt = torch.tensor(sc.transform(X), dtype=torch.float32)
    rt = torch.tensor(R, dtype=torch.float32)
    if Xv is not None:
        xv = torch.tensor(sc.transform(Xv), dtype=torch.float32)
        rv = torch.tensor(Rv, dtype=torch.float32)
    model = MLP(xt.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    best_v, best_ep = np.inf, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        for b in range(0, len(xt), 64):
            ib = perm[b:b + 64]
            opt.zero_grad()
            loss = (torch.abs(model(xt[ib]) - rt[ib]) / rt[ib]).mean()   # = MAPE / 100
            loss.backward()
            opt.step()
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                v = (torch.abs(model(xv) - rv) / rv).mean().item()
            if v < best_v:
                best_v, best_ep = v, ep + 1
            elif ep + 1 - best_ep >= patience:
                break
    return model, sc, best_ep


def predict_mlp(model, sc, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(sc.transform(X), dtype=torch.float32)).numpy()


def mlp_fit_predict(X, R, train, seed, epochs, threads=2):
    """Fit with a fixed number of epochs (no validation) and predict every row."""
    torch.set_num_threads(threads)
    model, sc, _ = fit_mlp(X[train], R[train], seed, epochs)
    return predict_mlp(model, sc, X)


# --- LightGBM (reproduction of the teammate's exp2 configuration) -------------------------
LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=24,
                   random_state=42, n_jobs=2, verbosity=-1)


def lgbm_fit_predict(frame, y, train):
    """frame/y in long format (day-major); train is a boolean mask over rows."""
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(**LGBM_PARAMS).fit(frame[train], y[train])
    return np.asarray(model.predict(frame)).ravel()


def mlp_seed_run(X, R, inner_tr, inner_va, train, seed, max_epochs=400, threads=4):
    """One seed: pick the epoch count on the inner split, refit on `train`, predict all rows."""
    torch.set_num_threads(threads)
    _, _, ep = fit_mlp(X[inner_tr], R[inner_tr], seed, max_epochs, X[inner_va], R[inner_va])
    model, sc, _ = fit_mlp(X[train], R[train], seed, max(ep, 1))
    return predict_mlp(model, sc, X), ep
