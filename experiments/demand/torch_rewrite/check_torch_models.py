"""Check src/demand/torch_models.py against neuralforecast 3.3.0.

Needs neuralforecast==3.3.0 (development only, not in the marking environment).
Checks identical initial weights, forward outputs, five training steps with dropout, and that a
neuralforecast-trained state_dict loads into our module. Run: python -m experiments.demand.torch_rewrite.check_torch_models
"""
import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
import torch
from torch.nn import functional as F
from neuralforecast.models import NBEATSx as NF_NBEATSx, NBEATS as NF_NBEATS, TFT as NF_TFT
from neuralforecast.losses.pytorch import MAE
from src.demand import torch_models as tm

H, C = 48, 10


def nf_model(kind, L, width, seed, futr):
    common = dict(h=H, input_size=L, loss=MAE(), scaler_type="identity", random_seed=seed)
    torch.manual_seed(seed)
    if kind == "nbeats":
        blocks = dict(stack_types=["identity"] * 3, n_blocks=[1] * 3, mlp_units=[[width, width]] * 3)
        if futr:
            return NF_NBEATSx(**common, **blocks, futr_exog_list=[f"c{i}" for i in range(C)])
        return NF_NBEATS(**common, **blocks)
    return NF_TFT(**common, futr_exog_list=[f"c{i}" for i in range(C)], hidden_size=width, n_head=4, dropout=0.1)


def my_model(kind, L, width, seed, futr):
    torch.manual_seed(seed)
    if kind == "nbeats":
        return tm.NBEATSx(H, L, C if futr else 0, mlp_units=[[width, width]] * 3, seed=seed)
    return tm.TFT(H, L, C, hidden_size=width, n_head=4, dropout=0.1, seed=seed)


def batch(n, L, gen, futr=True):
    y = torch.randn(n, L, 1, generator=gen)
    return dict(insample_y=y, insample_mask=torch.ones_like(y),
                futr_exog=torch.randn(n, L + H, C, generator=gen) if futr else None,
                hist_exog=None, stat_exog=None)


def check(kind, L, width, seed, futr=True):
    a, b = nf_model(kind, L, width, seed, futr), my_model(kind, L, width, seed, futr)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb), set(sa) ^ set(sb)
    init_same = all(torch.equal(sa[k], sb[k]) for k in sa)
    gen = torch.Generator().manual_seed(0)
    data = [batch(32, L, gen, futr) for _ in range(5)]
    targets = [torch.randn(32, H, generator=gen) for _ in range(5)]
    a.eval(); b.eval()
    with torch.no_grad():
        fwd = max((a(x) - b(x)).abs().max().item() for x in data)
    # Five training steps in our loop (dropout on, same RNG for both).
    diffs = []
    oa, ob = torch.optim.Adam(a.parameters(), 1e-3), torch.optim.Adam(b.parameters(), 1e-3)
    for model, opt in ((a, oa), (b, ob)):
        torch.manual_seed(123)
        model.train()
        for x, t in zip(data, targets):
            opt.zero_grad()
            F.l1_loss(model(x).squeeze(-1), t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
    train = max((a.state_dict()[k] - b.state_dict()[k]).abs().max().item() for k in sa)
    # Cross-load: a state_dict trained by neuralforecast loads into ours and predicts the same.
    c = my_model(kind, L, width, 999, futr)
    c.load_state_dict(a.state_dict()); a.eval(); c.eval()
    with torch.no_grad():
        cross = max((a(x) - c(x)).abs().max().item() for x in data)
    print(f"{kind:6s} L={L} w={width} seed={seed} futr={futr}: params={len(sa)} init_identical={init_same} "
          f"forward_maxdiff={fwd:.2e} after_5_steps_maxdiff={train:.2e} crossload_maxdiff={cross:.2e}")
    return init_same and fwd == 0 and train == 0 and cross == 0


ok = all([check("nbeats", 480, 128, 42), check("nbeats", 336, 64, 43), check("nbeats", 96, 32, 44, futr=False),
          check("tft", 336, 32, 42), check("tft", 96, 16, 43)])
print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
