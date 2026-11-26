"""The optimizer code of the two GPT training scripts behind the GPT-2 results, transcribed for a single process.

Taken line by line from the scripts that produced the 3,085-step (all five changes) and 3,120-step (without change 2)
results; the only edits are: no torch.compile, no torch.distributed (one rank holds every matrix), and the module
constants and the switch step passed as arguments. The scripts descend from the tuned nanoGPT optimization-track
baseline (#36, https://github.com/KellerJordan/modded-nanogpt, MIT license). Used by test_matches_reference_*.
"""
import torch
from torch import Tensor

ISO_LAM = 0.5
ISO_CLAMP = 2.0
ISO_BETA = 0.95
BM_BF, BM_BS = 0.85, 0.985
OSC_WMAX, OSC_BETA = 0.6, 0.9
OSC_RHO_F = (1 - BM_BF) / (1 + BM_BF)


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def iso_metric(heat: Tensor) -> Tensor:
    m = (heat / heat.mean().clamp_min(1e-30)).clamp_min(1e-30) ** ISO_LAM
    return m.clamp(1 / ISO_CLAMP, ISO_CLAMP)


def isomuon_update(grad, momentum, row_heat, col_heat, mu=0.95):
    momentum.lerp_(grad, 1 - mu)
    u = grad.lerp_(momentum, mu)
    bi = iso_metric(row_heat).rsqrt()[:, None]
    ai = iso_metric(col_heat).rsqrt()[None, :]
    P = zeropower_via_newtonschulz5((u * bi) * ai)
    D = (P.float() * bi) * ai
    D = D * (min(u.size(-2), u.size(-1)) ** 0.5 / (D.norm() + 1e-12))
    D = D * max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return D.to(grad.dtype)


def isomuon_readout_update(grad, m_fast, m_slow, row_heat, col_heat, w, mu):
    # isomuon_update_bm with the units already advanced and mu a 0-d tensor
    m_eff = w * m_fast + (1 - w) * m_slow
    u = grad.lerp_(m_eff.to(grad.dtype), mu)
    bi = iso_metric(row_heat).rsqrt()[:, None]
    ai = iso_metric(col_heat).rsqrt()[None, :]
    P = zeropower_via_newtonschulz5((u * bi) * ai)
    D = (P.float() * bi) * ai
    D = D * (min(u.size(-2), u.size(-1)) ** 0.5 / (D.norm() + 1e-12))
    D = D * max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return D.to(grad.dtype)


def isomuon_update_bm(grad, m_fast, m_slow, row_heat, col_heat, w, mu=0.95):
    # isomuon_update with the single momentum EMA replaced by the two-mode memory read with the gated weight w
    m_fast.lerp_(grad, 1 - BM_BF)
    m_slow.lerp_(grad, 1 - BM_BS)
    m_eff = w * m_fast + (1 - w) * m_slow
    u = grad.lerp_(m_eff, mu)
    bi = iso_metric(row_heat).rsqrt()[:, None]
    ai = iso_metric(col_heat).rsqrt()[None, :]
    P = zeropower_via_newtonschulz5((u * bi) * ai)
    D = (P.float() * bi) * ai
    D = D * (min(u.size(-2), u.size(-1)) ** 0.5 / (D.norm() + 1e-12))
    D = D * max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return D.to(grad.dtype)


def bm_readout_update(grad, m_fast, m_slow, w, mu):
    # readout of already-advanced units (0-d tensors w, mu); Nesterov mix and orthogonalization as in the baseline
    m_eff = w * m_fast + (1 - w) * m_slow
    update = grad.lerp_(m_eff.to(grad.dtype), mu)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class RefAllFive(torch.optim.Optimizer):
    """Step of the all-five script (all five changes: two relaxation modes, fluctuation-calibrated output budget, gated readout, tracking from step 0, fresh-share cap)."""

    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, bm_start=750, fresh_cap=0.25):
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        self.bm_start, self.fresh_cap, self.cur_step = bm_start, fresh_cap, 0

    @torch.no_grad()
    def step(self):
        FRESH_CAP = self.fresh_cap
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                if hasattr(p, "iso_rs2"):
                    # noise variance of the micro-batch gradients = E_k[g_k^2] - (E_k g_k)^2, averaged over the other axis
                    K = p.iso_count
                    gm = p.grad.float() / K
                    rh = p.iso_rs2 / K / p.size(1) - gm.square().sum(1) / p.size(1)
                    ch = p.iso_cs2 / K / p.size(0) - gm.square().sum(0) / p.size(0)
                    rh = rh.clamp_min(0) + 1e-30; ch = ch.clamp_min(0) + 1e-30
                    if "row_heat" not in state:
                        state["row_heat"] = rh; state["col_heat"] = ch
                    else:
                        state["row_heat"].lerp_(rh, 1 - ISO_BETA); state["col_heat"].lerp_(ch, 1 - ISO_BETA)
                    if "m_fast" not in state:
                        state["m_fast"] = torch.zeros_like(state["momentum"]); state["m_slow"] = torch.zeros_like(state["momentum"])
                    # oscillation gate statistics, tracked from step 0: units measured before they take this gradient
                    g = p.grad.float(); g_norm = g.norm().clamp_min(1e-20)
                    m_fast = state["m_fast"].float()
                    osc = (-(g * m_fast).sum() / (g_norm * m_fast.norm().clamp_min(1e-20))).clamp_min(0.0)
                    drift = state["m_slow"].float().norm() / g_norm
                    if "osc" not in state:
                        state["osc"], state["drift"] = osc, drift
                    else:
                        state["osc"].lerp_(osc, 1 - OSC_BETA); state["drift"].lerp_(drift, 1 - OSC_BETA)
                    if self.cur_step >= self.bm_start:
                        w = OSC_WMAX * (1 - OSC_RHO_F * state["osc"] / state["drift"].clamp_min(1e-20)).clamp(0.0, 1.0)
                        if FRESH_CAP > 0:
                            state["m_fast"].lerp_(p.grad, 1 - BM_BF); state["m_slow"].lerp_(p.grad, 1 - BM_BS)
                            _mu = torch.tensor(group["mu"], dtype=torch.float32, device=p.device)
                            _men = (w * state["m_fast"].float() + (1 - w) * state["m_slow"].float()).norm()
                            _mu = torch.maximum(_mu, 1.0 / (1.0 + FRESH_CAP / (1 - FRESH_CAP) * _men / g_norm))
                            update = isomuon_readout_update(p.grad, state["m_fast"], state["m_slow"], state["row_heat"], state["col_heat"], w, _mu)
                        else:
                            update = isomuon_update_bm(p.grad, state["m_fast"], state["m_slow"], state["row_heat"], state["col_heat"], w, mu=group["mu"])
                    else:
                        # before the switch: change 2 on Muon's momentum; the two modes follow the raw gradient
                        state["m_fast"].lerp_(p.grad, 1 - BM_BF); state["m_slow"].lerp_(p.grad, 1 - BM_BS)
                        update = isomuon_update(p.grad, state["momentum"], state["row_heat"], state["col_heat"], mu=group["mu"])
                else:
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])


class RefNoIso(torch.optim.Optimizer):
    """Step of the script without change 2 (changes 1, 3, 4 and 5)."""

    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, bm_start=750, fresh_cap=0.25):
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        self.bm_start, self.fresh_cap, self._step = bm_start, fresh_cap, 0

    @torch.no_grad()
    def step(self):
        BM_START, FRESH_CAP = self.bm_start, self.fresh_cap
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["m_fast"] = torch.zeros_like(p)
                    state["m_slow"] = torch.zeros_like(p)
                # oscillation gate statistics, tracked from step 0: units measured before they take this gradient
                g = p.grad.float(); g_norm = g.norm().clamp_min(1e-20)
                m_fast = state["m_fast"].float()
                osc = (-(g * m_fast).sum() / (g_norm * m_fast.norm().clamp_min(1e-20))).clamp_min(0.0)
                drift = state["m_slow"].float().norm() / g_norm
                if "osc" not in state:
                    state["osc"], state["drift"] = osc, drift
                else:
                    state["osc"].lerp_(osc, 1 - OSC_BETA); state["drift"].lerp_(drift, 1 - OSC_BETA)
                # advance the units with this gradient (before the in-place Nesterov mix below)
                state["m_fast"].lerp_(p.grad, 1 - BM_BF)
                state["m_slow"].lerp_(p.grad, 1 - BM_BS)
                if self._step >= BM_START:
                    w = OSC_WMAX * (1 - OSC_RHO_F * state["osc"] / state["drift"].clamp_min(1e-20)).clamp(0.0, 1.0)
                    mu = torch.tensor(group["mu"], dtype=torch.float32, device=p.device)
                    if FRESH_CAP > 0:
                        m_eff_n = (w * state["m_fast"].float() + (1 - w) * state["m_slow"].float()).norm()
                        mu = torch.maximum(mu, 1.0 / (1.0 + FRESH_CAP / (1 - FRESH_CAP) * m_eff_n / g_norm))
                    update = bm_readout_update(p.grad, state["m_fast"], state["m_slow"], w, mu)
                else:
                    # before the switch: the baseline's Muon step; the two units follow the raw gradient
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        self._step += 1
