"""Muon with five changes to its update, each switchable on its own, plus AdamW for non-matrix parameters.

Muon (the base): for every hidden weight matrix,

    m      <- lerp(m, g, 1 - mu)                       # single EMA momentum
    update =  scale * polar(lerp(g, m, mu))            # Nesterov mix, Newton-Schulz polar factor

The five changes (all off by default; ``physmuon.language_model()`` turns all five on):

1. Two relaxation modes (``bm_start``). From ``bm_start`` on, the single EMA is replaced by a convex
   combination of a fast and a slow mode (two EMAs):
       m_fast <- lerp(m_fast, g, 1 - beta_fast);   m_slow <- lerp(m_slow, g, 1 - beta_slow)
       m_eff  =  w * m_fast + (1 - w) * m_slow,    update = scale * polar(lerp(g, m_eff, mu))
2. Fluctuation-calibrated output budget (``iso=True``). The polar factor is taken in a diagonal metric built from the gradient-noise variance of
   the current step's micro-batches: row and column fluctuation estimates H_r, H_c (EMAs over steps),
       B = clamp((H_r / mean H_r)^lam, 1/c, c),   A = clamp((H_c / mean H_c)^lam, 1/c, c),
       D = B^-1/2 polar(B^-1/2 U A^-1/2) A^-1/2,  rescaled to the Frobenius norm sqrt(min(rows, cols)).
   Call ``observe_microbatch()`` after the backward pass of every micro-batch (see the README).
3. Gated readout (``osc_gate=True``, needs change 1). The weight of the fast mode is measured per matrix and
   per step instead of being fixed:
       w = osc_wmax * max(0, 1 - rho_f * o / d),   rho_f = (1 - beta_fast) / (1 + beta_fast),
   o = EMA_osc_beta[max(0, -cos(g, m_fast))]  (how strongly the gradient pushes back on the recent memory),
   d = EMA_osc_beta[|m_slow| / |g|]            (the persistent drift), both measured before the modes take g.
4. Tracking from step 0 (``bm_shadow=True``, needs change 1). The two modes and the gate statistics follow the gradient from
   step 0; at ``bm_start`` only the readout switches, from the single EMA to the two modes. Without it, both
   modes are created at ``bm_start`` as copies of the live momentum.
5. Fresh-share cap (``fresh_cap=phi > 0``, needs change 1). After the switch the Nesterov coefficient is
   raised when needed so that the current gradient makes up at most the share phi of the update input:
       mu_t = max(mu, 1 / (1 + phi / (1 - phi) * |m_eff| / |g|)).

Optional and not one of the five: a spike guard (``spike_guard > 0``) that keeps steps whose gradient norm exceeds
``spike_guard`` times the running median out of the slow mode, a stability safeguard for heavy-tailed gradients that
follows earlier work (SPAM, Huang et al., ICLR 2025, arXiv:2501.06842; ZClip, Kumar et al. 2025, arXiv:2504.02507).

Change 2 acts on 2-D parameters only. Parameters with more than two dimensions are treated as stacks of matrices over
their last two dims; for them change 2 is skipped and the gate and the fresh share are measured on the whole tensor.
"""
from __future__ import annotations

import bisect
import math
import warnings
from typing import Iterable

import torch
import torch.distributed as dist

# Newton-Schulz variants: Keller Jordan's quintic iteration (5 steps) and the cubic iteration (12 steps) of the
# tuned nanoGPT optimization-track baseline (#36), on which the GPT results of this package were measured.
_NS = {"quintic": ((3.4445, -4.7750, 2.0315), 5), "cubic": ((2.0, -1.5, 0.5), 12)}


def newton_schulz(G: torch.Tensor, mode: str = "quintic", steps: int | None = None, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration towards the polar factor of G (last two dims), in bfloat16."""
    (a, b, c), default_steps = _NS[mode]
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(default_steps if steps is None else steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Keller Jordan's quintic Newton-Schulz iteration (kept for backward compatibility)."""
    return newton_schulz(G, "quintic", steps, eps)


def _iso_metric(heat: torch.Tensor, lam: float, clamp: float) -> torch.Tensor:
    m = (heat / heat.mean().clamp_min(1e-30)).clamp_min(1e-30) ** lam
    return m.clamp(1 / clamp, clamp)


class PhysMuon(torch.optim.Optimizer):
    """Muon with the five switchable changes above for groups with ``use_muon=True``; AdamW for the other groups.

    Group options (defaults in brackets):
      use_muon [all params of the group have ndim >= 2], lr, weight_decay [0] (decoupled, multiplied by lr),
      Muon groups:
        momentum mu [0.95], nesterov [True], ns ["quintic"] or "cubic", ns_steps [5 for quintic, 12 for cubic],
        scale: "shape" -> sqrt(max(1, rows/cols)) (as in Muon), or a float c -> c * sqrt(max(rows, cols)),
        1. bm_start [None = single EMA], bm_beta_fast [0.85], bm_beta_slow [0.98], bm_w_fast [0.4385]
           (fixed weight of the fast mode, used when osc_gate is off),
        2. iso [False], iso_lam [0.5], iso_clamp [2.0], iso_beta [0.95], iso_sync [True] (average the fluctuation estimates over
           ranks when torch.distributed is initialized),
        3. osc_gate [False], osc_wmax [0.6], osc_beta [0.9],
        4. bm_shadow [False],
        5. fresh_cap [0.0 = off];
      AdamW groups: betas [(0.9, 0.95)], eps [1e-8].
    Optimizer-level options: spike_guard [0 = off] (threshold in multiples of the running median of the global
    gradient norm), spike_window [1000] (steps in the running window).
    All optimizer state (the two modes, gate statistics, fluctuation estimates) lives in ``state`` and survives ``state_dict`` /
    ``load_state_dict``.
    """

    def __init__(self, params: Iterable, lr: float = 0.02, weight_decay: float = 0.0, momentum: float = 0.95,
                 nesterov: bool = True, ns: str = "quintic", ns_steps: int | None = None, scale="shape",
                 bm_start: int | None = None, bm_beta_fast: float = 0.85, bm_beta_slow: float = 0.98,
                 bm_w_fast: float = 0.4385, iso: bool = False, iso_lam: float = 0.5, iso_clamp: float = 2.0,
                 iso_beta: float = 0.95, iso_sync: bool = True, osc_gate: bool = False, osc_wmax: float = 0.6,
                 osc_beta: float = 0.9, bm_shadow: bool = False, fresh_cap: float = 0.0,
                 betas=(0.9, 0.95), eps: float = 1e-8, spike_guard: float = 0.0, spike_window: int = 1000):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns=ns,
                        ns_steps=ns_steps, scale=scale, bm_start=bm_start, bm_beta_fast=bm_beta_fast,
                        bm_beta_slow=bm_beta_slow, bm_w_fast=bm_w_fast, iso=iso, iso_lam=iso_lam,
                        iso_clamp=iso_clamp, iso_beta=iso_beta, iso_sync=iso_sync, osc_gate=osc_gate,
                        osc_wmax=osc_wmax, osc_beta=osc_beta, bm_shadow=bm_shadow, fresh_cap=fresh_cap,
                        betas=betas, eps=eps, use_muon=None)
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["use_muon"] is None:
                group["use_muon"] = all(p.ndim >= 2 for p in group["params"])
            if group["use_muon"]:
                self._check_muon_group(group)
        if not spike_guard >= 0.0 or spike_window < 10:
            raise ValueError("spike_guard must be >= 0 and spike_window >= 10")
        self.spike_guard = float(spike_guard)
        self.spike_window = int(spike_window)
        self._spike_sorted: list[float] = []     # running window of global gradient norms, sorted ...
        self._spike_fifo: list[float] = []       # ... and in arrival order
        self.spike_frozen_steps = 0              # how often the slow mode skipped a step (diagnostic)
        self._warned_iso = False

    @staticmethod
    def _check_muon_group(group):
        if group["ns"] not in _NS:
            raise ValueError(f"ns must be one of {sorted(_NS)}")
        if not 0.0 <= group["momentum"] < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        needs_memory = [k for k in ("osc_gate", "bm_shadow") if group[k]] + (["fresh_cap"] if group["fresh_cap"] else [])
        if group["bm_start"] is None:
            if needs_memory:
                raise ValueError(f"{needs_memory} act on the two relaxation modes: set bm_start")
        else:
            for k in ("bm_beta_fast", "bm_beta_slow"):
                if not 0.0 <= group[k] < 1.0:
                    raise ValueError(f"{k} must lie in [0, 1)")
            if not 0.0 <= group["bm_w_fast"] <= 1.0:
                raise ValueError("bm_w_fast must lie in [0, 1]")
            if not 0.0 < group["osc_wmax"] <= 1.0 or not 0.0 <= group["osc_beta"] < 1.0:
                raise ValueError("osc_wmax must lie in (0, 1] and osc_beta in [0, 1)")
            if not 0.0 <= group["fresh_cap"] < 1.0:
                raise ValueError("fresh_cap must lie in [0, 1)")
        if group["iso"] and (group["iso_lam"] < 0.0 or group["iso_clamp"] < 1.0 or not 0.0 <= group["iso_beta"] < 1.0):
            raise ValueError("iso_lam must be >= 0, iso_clamp >= 1 and iso_beta in [0, 1)")

    @staticmethod
    def _scale(p: torch.Tensor, scale) -> float:
        rows, cols = p.shape[-2], p.shape[-1]
        if scale == "shape":
            return max(1, rows / cols) ** 0.5
        return float(scale) * math.sqrt(max(rows, cols))

    # ---------------------------------------------------------------------------------------------------------
    # Change 2: micro-batch fluctuation statistics
    # ---------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def observe_microbatch(self):
        """Record the gradient of the micro-batch whose backward pass just finished (groups with iso=True only).

        Call it after every ``loss.backward()`` of a step, while ``p.grad`` accumulates over the micro-batches. The
        increment of ``p.grad`` since the previous call is that micro-batch's gradient; its squared row and column
        sums give the noise variance per output and per input channel. Statistics are cleared by ``step()``.
        """
        for group in self.param_groups:
            if not (group["use_muon"] and group["iso"]):
                continue
            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue
                st = self.state[p]
                if st.get("iso_count", 0) == 0:
                    st["iso_rs2"] = torch.zeros(p.size(0), device=p.device)
                    st["iso_cs2"] = torch.zeros(p.size(1), device=p.device)
                    st["iso_count"] = 0
                    st.pop("iso_prev", None)
                gk = p.grad.float() if "iso_prev" not in st else p.grad.float() - st["iso_prev"]
                gk2 = gk.square()
                st["iso_rs2"] += gk2.sum(1)
                st["iso_cs2"] += gk2.sum(0)
                st["iso_count"] += 1
                st["iso_prev"] = p.grad.float().clone()

    def _refresh_heats(self, group):
        """Turn this step's micro-batch statistics into row/column fluctuation estimates (EMA over steps) and clear the statistics."""
        todo = []
        for p in group["params"]:
            st = self.state[p]
            K = st.get("iso_count", 0)
            if p.grad is None or p.ndim != 2 or K == 0:
                continue
            gm = st["iso_prev"] / K                                      # mean micro-batch gradient
            rh = st["iso_rs2"] / K / p.size(1) - gm.square().sum(1) / p.size(1)
            ch = st["iso_cs2"] / K / p.size(0) - gm.square().sum(0) / p.size(0)
            todo.append((p, rh.clamp_min(0) + 1e-30, ch.clamp_min(0) + 1e-30))
            st["iso_count"] = 0
            st.pop("iso_prev", None)
        if not todo:
            if not self._warned_iso and any(p.grad is not None and p.ndim == 2 and "row_heat" not in self.state[p]
                                            for p in group["params"]):
                warnings.warn("iso=True but observe_microbatch() was never called: change ② (the fluctuation-calibrated "
                              "output budget) falls back to the plain polar factor until micro-batch statistics "
                              "arrive", stacklevel=3)
                self._warned_iso = True
            return
        if group["iso_sync"] and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            flat = torch.cat([torch.cat([rh, ch]) for _, rh, ch in todo])
            dist.all_reduce(flat)
            flat /= dist.get_world_size()
            out, i = [], 0
            for p, rh, ch in todo:
                n_r, n_c = rh.numel(), ch.numel()
                out.append((p, flat[i:i + n_r], flat[i + n_r:i + n_r + n_c]))
                i += n_r + n_c
            todo = out
        for p, rh, ch in todo:
            st = self.state[p]
            if "row_heat" not in st:
                st["row_heat"], st["col_heat"] = rh, ch
            else:
                st["row_heat"].lerp_(rh, 1 - group["iso_beta"])
                st["col_heat"].lerp_(ch, 1 - group["iso_beta"])

    # ---------------------------------------------------------------------------------------------------------
    # spike guard (optional safeguard, not one of the five changes)
    # ---------------------------------------------------------------------------------------------------------
    def _global_grad_norm(self) -> float:
        norms = [p.grad.detach().float().norm() for g in self.param_groups for p in g["params"] if p.grad is not None]
        return float(torch.stack(norms).norm()) if norms else 0.0

    def _spike_decision(self, grad_norm: float | None) -> bool:
        """True = this step is a spike: the slow mode must skip it."""
        if self.spike_guard <= 0.0:
            return False
        v = float(grad_norm) if grad_norm is not None else self._global_grad_norm()
        window = self._spike_sorted
        spike = not math.isfinite(v)
        if len(window) >= min(100, self.spike_window) and not spike:   # need a filled-enough window first
            spike = v > self.spike_guard * window[len(window) // 2]
        if math.isfinite(v):
            bisect.insort(window, v)
            self._spike_fifo.append(v)
            if len(self._spike_fifo) > self.spike_window:
                window.pop(bisect.bisect_left(window, self._spike_fifo.pop(0)))
        self.spike_frozen_steps += int(spike)
        return spike

    # ---------------------------------------------------------------------------------------------------------
    # step
    # ---------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None, grad_norm: float | None = None):
        """One optimizer step. ``grad_norm``: the pre-clipping global gradient norm, used only by the spike guard."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        spike = self._spike_decision(grad_norm)
        for group in self.param_groups:
            if group["use_muon"]:
                if group["iso"]:
                    self._refresh_heats(group)
                self._muon_group(group, spike)
            else:
                self._adamw_group(group)
        return loss

    def _polar(self, u: torch.Tensor, p: torch.Tensor, st, group):
        """Polar factor of the update input u, scaled; returns (update, P, (bi, ai) or None) for p.add_(update, -lr)."""
        ns, ns_steps, scale = group["ns"], group["ns_steps"], group["scale"]
        if group["iso"] and p.ndim == 2 and "row_heat" in st:
            bi = _iso_metric(st["row_heat"], group["iso_lam"], group["iso_clamp"]).rsqrt()[:, None]
            ai = _iso_metric(st["col_heat"], group["iso_lam"], group["iso_clamp"]).rsqrt()[None, :]
            P = newton_schulz((u * bi) * ai, ns, ns_steps)
            D = (P.float() * bi) * ai
            D = D * (min(u.size(-2), u.size(-1)) ** 0.5 / (D.norm() + 1e-12))
            D = D * self._scale(p, scale)
            return D.to(p.grad.dtype), P, (bi, ai)
        if u.ndim > 2:
            shape = u.shape
            P = newton_schulz(u.reshape(-1, shape[-2], shape[-1]), ns, ns_steps).reshape(shape)
        else:
            P = newton_schulz(u, ns, ns_steps)
        upd = P * self._scale(p, scale)
        return upd, P, None

    def _muon_group(self, group, spike: bool = False):
        lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
        bm_start = group["bm_start"]
        bf, bs = group["bm_beta_fast"], group["bm_beta_slow"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "step" not in st:
                st["step"] = 0
                st["momentum"] = torch.zeros_like(g)
                if bm_start is not None and group["bm_shadow"]:
                    st["m_fast"] = torch.zeros_like(g)        # 4. tracking from step 0: the modes exist from step 0
                    st["m_slow"] = torch.zeros_like(g)
            t = st["step"]
            st["step"] = t + 1
            if bm_start is None or (t < bm_start and not group["bm_shadow"]):
                u = self._single_ema_input(g, st, group)
            elif "m_fast" not in st:
                # 1. without tracking from step 0: create both modes from the live momentum; this step equals Muon's
                u = self._single_ema_input(g, st, group)
                st["m_fast"] = st["momentum"].clone()
                st["m_slow"] = st["momentum"].clone()
            else:
                gf = g.float()
                g_norm = gf.norm().clamp_min(1e-20)
                if group["osc_gate"]:
                    # 3. gate statistics, measured before the modes take this gradient
                    mf = st["m_fast"].float()
                    osc = (-(gf * mf).sum() / (g_norm * mf.norm().clamp_min(1e-20))).clamp_min(0.0)
                    drift = st["m_slow"].float().norm() / g_norm
                    if "osc" not in st:
                        st["osc"], st["drift"] = osc, drift
                    else:
                        st["osc"].lerp_(osc, 1 - group["osc_beta"])
                        st["drift"].lerp_(drift, 1 - group["osc_beta"])
                st["m_fast"].lerp_(g, 1 - bf)
                if not spike:                                    # spike guard: the slow mode skips outlier steps
                    st["m_slow"].lerp_(g, 1 - bs)
                if t < bm_start:
                    u = self._single_ema_input(g, st, group)     # 4. before the switch: Muon's update, the modes only track
                else:
                    if group["osc_gate"]:
                        rho_f = (1 - bf) / (1 + bf)
                        w = group["osc_wmax"] * (1 - rho_f * st["osc"] / st["drift"].clamp_min(1e-20)).clamp(0.0, 1.0)
                    else:
                        w = group["bm_w_fast"]
                    m_eff = w * st["m_fast"] + (1 - w) * st["m_slow"]
                    mu_t = mu
                    if group["fresh_cap"] > 0:
                        # 5. fresh-share cap: the current gradient's share of the update input <= phi
                        phi = group["fresh_cap"]
                        mu_t = torch.maximum(torch.tensor(mu, dtype=torch.float32, device=p.device),
                                             1.0 / (1.0 + phi / (1 - phi) * m_eff.float().norm() / g_norm))
                    u = g.lerp(m_eff.to(g.dtype), mu_t) if group["nesterov"] else m_eff.to(g.dtype)
                    st["momentum"].copy_(m_eff)                  # keep ``momentum`` = the memory read this step
            upd, _, _ = self._polar(u, p, st, group)
            p.mul_(1 - lr * wd)
            p.add_(upd, alpha=-lr)

    @staticmethod
    def _single_ema_input(g, st, group):
        m = st["momentum"]
        m.lerp_(g, 1 - group["momentum"])
        return g.lerp(m, group["momentum"]) if group["nesterov"] else m.clone()

    def _adamw_group(self, group):
        lr, wd, (b1, b2), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "step" not in st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(g)
                st["exp_avg_sq"] = torch.zeros_like(g)
            st["step"] += 1
            k = st["step"]
            st["exp_avg"].lerp_(g, 1 - b1)
            st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (st["exp_avg_sq"] / (1 - b2 ** k)).sqrt_().add_(eps)
            if wd:
                p.mul_(1 - lr * wd)
            p.addcdiv_(st["exp_avg"], denom, value=-lr / (1 - b1 ** k))
