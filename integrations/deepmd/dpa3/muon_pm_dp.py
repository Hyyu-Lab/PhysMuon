"""PhysMuon for DPA-3 models in deepmd-kit (PyTorch backend), installed as an optimizer hook.

This is the code that produced the paper's four DPA-3 results (MPtrj, SPICE2, SPICE2 with a large batch, ANI-1x).
Launch training with run_dp_pm.py instead of `dp`; input.json keeps optimizer type "Adam" and its learning-rate
schedule, and the hook swaps the optimizer object that deepmd builds. The optimizer is chosen by environment variables:

ARM=adam : deepmd's Adam, untouched.
ARM=muon : Muon, the baseline of the paper. Matrix parameters (2-D `matrix` weights of the MLP layers, excluding the
           type embedding) take Nesterov momentum (MUON_MOMENTUM, default 0.95) -> Newton-Schulz-5 orthogonalization,
           RMS-matched to Adam (update *= MUON_RMS * sqrt(max(m, n)), MUON_RMS = 0.2), so the Adam learning-rate schedule
           is shared unchanged. Everything else (biases, idt, 1-D, type embedding) stays on Adam with the same betas/eps.
ARM=bm   : PhysMuon's two-mode memory (change 1). From step BM_START on, the single momentum EMA is replaced by
           m_eff = BM_W_FAST * EMA_BM_BETA_FAST(g) + (1 - BM_W_FAST) * EMA_BM_BETA_SLOW(g); before BM_START the update is
           the ARM=muon update with MUON_MOMENTUM. Without PM_SHADOW=1 both modes start from the live momentum at
           BM_START, so that step's update equals the ARM=muon update.
PM=1     : (with ARM=bm) changes 3-5 on top of change 1, with the semantics of physmuon/optim.py:
           PM_GATE=1 gated readout, w = PM_WMAX * clip(1 - rho_f * o / d, 0, 1), rho_f = (1 - b_fast) / (1 + b_fast),
                     o and d smoothed with PM_OSC_BETA (default 0.9);
           PM_SHADOW=1 both modes (and the gate statistics) follow the gradient from step 0 (change 4);
           PM_PHI > 0 fresh-share cap on the Nesterov coefficient (change 5; PM_PHI=0 turns it off).
ISO=1 (with ARM=muon|bm): IsoMuon, the noise-calibrated diagonal metric of PR #370, on every Muon-routed matrix.
       Noise statistic as in the PR: the step's batch is cut into ISO_K contiguous groups of frames, each group is run
       forward/backward on its own (loss weighted by its frame share, so the accumulated gradient is the full-batch
       gradient), and per element  v = N/(K-1) [sum_k G_k^2/n_k - G^2/N]  estimates the batch-gradient noise variance;
       its row / column means are the fluctuation estimates (EMA ISO_BETA).  B = clamp((row_heat/mean)^lam, 1/c, c), A likewise, and
       D = B^-1/2 NS(B^-1/2 U A^-1/2) A^-1/2, re-aligned to the Frobenius norm of the NS output, then the same RMS factor.
       The split is needed because forces come from a second backward pass: a hook on a layer output would miss that part
       of the weight gradient.  ISO_LAM=0 keeps the update of ARM=muon or ARM=bm.
"""
import os
import torch

ARM = os.environ.get("ARM", "adam")
# PhysMuon changes 3-5 (semantics of physmuon/optim.py) with ARM=bm; PM=0 keeps change 1 alone.
#   PM_GATE=1: w = PM_WMAX*clip(1 - rho_f*o/d, 0, 1), o = EMA max(0,-cos(g, M_f)), d = EMA |M_s|/|g| (before the units take g)
#   PM_SHADOW=1: both units (and the gate statistics) track g from step 1; before BM_START the update is Muon's
#   PM_PHI>0: Nesterov coefficient mu_t = max(mu, 1/(1 + phi/(1-phi)*|M|/|g|))
PM = os.environ.get("PM", "0") == "1"
PM_GATE = os.environ.get("PM_GATE", "1") == "1"
PM_WMAX = float(os.environ.get("PM_WMAX", "0.6"))
PM_OSC_BETA = float(os.environ.get("PM_OSC_BETA", "0.9"))
PM_SHADOW = os.environ.get("PM_SHADOW", "1") == "1"
PM_PHI = float(os.environ.get("PM_PHI", "0.25"))
BM_START = int(os.environ.get("BM_START", "0"))
BM_BETA_FAST = float(os.environ.get("BM_BETA_FAST", "0.85"))
BM_BETA_SLOW = float(os.environ.get("BM_BETA_SLOW", "0.98"))
BM_W_FAST = float(os.environ.get("BM_W_FAST", "0.4385"))
MUON_MOMENTUM = float(os.environ.get("MUON_MOMENTUM", "0.95"))
MUON_RMS = float(os.environ.get("MUON_RMS", "0.2"))
HM_MOMENTUM = os.environ.get("HM_MOMENTUM", "")
RT_PERIOD = int(os.environ.get("RT_PERIOD", "10"))
RT_STREAM_K = int(os.environ.get("RT_STREAM_K", "4"))
# Generalized (time-dependent) two-mode schedule: "step:b1:b2:w;step:b1:b2:w;..."  From each listed optimizer step on,
# the two EMA modes run at rates b1, b2 and mix with weight w on mode 1.  The modes persist across phases; the first
# listed step seeds both from the single EMA (that step's update equals Muon's).  Muon (b1 = b2 = mu forever), a late
# deeper slow mode (b1 = b2) and a fixed two-mode mix are all special cases.  Empty: one phase from the BM_* variables.
BM_SCHED = os.environ.get("BM_SCHED", "")
ISO = os.environ.get("ISO", "0") == "1"
ISO_LAM = float(os.environ.get("ISO_LAM", "0.5"))
ISO_CLAMP = float(os.environ.get("ISO_CLAMP", "2.0"))
ISO_BETA = float(os.environ.get("ISO_BETA", "0.95"))
ISO_K = int(os.environ.get("ISO_K", "4"))
ISO_REPORT = int(os.environ.get("ISO_REPORT", "1000"))     # steps between fluctuation-dispersion reports
_ISO_PARAMS = []            # the Muon-routed matrices; filled when the optimizer is created
_ISO_LOG = {"calls": 0, "split": 0, "frames": 0}


def _iso_metric(heat):
    m = (heat / heat.mean().clamp_min(1e-30)).clamp_min(1e-30) ** ISO_LAM
    return m.clamp(1 / ISO_CLAMP, ISO_CLAMP)


def _iso_polar(u, st):
    bi = _iso_metric(st["row_heat"]).rsqrt()[:, None]
    ai = _iso_metric(st["col_heat"]).rsqrt()[None, :]
    P = newton_schulz5((u.float() * bi) * ai)
    D = (P * bi) * ai
    return D * (P.norm() / (D.norm() + 1e-12))


def bm_phases():
    if not BM_SCHED:
        return [(BM_START, BM_BETA_FAST, BM_BETA_SLOW, BM_W_FAST)]
    out = []
    for item in BM_SCHED.split(";"):
        if item.strip():
            t, b1, b2, w = item.split(":")
            out.append((int(t), float(b1), float(b2), float(w)))
    return sorted(out)


def newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class MuonAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8,
                 kinds=None, momentum=MUON_MOMENTUM, bm=False, bm_start=0, **_ignored):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, momentum=momentum)
        super().__init__(params, defaults)
        self.kinds = kinds or {}
        self.bm = bm
        self.phases = bm_phases() if bm else []
        self.t = 0
        self._announced = set()

    def _pm_input(self, p, g, st, mu):
        start, bf, bs, w_fixed = self.phases[0]
        m = st["momentum"]
        if PM_SHADOW and "m_fast" not in st:
            st["m_fast"] = torch.zeros_like(p); st["m_slow"] = torch.zeros_like(p)      # 4. the units exist from step 1
        if not PM_SHADOW and self.t < start:
            m.lerp_(g, 1 - mu)
            return g.lerp(m, mu)
        if "m_fast" not in st:
            # switch step without shadow onset: baseline recursion once more, then seed both units from it
            m.lerp_(g, 1 - mu)
            st["m_fast"] = m.clone(); st["m_slow"] = m.clone()
            return g.lerp(m, mu)
        gf = g.float()
        gn = gf.norm().clamp_min(1e-20)
        if PM_GATE:
            mf = st["m_fast"].float()
            osc = (-(gf * mf).sum() / (gn * mf.norm().clamp_min(1e-20))).clamp_min(0.0)
            drift = st["m_slow"].float().norm() / gn
            if "osc" not in st:
                st["osc"], st["drift"] = osc, drift
            else:
                st["osc"].lerp_(osc, 1 - PM_OSC_BETA); st["drift"].lerp_(drift, 1 - PM_OSC_BETA)
        st["m_fast"].lerp_(g, 1 - bf); st["m_slow"].lerp_(g, 1 - bs)
        if self.t < start:
            m.lerp_(g, 1 - mu)                                                        # 4. before the switch: Muon's update
            return g.lerp(m, mu)
        if PM_GATE:
            rho_f = (1 - bf) / (1 + bf)
            w = PM_WMAX * (1 - rho_f * st["osc"] / st["drift"].clamp_min(1e-20)).clamp(0.0, 1.0)
        else:
            w = w_fixed
        m_eff = w * st["m_fast"] + (1 - w) * st["m_slow"]
        mu_t = mu
        if PM_PHI > 0:
            mu_t = torch.maximum(torch.tensor(mu, dtype=torch.float32, device=g.device),
                                 1.0 / (1.0 + PM_PHI / (1 - PM_PHI) * m_eff.float().norm() / gn))
        m.copy_(m_eff)
        return g.lerp(m_eff, mu_t)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self.t += 1
        ph = None                                       # current two-mode phase (start, b1, b2, w), None before the first
        for k, cand in enumerate(self.phases):
            if self.t >= cand[0]:
                ph = cand
                if k not in self._announced:
                    print(f"[muon_bm_dp] BM phase {k} from t={self.t}: b1={cand[1]} b2={cand[2]} w1={cand[3]}"
                          + ("  (units seeded from the single EMA; kernel active from the next step)" if k == 0 else ""), flush=True)
                    self._announced.add(k)
        for group in self.param_groups:
            lr, wd, (b1, b2), eps, mu = group["lr"], group["weight_decay"], group["betas"], group["eps"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if self.kinds.get(id(p), "adam") == "muon":
                    if "momentum" not in st:
                        st["momentum"] = torch.zeros_like(p)
                    if ISO and getattr(p, "iso_fresh", False):
                        if "row_heat" not in st:
                            st["row_heat"] = p.iso_rh.clone(); st["col_heat"] = p.iso_ch.clone()
                        else:
                            st["row_heat"].lerp_(p.iso_rh, 1 - ISO_BETA); st["col_heat"].lerp_(p.iso_ch, 1 - ISO_BETA)
                        p.iso_fresh = False
                    m = st["momentum"]
                    if PM and self.bm:
                        u = self._pm_input(p, g, st, mu)
                    else:
                        m.lerp_(g, 1 - mu)
                    if PM and self.bm:
                        pass
                    elif ph is not None and "m_fast" not in st:
                        # switch step: take the baseline update and seed both modes from the
                        # just-advanced single EMA, so this step is bit-identical to Muon
                        # and the two-mode recursion starts on the next step.
                        st["m_fast"] = m.clone(); st["m_slow"] = m.clone()
                        u = g.lerp(m, mu)
                    elif ph is not None:
                        _, rb1, rb2, w1 = ph
                        st["m_fast"].lerp_(g, 1 - rb1); st["m_slow"].lerp_(g, 1 - rb2)
                        m_eff = w1 * st["m_fast"] + (1 - w1) * st["m_slow"]
                        u = g.lerp(m_eff, mu)
                    else:
                        u = g.lerp(m, mu)
                    if ISO and ISO_LAM > 0 and "row_heat" in st:
                        D = _iso_polar(u, st) * (MUON_RMS * max(p.size(0), p.size(1)) ** 0.5)
                    else:
                        D = newton_schulz5(u) * (MUON_RMS * max(p.size(0), p.size(1)) ** 0.5)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(D.to(p.dtype), alpha=-lr)
                else:
                    if "exp_avg" not in st:
                        st["exp_avg"] = torch.zeros_like(p); st["exp_avg_sq"] = torch.zeros_like(p); st["step"] = 0
                    st["step"] += 1
                    ea, es, k = st["exp_avg"], st["exp_avg_sq"], st["step"]
                    ea.lerp_(g, 1 - b1); es.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bc1 = 1 - b1 ** k; bc2 = 1 - b2 ** k
                    denom = (es / bc2).sqrt_().add_(eps)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(ea, denom, value=-lr / bc1)
        if ISO and self.t % ISO_REPORT == 0:
            _iso_report(self)
        return loss


def _iso_report(opt):
    ls = {"row": [], "col": []}; cl = {"row": [], "col": []}
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, {})
            if "row_heat" not in st:
                continue
            for side in ("row", "col"):
                h = st[side + "_heat"]
                ls[side].append(float(h.clamp_min(1e-30).log().std()))
                m = _iso_metric(h)
                cl[side].append(float(((m <= 1 / ISO_CLAMP + 1e-6) | (m >= ISO_CLAMP - 1e-6)).float().mean()))
    n = len(ls["row"])
    if n:
        print(f"[muon_bm_dp] iso t={opt.t} matrices={n} split_steps={_ISO_LOG['split']}/{_ISO_LOG['calls']} "
              f"mean_frames={_ISO_LOG['frames'] / max(1, _ISO_LOG['split']):.1f} "
              f"log_heat_std row {sum(ls['row']) / n:.3f} col {sum(ls['col']) / n:.3f} "
              f"clamped row {sum(cl['row']) / n:.3f} col {sum(cl['col']) / n:.3f}", flush=True)


def _slice_frames(v, a, b, nf):
    if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == nf:
        return v[a:b]
    return v


def _merge_more(parts):
    out = {}
    for key, v0 in parts[0][1].items():
        vals = [(w, more.get(key)) for w, more in parts]
        try:
            if key.startswith("rmse"):
                out[key] = sum(w * v ** 2 for w, v in vals) ** 0.5
            else:
                out[key] = sum(w * v for w, v in vals)
        except Exception:
            out[key] = v0
    return out


def _merge_pred(preds):
    out = {}
    for key, v0 in preds[0].items():
        vs = [p.get(key) for p in preds]
        if all(torch.is_tensor(x) and x.dim() > 0 for x in vs):
            try:
                out[key] = torch.cat([x.detach() for x in vs], 0)
                continue
            except Exception:
                pass
        out[key] = v0
    return out


def install_microbatch():
    """Training calls of the model wrapper run the batch as ISO_K frame groups (forward + backward each) and leave the
    per-step fluctuation estimates on the Muon-routed matrices; the returned loss is a detached leaf, so the trainer's backward() adds
    nothing and its clipping / optimizer step see the full-batch gradient."""
    from deepmd.pt.train import wrapper as W
    orig = W.ModelWrapper.forward
    shown = []

    def forward(self, coord, atype, spin=None, box=None, cur_lr=None, label=None, task_key=None, skip_loss=False,
                do_atomic_virial=False, fparam=None, aparam=None, charge_spin=None):
        extra = dict(spin=spin, box=box, fparam=fparam, aparam=aparam, charge_spin=charge_spin)
        if not (self.training and torch.is_grad_enabled() and label is not None and not skip_loss and _ISO_PARAMS):
            return orig(self, coord, atype, cur_lr=cur_lr, label=label, task_key=task_key, skip_loss=skip_loss,
                        do_atomic_virial=do_atomic_virial, **extra)
        nf = coord.shape[0]
        K = min(ISO_K, nf)
        _ISO_LOG["calls"] += 1
        if not shown:
            shown.append(1)
            print("[muon_bm_dp] iso split: inputs " + str({k: tuple(v.shape) for k, v in dict(coord=coord, atype=atype, **extra).items() if torch.is_tensor(v)})
                  + " labels " + str({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in label.items()}), flush=True)
        if K < 2:        # a one-frame batch: no noise estimate this step (the fluctuation estimates keep their EMA)
            return orig(self, coord, atype, cur_lr=cur_lr, label=label, task_key=task_key, skip_loss=skip_loss,
                        do_atomic_virial=do_atomic_virial, **extra)
        for p in _ISO_PARAMS:
            assert p.grad is None, "iso split expects gradients cleared (set_to_none) before the training forward"
        bounds = [(nf * k) // K for k in range(K + 1)]
        rs = [torch.zeros(p.shape[0], device=p.device) for p in _ISO_PARAMS]
        cs = [torch.zeros(p.shape[1], device=p.device) for p in _ISO_PARAMS]
        prev = [None] * len(_ISO_PARAMS)
        preds, mores, total = [], [], 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            pred, loss, more = orig(self, coord[a:b], atype[a:b], cur_lr=cur_lr,
                                    label={k: _slice_frames(v, a, b, nf) for k, v in label.items()},
                                    task_key=task_key, skip_loss=skip_loss, do_atomic_virial=do_atomic_virial,
                                    **{k: _slice_frames(v, a, b, nf) for k, v in extra.items()})
            w = (b - a) / nf
            (loss * w).backward()
            with torch.no_grad():
                for j, p in enumerate(_ISO_PARAMS):
                    if p.grad is None:
                        continue
                    g = p.grad.detach().clone()
                    Gk = (g if prev[j] is None else g - prev[j]).float().square()
                    rs[j].add_(Gk.sum(1), alpha=1.0 / (b - a)); cs[j].add_(Gk.sum(0), alpha=1.0 / (b - a))
                    prev[j] = g
            total = total + loss.detach() * w
            preds.append(pred); mores.append((w, more))
        with torch.no_grad():
            for j, p in enumerate(_ISO_PARAMS):
                if p.grad is None:
                    continue
                G2 = p.grad.float().square()
                rh = (rs[j] - G2.sum(1) / nf) * (nf / (K - 1)) / p.shape[1]
                ch = (cs[j] - G2.sum(0) / nf) * (nf / (K - 1)) / p.shape[0]
                p.iso_rh = rh.clamp_min(0) + 1e-30; p.iso_ch = ch.clamp_min(0) + 1e-30; p.iso_fresh = True
        _ISO_LOG["split"] += 1; _ISO_LOG["frames"] += nf
        return _merge_pred(preds), total.clone().requires_grad_(True), _merge_more(mores)

    W.ModelWrapper.forward = forward
    print(f"[muon_bm_dp] iso installed: lam={ISO_LAM} clamp={ISO_CLAMP} beta={ISO_BETA} K={ISO_K}", flush=True)


def _may_rt_classes():
    from rt_may.hybrid_muon import HybridMuonOptimizer
    from rt_may.hybrid_rt43_muon import HybridRT43MuonDualCleanOptimizer

    class DualCleanWithReplica(HybridRT43MuonDualCleanOptimizer):
        """On the steps where needs_replica_grads() (every rt_period-th step) take the next training batch, compute its
        gradient with the current loss prefactors and the same global-norm clip as the trainer, and inject it."""
        trainer = None
        n_replica = 0

        def step(self, closure=None):
            tr = self.trainer
            if tr is not None and self.needs_replica_grads():
                inp, lab, _ = tr._next_training_batch("Default")
                params = [p for p in tr.wrapper.parameters() if p.requires_grad]
                with torch.enable_grad():
                    _, loss, _ = tr.wrapper(**inp, cur_lr=tr.scheduler.get_last_lr()[0], label=lab, task_key="Default")
                    grads = torch.autograd.grad(loss, params, allow_unused=True)
                if tr.gradient_max_norm > 0.0:
                    total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads if g is not None]))
                    coef = (tr.gradient_max_norm / (total + 1e-6)).clamp(max=1.0)
                    grads = [None if g is None else g.detach() * coef for g in grads]
                self.set_replica_grads(dict(zip(params, grads)))
                self.n_replica += 1
            return super().step(closure)

    return HybridMuonOptimizer, DualCleanWithReplica


def install_rt():
    from deepmd.pt.train import training as T
    HM, RT = _may_rt_classes()
    mom = float(HM_MOMENTUM) if HM_MOMENTUM else (0.95 if ARM == "hmuon" else 0.98)

    def patched(self, optimizer_class, **kwargs):
        common = dict(lr=kwargs["lr"], momentum=mom, weight_decay=0.001, adam_betas=(0.9, 0.95), lr_adjust=0.0,
                      lr_adjust_coeff=0.18, muon_mode="slice", named_parameters=tuple(self.wrapper.named_parameters()),
                      enable_gram=False, flash_muon=False, magma_muon=True)
        if ARM == "hmuon":
            opt = HM(self.wrapper.parameters(), **common)
        else:
            opt = RT(self.wrapper.parameters(), **common, rt_stream_k=RT_STREAM_K, rt_power_iters=1, rt_basis_refresh_period=1,
                     rt_max_blocks=0, rt_period=RT_PERIOD, rt_warmup_steps=0, rt_warmup_period=1, rt_tail_start_step=0,
                     rt_tail_period=1, rt_lr_switch=0.05, rt_beta_grid=(0.001, 0.005, 0.02, 0.1, 0.5), rt_hard_delta=0.35,
                     rt_soft_delta=0.35, rt_lambda_noise=2.0, rt_lambda_consistency=0.25, rt_lambda_diff=0.5,
                     rt_lambda_uncert=0.5, rt_lambda_tube=0.05, rt_phi_threshold_hard=0.0, rt_phi_threshold_soft=0.0,
                     rt_soft_tau=0.01, rt_delta_max_norm=(0.0 if ARM == "rtavg" else 0.25), rt_replica_rms_match=True,
                     rt_dual_average_muon_base=True, rt_replica_period=1, rt_temporal_fill=False,
                     rt_diag_dir="optimizer_diag", rt_log_interval=RT_PERIOD, rt_name=ARM)
            opt.trainer = self
        opt._build_param_routing()
        r = opt._routing[0]
        n = lambda k: sum(e["param"].numel() for e in r[k])
        print(f"[muon_bm_dp] arm={ARM} momentum={mom} lr={kwargs.get('lr')} muon_params={n('muon_params')} "
              f"adam_params={n('adam_no_decay')} adamw_params={n('adam_decay')} muon_blocks={len(r['muon_params'])}"
              + (f" rt_period={RT_PERIOD} stream_k={RT_STREAM_K}" if ARM != "hmuon" else ""), flush=True)
        return opt

    T.Trainer._create_optimizer = patched
    print(f"[muon_bm_dp] installed arm={ARM}", flush=True)


def install():
    if ARM == "adam":
        print("[muon_bm_dp] arm=adam (stock deepmd Adam)", flush=True)
        return
    if ARM in ("hmuon", "rt", "rtavg"):
        assert not ISO, "ISO is wired only for ARM=muon|bm"
        return install_rt()
    assert ARM in ("muon", "bm"), ARM
    from deepmd.pt.train import training as T
    orig = T.Trainer._create_optimizer

    def patched(self, optimizer_class, **kwargs):
        kinds, n_muon, n_adam = {}, 0, 0
        for name, p in self.wrapper.named_parameters():
            if p.ndim == 2 and min(p.shape) >= 2 and "type_embedding" not in name:
                kinds[id(p)] = "muon"; n_muon += p.numel()
            else:
                kinds[id(p)] = "adam"; n_adam += p.numel()
        kwargs.pop("fused", None)
        print(f"[muon_bm_dp] arm={ARM} muon_params={n_muon} adam_params={n_adam} lr={kwargs.get('lr')} wd={kwargs.get('weight_decay')} "
              f"mu={MUON_MOMENTUM} bm_phases={bm_phases() if ARM == 'bm' else []}", flush=True)
        if ISO:
            _ISO_PARAMS[:] = [p for _, p in self.wrapper.named_parameters() if kinds[id(p)] == "muon"]
        return orig(self, MuonAdam, kinds=kinds, bm=(ARM == "bm"), bm_start=BM_START, **kwargs)

    T.Trainer._create_optimizer = patched
    if ISO:
        install_microbatch()
    print(f"[muon_bm_dp] installed arm={ARM} iso={int(ISO)}", flush=True)
