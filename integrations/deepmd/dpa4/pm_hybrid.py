"""PhysMuon inside deepmd-kit's own HybridMuon optimizer, for DPA4 models.

This is the code that produced the paper's DPA4-Mini and DPA4-Pro (Matbench Discovery) results. Launch training with
run_dp_dpa4_pm.py instead of `dp`. ARM=bm with BM_SCHED switches on PhysMuon's two-mode memory (change 1); ISO=1 is
change 2 (see below); PM=1 adds changes 3-5 (PM=0 keeps change 1 alone):
  PM_GATE=1  oscillation-gated readout per Muon slice: before the units take g, o = max(0,-cos(g,u_fast)) and
             d = |u_slow|/|g| are EMAs (PM_OSC_BETA); w = PM_WMAX*clip(1 - rho_f*o/d, 0, 1), rho_f = (1-b1)/(1+b1)
  PM_SHADOW=1 both units (and the gate statistics) track g from the first step; before the first BM_SCHED step the
             update is HybridMuon's own, from then on it reads the units (no seeding from the momentum buffer)
  PM_PHI>0   Nesterov coefficient per slice mu_t = max(mu, 1/(1 + phi/(1-phi)*|M|/|g|)), M the read memory


Apart from ISO=1 (below), only the Muon momentum recursion changes; routing, Newton-Schulz / Gram orthogonalisation, Magma-lite damping,
update-RMS scaling, weight decay, the Adam paths and the learning-rate schedule are deepmd's HybridMuon unchanged.
Selected by env ARM=hmuon (stock HybridMuon, nothing patched) | bm.

BM_SCHED="step:b1:b2:w;step:b1:b2:w;..."  From each listed optimizer step on, the Muon momentum is the convex
combination w*EMA_b1(g) + (1-w)*EMA_b2(g) of two units that persist across phases; the first listed step seeds both
units from the live single-EMA momentum, so that step's update equals HybridMuon's.  The Nesterov mix keeps
HybridMuon's momentum coefficient.  The momentum buffer is overwritten with the effective kernel output, so
Magma-lite measures its momentum-gradient alignment on the momentum that is actually used.

Experimental, not used in the paper -- time-temperature stretching (BM_TTS_P > 0): the two memory times of the current phase, tau_i = b_i / (1 - b_i), are stretched
with the learning-rate drop, tau_i(t) = min(BM_TTS_CAP, tau_i * (lr_max / lr(t)) ** BM_TTS_P), lr_max = the largest
lr seen so far (so the ratio is 1 through warmup and the stable phase).  b1 = b2 gives a single stretched EMA.
Restart from a checkpoint: BM_T0 = the optimizer step the run resumes at, BM_LRMAX = the peak lr if resuming in decay.

GRAD_LOG=1 (any ARM value, diagnostics only, does not change the update): writes gradnorm.txt in the run directory with
"step  norm_seen_by_optimizer  pre_clip_norm" per step (pre_clip_norm = -1 when training.gradient_max_norm is off).

Restarts keep the two memory modes: they live in the optimizer state of each Muon parameter (keys bm_u0 / bm_u1), so
deepmd writes them into every checkpoint and --restart loads them back; a checkpoint without them (a HybridMuon run)
seeds both modes from its momentum buffer at the first two-mode step, as at the switch step.

Spike guard (BM_GATE_K > 0, ARM=bm): DPA-4.0.1-Pro's per-step gradient norm is heavy-tailed (pathological near-contact
frames give pre-clip norms of 1e6-1e12 against a median of ~12.5; every step is clipped to the training's
gradient_max_norm anyway, so a spike step's clipped gradient is a pure outlier direction).  The slow unit's ~200-step
memory then holds a spike for its whole window, and with the noise averaged out its norm is small, so the spike
direction takes a large share of the orthogonalised update for many steps; the model is pushed off on other frames and
aftershock spikes follow.  With the gate on, a step whose pre-clip norm exceeds BM_GATE_K x the running median of the
last BM_GATE_WIN steps (non-finite counts as a spike) leaves the slow unit unchanged; the fast unit, whose memory is as
short as HybridMuon's, and the Nesterov term see it as before.  BM_GATE_MED0 seeds the median right after a (re)start.

GRAD_ACC=n (any ARM value): every optimizer step accumulates n per-rank micro-batches, so the gradient is the mean over k*n
per-rank batches on k GPUs, clipped afterwards: k GPUs run the k*n-GPU recipe at n times the step time.  Every
micro-batch is a normal synchronised DDP iteration (DDP averages the accumulated .grad, so after the second backward the
gradient is avg_1 + avg_2): no_sync is not usable here, because deepmd's display-time validation runs the DDP wrapper
with grad enabled, which arms the reducer, and a no_sync backward right after it marks parameters ready against the
validation graph ("marked as ready twice").  deepmd 3.2.0's PyTorch trainer has no accumulation (acc_freq is only an
argcheck entry), so Trainer._run is re-compiled with the micro-batch loop inserted before its backward call.

ISO=1 (any ARM value, DDP on k >= 2 GPUs): change 2, IsoMuon's noise-calibrated diagonal metric (PR #370), on every
Muon-routed matrix slice. Fluctuation estimate without splitting the batch: the k ranks hold independent gradients of
the step; per element the rank mean of (g_rank - g_mean)^2 is (k-1)/k of one rank's gradient variance (for k = 2,
((g_0 - g_1)/2)^2, an unbiased estimate of the variance of the averaged gradient), a constant factor that cancels in the
metric. A DDP communication hook takes it before the averaged gradient is written back; its row / column means (the
row and column fluctuation estimates) are averaged over the ranks (identical metric on all ranks) and smoothed with
EMA ISO_BETA. With GRAD_ACC > 1 only the last micro-batch's rank differences enter.  Metric B = clamp((row_var/mean)^lam,
1/c, c), A likewise over columns; the update fed to Newton-Schulz is B^-1/2 U A^-1/2 and the orthogonalised output P
becomes B^-1/2 P A^-1/2 re-aligned to P's Frobenius norm per slice, before HybridMuon's RMS scale and Magma damping
(deepmd's step / Gram-bucket code is re-compiled with one inserted call at each of its two NS exits).
Unlike the PR's in-batch split, the ranks may draw different structure buckets, so the estimate also contains the
bucket-to-bucket spread of the gradient.

"""
import atexit
import os
import torch

ARM = os.environ.get("ARM", "hmuon")
BM_SCHED = os.environ.get("BM_SCHED", "")
GRAD_LOG = os.environ.get("GRAD_LOG", "0") == "1"
BM_GATE_K = float(os.environ.get("BM_GATE_K", "0"))         # spike gate threshold (x running median of pre-clip norm); 0 = off
BM_GATE_WIN = int(os.environ.get("BM_GATE_WIN", "1000"))     # running-median window (optimizer steps)
BM_GATE_MED0 = float(os.environ.get("BM_GATE_MED0", "0"))    # median prior right after a (re)start (0 = gate only after 100 steps)
BM_GATE_REPORT = int(os.environ.get("BM_GATE_REPORT", "5000"))  # report interval (optimizer steps)
BM_NESTEROV_MU = os.environ.get("BM_NESTEROV_MU", "")
BM_MAGMA_HM = os.environ.get("BM_MAGMA_HM", "0") == "1"
MAGMA_LOG = int(os.environ.get("MAGMA_LOG", "0"))
BM_TTS_P = float(os.environ.get("BM_TTS_P", "0"))
BM_TTS_CAP = float(os.environ.get("BM_TTS_CAP", "1000"))
REJ_KAPPA = float(os.environ.get("REJ_KAPPA", "0"))
REJ_MODE = os.environ.get("REJ_MODE", "pass")
REJ_SHARP = int(os.environ.get("REJ_SHARP", "4"))
ISO = os.environ.get("ISO", "0") == "1"
ISO_LAM = float(os.environ.get("ISO_LAM", "0.5"))
ISO_CLAMP = float(os.environ.get("ISO_CLAMP", "2.0"))
ISO_BETA = float(os.environ.get("ISO_BETA", "0.95"))
ISO_REPORT = int(os.environ.get("ISO_REPORT", "2000"))
NOISE_LOG = os.environ.get("NOISE_LOG", "0") == "1"
# ISO_SPIKE_SKIP=1: exclude spike-gated and non-finite steps from change 2's fluctuation estimate.
ISO_SPIKE_SKIP = os.environ.get("ISO_SPIKE_SKIP", "0") == "1"
PM = os.environ.get("PM", "0") == "1"
PM_GATE = os.environ.get("PM_GATE", "1") == "1"
PM_WMAX = float(os.environ.get("PM_WMAX", "0.6"))
PM_OSC_BETA = float(os.environ.get("PM_OSC_BETA", "0.9"))
PM_SHADOW = os.environ.get("PM_SHADOW", "1") == "1"
PM_PHI = float(os.environ.get("PM_PHI", "0.25"))
_PMS = {}   # id(momentum buffer) -> [osc EMA, drift EMA]
_ISO = {"raw": {}, "heat": {}, "cur": {}, "buf2p": {}, "hooks": 0, "syncs": 0, "pre": 0, "post": 0, "layout_ok": None,
        "noise": None, "gmean2": None, "batch": (0, 0), "nlog": [], "skipped": 0}


def _eff_view(shape):
    """(slices, rows, cols) of HybridMuon's slice-mode matrix view: singleton dims dropped, last two dims are the matrix."""
    eff = [int(d) for d in shape if d != 1]
    if len(eff) < 2:
        return None
    bs = 1
    for d in eff[:-2]:
        bs *= d
    return bs, eff[-2], eff[-1]


def _iso_metric(heat):
    m = (heat / heat.mean(-1, keepdim=True).clamp_min(1e-30)).clamp_min(1e-30) ** ISO_LAM
    return m.clamp(1.0 / ISO_CLAMP, ISO_CLAMP)


def _iso_comm_hook(state, bucket):
    """allreduce (mean) as DDP's default hook, plus the per-element (g_rank - g_mean)^2 row / column means of every
    matrix-shaped parameter in the bucket (rank-local; averaged over ranks in _iso_before_step)."""
    import torch.distributed as dist
    buf = bucket.buffer()
    world = dist.get_world_size()
    local = buf.detach().clone()
    fut = dist.all_reduce(buf.div_(world), async_op=True).get_future()
    params = bucket.parameters()

    def after(f):
        mean = f.value()[0]
        with torch.no_grad():
            if _ISO["layout_ok"] is None:
                _ISO["layout_ok"] = sum(p.numel() for p in params) == mean.numel()
                if not _ISO["layout_ok"]:
                    print(f"[bm_hybrid] iso: bucket layout mismatch ({sum(p.numel() for p in params)} vs {mean.numel()}), "
                          "heat estimation disabled", flush=True)
            if _ISO["layout_ok"]:
                off = 0
                for p in params:
                    n = p.numel()
                    v3 = _eff_view(p.shape)
                    if v3 is not None:
                        d = (local[off:off + n] - mean[off:off + n]).float().view(v3)
                        v = d * d
                        _ISO["raw"][id(p)] = (v.mean(-1), v.mean(-2))
                        if NOISE_LOG:
                            m2 = mean[off:off + n].float().square().sum()
                            _ISO["noise"] = v.sum() if _ISO["noise"] is None else _ISO["noise"] + v.sum()
                            _ISO["gmean2"] = m2 if _ISO["gmean2"] is None else _ISO["gmean2"] + m2
                    off += n
        return mean

    return fut.then(after)


def _iso_before_step(opt):
    """Average this step's rank-local fluctuation estimates over the ranks (identical metric everywhere), fold them into the EMA,
    and map HybridMuon's momentum buffers back to their parameters for the pre-scaling."""
    import torch.distributed as dist
    _ISO["buf2p"] = {id(st["momentum_buffer"]): p for p, st in opt.state.items() if "momentum_buffer" in st}
    raw = _ISO["raw"]
    if not raw:
        return
    order = [p for g in opt.param_groups for p in g["params"] if id(p) in raw]
    flat = torch.cat([torch.cat([raw[id(p)][0].reshape(-1), raw[id(p)][1].reshape(-1)]) for p in order])
    if ISO_SPIKE_SKIP:                               # last entry: this rank's spike-gate flag (summed over ranks below)
        flat = torch.cat([flat, torch.ones(1, device=flat.device, dtype=flat.dtype) if _STATE.get("gate")
                          else torch.zeros(1, device=flat.device, dtype=flat.dtype)])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flat)
        flat.div_(dist.get_world_size())
    if ISO_SPIKE_SKIP:
        spike = bool(flat[-1] > 0) or not bool(torch.isfinite(flat).all())   # identical on every rank after all_reduce
        flat = flat[:-1]
        if spike:                                    # the fluctuation estimate comes from the pre-clip gradient; a spike step's d^2
            _ISO["raw"] = {}                         # overflows fp32 (inf/inf = NaN in _iso_metric), so it is left out,
            _ISO["skipped"] += 1                     # as the slow unit leaves it out
            return
    off = 0
    for p in order:
        r0, c0 = raw[id(p)]
        rh = flat[off:off + r0.numel()].view_as(r0).clamp_min(1e-30); off += r0.numel()
        ch = flat[off:off + c0.numel()].view_as(c0).clamp_min(1e-30); off += c0.numel()
        h = _ISO["heat"].get(id(p))
        if h is None:
            _ISO["heat"][id(p)] = [rh.clone(), ch.clone()]
        else:
            h[0].lerp_(rh, 1.0 - ISO_BETA); h[1].lerp_(ch, 1.0 - ISO_BETA)
    _ISO["raw"] = {}
    _ISO["syncs"] += 1
    if NOISE_LOG and _ISO["noise"] is not None:
        nf, na = _ISO["batch"]
        row = torch.tensor([float(nf), float(na)], device=flat.device)
        if dist.is_available() and dist.is_initialized():
            allrows = [torch.zeros_like(row) for _ in range(dist.get_world_size())]
            dist.all_gather(allrows, row)
            row = torch.cat(allrows)
        _ISO["nlog"].append(torch.cat([torch.tensor([float(_STATE["t"])], device=flat.device), row,
                                       _ISO["noise"].reshape(1), _ISO["gmean2"].reshape(1)]))
        _ISO["noise"] = None; _ISO["gmean2"] = None
        if len(_ISO["nlog"]) >= 500:
            _flush_noise()


def _flush_noise():
    if _ISO["nlog"] and int(os.environ.get("RANK", "0")) == 0:
        with open("noise.txt", "a") as f:
            for r in torch.stack(_ISO["nlog"]).cpu().tolist():
                f.write(" ".join(f"{x:.6g}" for x in r) + "\n")
    _ISO["nlog"] = []


def _iso_pre(updates, momentum_buffers):
    """Nesterov updates -> B^-1/2 U A^-1/2 per matrix slice; remembers (B^-1/2, A^-1/2) for the post-scaling."""
    out = []
    for u, b in zip(updates, momentum_buffers, strict=True):
        p = _ISO["buf2p"].get(id(b))
        h = _ISO["heat"].get(id(p)) if p is not None else None
        v3 = _eff_view(u.shape)
        if h is None or v3 is None or tuple(h[0].shape) != (v3[0], v3[1]) or tuple(h[1].shape) != (v3[0], v3[2]):
            out.append(u)
            continue
        bi, ai = _iso_metric(h[0]).rsqrt(), _iso_metric(h[1]).rsqrt()
        _ISO["cur"][id(p)] = (bi, ai)
        out.append((u.float().reshape(v3) * bi[:, :, None] * ai[:, None, :]).to(u.dtype).reshape(u.shape))
        _ISO["pre"] += 1
    return out


def _ISO_POST(param, orth_slice):
    """Called by the re-compiled HybridMuon code on each orthogonalised (slices, rows, cols) block."""
    cur = _ISO["cur"].get(id(param))
    if cur is None or tuple(orth_slice.shape[-2:]) != (cur[0].shape[-1], cur[1].shape[-1]):
        return orth_slice
    bi, ai = cur
    P = orth_slice.float()
    D = P * bi[:, :, None] * ai[:, None, :]
    D = D * (P.norm(dim=(-2, -1), keepdim=True) / (D.norm(dim=(-2, -1), keepdim=True) + 1e-12))
    _ISO["post"] += 1
    return D.to(orth_slice.dtype)


def _iso_report(t):
    if int(os.environ.get("RANK", "0")) != 0 or not _ISO["heat"]:
        return
    ls, cl = {0: [], 1: []}, {0: [], 1: []}
    for h in _ISO["heat"].values():
        for k in (0, 1):
            ls[k].append(float(h[k].log().std(-1).mean()))
            m = _iso_metric(h[k])
            cl[k].append(float(((m <= 1 / ISO_CLAMP + 1e-6) | (m >= ISO_CLAMP - 1e-6)).float().mean()))
    n = len(ls[0])
    print(f"[bm_hybrid] iso step {t}: tensors={n} syncs={_ISO['syncs']} spike_skipped={_ISO['skipped']} pre={_ISO['pre']} post={_ISO['post']} "
          f"log_heat_std row {sum(ls[0]) / n:.3f} col {sum(ls[1]) / n:.3f} clamped row {sum(cl[0]) / n:.3f} col {sum(cl[1]) / n:.3f}",
          flush=True)


def _install_iso(H):
    """Re-compile HybridMuon.step and _process_merged_gram_buckets with one _ISO_POST call at each NS exit (in the live
    module namespace, so later patches of module globals stay visible), and put the fluctuation hook on every DDP instance."""
    import __future__
    import inspect
    import textwrap
    H._ISO_POST = _ISO_POST
    for name, anchor, where in (("step", "orth_slice = orthogonalized[off : off + bs]", "after"),
                                ("_process_merged_gram_buckets", "# Per-entry scale (depends on original max(rows, cols)).", "before")):
        src = textwrap.dedent(inspect.getsource(getattr(H.HybridMuonOptimizer, name)))
        assert "super(" not in src and "__class__" not in src, name
        lines = src.split("\n")
        hits = [i for i, line in enumerate(lines) if line.strip() == anchor]
        assert len(hits) == 1, (name, anchor, len(hits))
        i = hits[0]
        ind = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        lines.insert(i + 1 if where == "after" else i, ind + 'orth_slice = _ISO_POST(entry["param"], orth_slice)')
        code = compile("\n".join(lines), H.__file__, "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
        exec(code, H.__dict__)
        fn = H.__dict__[name]
        if name == "step" and "no_grad" not in src.split("def step")[0]:
            fn = torch.no_grad()(fn)                 # getsource dropped the decorator: restore it
        setattr(H.HybridMuonOptimizer, name, fn)
    from torch.nn.parallel import DistributedDataParallel as DDP
    orig_init = DDP.__init__

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        import torch.distributed as dist
        # k >= 2 ranks: the rank-averaged (g_rank - g_mean)^2 is (k-1)/k of one rank's gradient
        # variance, a constant factor that cancels in the metric (variance / its mean over channels); k = 2 is unchanged.
        assert dist.get_world_size() >= 2, "ISO heat from rank differences needs at least 2 ranks"
        self.register_comm_hook(state=None, hook=_iso_comm_hook)
        _ISO["hooks"] += 1
        print(f"[bm_hybrid] iso: heat hook registered on DDP (world {dist.get_world_size()})", flush=True)

    DDP.__init__ = init
    if NOISE_LOG:
        from deepmd.pt.train import training as T
        orig_next = T.Trainer._next_training_batch

        def next_batch(self, *args, **kwargs):
            out = orig_next(self, *args, **kwargs)
            c = out[0].get("coord")
            if torch.is_tensor(c):
                _ISO["batch"] = (int(c.shape[0]), int(c.shape[1]) if c.ndim >= 2 else 0)
            return out
        T.Trainer._next_training_batch = next_batch
    print(f"[bm_hybrid] iso installed: lam={ISO_LAM} clamp={ISO_CLAMP} beta={ISO_BETA} noise_log={int(NOISE_LOG)} spike_skip={int(ISO_SPIKE_SKIP)} "
          f"(post-scaling compiled into step and _process_merged_gram_buckets)", flush=True)


def bm_phases():
    out = []
    for item in BM_SCHED.split(";"):
        if item.strip():
            t, b1, b2, w = item.split(":")
            out.append((int(t), float(b1), float(b2), float(w)))
    return sorted(out)


_STATE = {"t": int(os.environ.get("BM_T0", "0")), "phases": bm_phases(), "announced": set(), "log": [], "preclip": None,
          "lr": None, "lrmax": float(os.environ.get("BM_LRMAX", "0")), "tts_printed": set()}
_UNITS = {}   # id(momentum buffer) -> [unit1, unit2]


def _param_state_of(buf):
    """The optimizer state entry that owns this Muon momentum buffer (buffers are created on the first step and replaced
    by load_state_dict on a restart, so the map is rebuilt whenever a buffer is not in it)."""
    m = _STATE.get("buf2state")
    if m is None or id(buf) not in m:
        m = {}
        opt = _STATE.get("opt")
        if opt is not None:
            for st in opt.state.values():
                b = st.get("momentum_buffer") if isinstance(st, dict) else None
                if b is not None:
                    m[id(b)] = st
        _STATE["buf2state"] = m
    return m.get(id(buf))


def _rej_gate(D):
    """Experimental (not used in the paper): part of the fast-slow mismatch D living in collective modes with sigma > REJ_KAPPA x RMS singular
    value (per matrix; tensors with ndim >= 3 are stacks of matrices over the last two dims). Returns (gated, frac)."""
    X = D.float()
    squeeze = X.ndim == 2
    if squeeze:
        X = X.unsqueeze(0)
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    tall = X.shape[-2] >= X.shape[-1]
    S = X.mT @ X if tall else X @ X.mT
    k = S.shape[-1]
    fro2 = (X * X).sum(dim=(-2, -1), keepdim=True)
    theta2 = (REJ_KAPPA * REJ_KAPPA) * fro2 / k + 1e-30
    I = torch.eye(k, device=X.device, dtype=X.dtype).expand_as(S)
    L = torch.linalg.cholesky(S + theta2 * I)
    Z = I - (2.0 * theta2) * torch.cholesky_inverse(L)
    for _ in range(REJ_SHARP):
        Z = 1.5 * Z - 0.5 * (Z @ Z @ Z)
    P = 0.5 * (I + Z)
    G = X @ P if tall else P @ X
    frac = (G * G).sum() / ((X * X).sum() + 1e-30)
    return G.reshape(D.shape).to(D.dtype), frac


def _flush_log():
    if not _STATE["log"] or int(os.environ.get("RANK", "0")) != 0:
        _STATE["log"] = []
        return
    rows = torch.stack([torch.stack(r) for r in _STATE["log"]]).cpu().tolist()
    with open("gradnorm.txt", "a") as f:
        for s, n, c in rows:
            f.write(f"{int(s)} {n:.6g} {c:.6g}\n")
    _STATE["log"] = []


def _flush_at_exit():
    try:
        _flush_log()
        _flush_noise()
    except Exception:
        pass


def _install_sync_buckets():
    """SYNC_BUCKETS=1: every DDP rank draws the same system (structure-size bucket) each step, each rank taking its own
    frames of it (deepmd's per-system DistributedSampler) => one step = one bucket of ceil(atoms_per_batch / N) frames split
    over the ranks, as in the single-GPU recipe, and the ranks' work is balanced. deepmd draws systems with a
    WeightedRandomSampler on the global RNG, whose state differs between ranks; give it a generator seeded the same
    everywhere instead."""
    from deepmd.pt.utils import dataloader as DL
    orig = DL.get_weighted_sampler
    seed = int(os.environ.get("SYNC_SEED", "20261001"))

    def synced(*args, **kwargs):
        sampler = orig(*args, **kwargs)
        sampler.generator = torch.Generator().manual_seed(seed)
        return sampler
    DL.get_weighted_sampler = synced
    from deepmd.pt.train import training as T
    orig_next = T.Trainer._next_training_batch
    seen = [0]

    def next_batch(self, *args, **kwargs):            # check: print each rank's batch shape for the first steps
        out = orig_next(self, *args, **kwargs)
        if seen[0] < 6:
            seen[0] += 1
            c = out[0].get("coord")
            print(f"[bm_hybrid] sync_buckets check rank {os.environ.get('RANK', '0')} call {seen[0]}: coord {tuple(c.shape)}", flush=True)
        return out
    T.Trainer._next_training_batch = next_batch
    print(f"[bm_hybrid] sync_buckets: system sampler generator seeded {seed} on every rank", flush=True)


def _install_grad_acc(n):
    """GRAD_ACC=n: see the module docstring.  Text-patch of Trainer._run: the HybridMuon branch's single
    forward/backward becomes n micro-batches (each after the first from _next_training_batch); every backward is scaled
    by 1/n so the summed gradient is the micro-batch mean."""
    import inspect
    import textwrap
    from deepmd.pt.train import training as T
    src = textwrap.dedent(inspect.getsource(T.Trainer._run))
    lines = src.split("\n")
    k = [i for i, l in enumerate(lines) if l.strip() == "model_pred, loss, more_loss = self.wrapper("]
    assert len(k) == 1, f"grad_acc: expected one HybridMuon forward call in Trainer._run, found {len(k)}"
    k = k[0]
    ind = lines[k][: len(lines[k]) - len(lines[k].lstrip())]
    assert lines[k + 1].strip() == "**input_dict, cur_lr=pref_lr, label=label_dict, task_key=task_key"
    assert lines[k + 2].strip() == ")" and lines[k + 3].strip() == "loss.backward()", "grad_acc: unexpected deepmd source"
    loop = [
        f"{ind}for _acc_i in range(_GRAD_ACC - 1):",
        f"{ind}    _acc_out = self.wrapper(**input_dict, cur_lr=pref_lr, label=label_dict, task_key=task_key)",
        f"{ind}    (_acc_out[1] / _GRAD_ACC).backward()",
        f"{ind}    del _acc_out",
        f"{ind}    input_dict, label_dict, log_dict = self._next_training_batch(task_key)",
    ]
    lines = lines[:k] + loop + lines[k:k + 3] + [f"{ind}(loss / _GRAD_ACC).backward()"] + lines[k + 4:]
    T._GRAD_ACC = n
    ns = {}
    exec(compile("\n".join(lines), T.__file__, "exec"), T.__dict__, ns)
    T.Trainer._run = ns["_run"]
    print(f"[bm_hybrid] grad_acc={n}: {n} synchronised micro-batches per rank per optimizer step", flush=True)


def _gate_decide():
    """Spike gate: _STATE["gate"] = this step's pre-clip norm > BM_GATE_K x running median (see the module docstring)."""
    import bisect
    import math
    pre = _STATE.get("preclip")
    v = float(pre) if pre is not None else float("nan")
    win = _STATE.setdefault("gate_win", [])              # sorted window values
    fifo = _STATE.setdefault("gate_fifo", [])            # insertion order
    if win:
        med = win[len(win) // 2]
    else:
        med = BM_GATE_MED0
    ready = len(win) >= 100 or BM_GATE_MED0 > 0
    gate = (not math.isfinite(v)) or (ready and med > 0 and v > BM_GATE_K * med)
    _STATE["gate"] = gate
    _STATE["gate_n"] = _STATE.get("gate_n", 0) + int(gate)
    if math.isfinite(v):
        bisect.insort(win, v)
        fifo.append(v)
        if len(fifo) > BM_GATE_WIN:
            win.pop(bisect.bisect_left(win, fifo.pop(0)))
    t = _STATE["t"]
    if t % BM_GATE_REPORT == 0 and int(os.environ.get("RANK", "0")) == 0:
        print(f"[bm_hybrid] gate step {t}: slow unit frozen on {_STATE['gate_n']} of the last {BM_GATE_REPORT} steps "
              f"(running median {med:.3g}, threshold {BM_GATE_K * med:.3g})", flush=True)
        _STATE["gate_n"] = 0


def _pm_gate_stats(g, units, key):
    """Gate statistics before the units take g (EMA PM_OSC_BETA, first value as is)."""
    gf = g.float()
    gn = gf.norm().clamp_min(1e-20)
    mf = units[0].float()
    osc = (-(gf * mf).sum() / (gn * mf.norm().clamp_min(1e-20))).clamp_min(0.0)
    drift = units[1].float().norm() / gn
    s = _PMS.get(key)
    if s is None:
        _PMS[key] = [osc, drift]
    else:
        s[0].lerp_(osc, 1 - PM_OSC_BETA); s[1].lerp_(drift, 1 - PM_OSC_BETA)


def install():
    atexit.register(_flush_at_exit)
    if os.environ.get("SYNC_BUCKETS", "0") == "1":
        _install_sync_buckets()
    if int(os.environ.get("GRAD_ACC", "1")) > 1:
        _install_grad_acc(int(os.environ["GRAD_ACC"]))
    from deepmd.pt.optimizer import hybrid_muon as H
    if ISO:
        _install_iso(H)                              # must precede capturing step: it re-compiles step
    orig_step = H.HybridMuonOptimizer.step

    if GRAD_LOG or BM_GATE_K > 0:
        from deepmd.pt.train import training as T
        orig_clip = T.clip_grad_norm_

        def clip_logged(*args, **kwargs):
            total = orig_clip(*args, **kwargs)
            _STATE["preclip"] = total.detach().float()
            return total
        T.clip_grad_norm_ = clip_logged

    def step(self, *args, **kwargs):
        _STATE["t"] += 1
        _STATE["opt"] = self
        lr = float(self.param_groups[0]["lr"])
        _STATE["lr"] = lr
        _STATE["lrmax"] = max(_STATE["lrmax"], lr)
        if BM_GATE_K > 0:
            _gate_decide()
        if MAGMA_LOG > 0 and _STATE["t"] % MAGMA_LOG == 0 and int(os.environ.get("RANK", "0")) == 0:
            sc = [st["magma_score"].float().flatten() for st in self.state.values() if isinstance(st, dict) and torch.is_tensor(st.get("magma_score"))]
            bufs = [st["momentum_buffer"] for st in self.state.values() if isinstance(st, dict) and torch.is_tensor(st.get("momentum_buffer"))]
            if sc:
                v = torch.cat(sc)
                q = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9], device=v.device))
                rms = torch.stack([b.float().pow(2).mean().sqrt() for b in bufs]).mean() if bufs else torch.tensor(0.0)
                with open("magma.txt", "a") as f:
                    f.write(f"{_STATE['t']} {v.mean().item():.5f} {q[0].item():.5f} {q[1].item():.5f} {q[2].item():.5f} {v.numel()} {rms.item():.4e}\n")
        if GRAD_LOG:
            grads = [p.grad for g in self.param_groups for p in g["params"] if p.grad is not None]
            seen = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)).float())
            pre = _STATE["preclip"] if _STATE["preclip"] is not None else torch.full_like(seen, -1.0)
            _STATE["log"].append([torch.tensor(float(_STATE["t"]), device=seen.device), seen, pre.to(seen.device)])
            _STATE["preclip"] = None
            if len(_STATE["log"]) >= 500:
                _flush_log()
        if ISO:
            _iso_before_step(self)
            if _STATE["t"] % ISO_REPORT == 0:
                _iso_report(_STATE["t"])
        return orig_step(self, *args, **kwargs)

    H.HybridMuonOptimizer.step = step
    base_updates = H._compute_muon_nesterov_updates
    if ARM == "hmuon":
        if ISO:
            def hmuon_iso_updates(gradients, momentum_buffers, momentum, use_foreach=False):
                return _iso_pre(base_updates(gradients, momentum_buffers, momentum, use_foreach), momentum_buffers)
            H._compute_muon_nesterov_updates = hmuon_iso_updates
        print(f"[bm_hybrid] arm=hmuon (stock deepmd HybridMuon) grad_log={GRAD_LOG} iso={int(ISO)}", flush=True)
        return
    assert ARM == "bm" and _STATE["phases"], "ARM=bm needs BM_SCHED"
    orig_updates = H._compute_muon_nesterov_updates

    def bm_updates(gradients, momentum_buffers, momentum, use_foreach=False):
        t = _STATE["t"]
        ph = None
        for k, cand in enumerate(_STATE["phases"]):
            if t >= cand[0]:
                ph = cand
                if k not in _STATE["announced"]:
                    _STATE["announced"].add(k)
                    print(f"[bm_hybrid] phase {k} from optimizer step {t}: b1={cand[1]} b2={cand[2]} w1={cand[3]}", flush=True)
        if ph is None:
            if PM and PM_SHADOW:                    # 4. the units (and gate statistics) track g from the first step
                _, sb1, sb2, _ = _STATE["phases"][0]
                for g, buf in zip(gradients, momentum_buffers, strict=True):
                    units = _UNITS.get(id(buf))
                    if units is None:
                        units = _UNITS[id(buf)] = [torch.zeros_like(buf), torch.zeros_like(buf)]
                        st = _param_state_of(buf)
                        if st is not None:
                            st["bm_u0"], st["bm_u1"] = units
                    if PM_GATE:
                        _pm_gate_stats(g, units, id(buf))
                    units[0].lerp_(g, 1.0 - sb1)
                    if not _STATE.get("gate"):
                        units[1].lerp_(g, 1.0 - sb2)
            return orig_updates(gradients, momentum_buffers, momentum, use_foreach)
        _, b1, b2, w = ph
        mu_nes = float(BM_NESTEROV_MU) if BM_NESTEROV_MU else momentum
        if REJ_KAPPA > 0 and t % 1000 == 1 and _STATE.get("rej", [0, 0])[1] > 0 and t not in _STATE["tts_printed"]:
            _STATE["tts_printed"].add(t)
            print(f"[bm_hybrid] rej step {t}: passed_energy_frac={_STATE['rej'][0] / _STATE['rej'][1]:.4f}", flush=True)
            _STATE["rej"] = [0.0, 0]
        if BM_TTS_P > 0 and _STATE["lr"]:
            r = (_STATE["lrmax"] / _STATE["lr"]) ** BM_TTS_P
            tau1 = min(BM_TTS_CAP, b1 / (1.0 - b1) * r)
            tau2 = min(BM_TTS_CAP, b2 / (1.0 - b2) * r)
            b1, b2 = tau1 / (1.0 + tau1), tau2 / (1.0 + tau2)
            if t % 5000 == 0 and t not in _STATE["tts_printed"]:
                _STATE["tts_printed"].add(t)
                print(f"[bm_hybrid] tts step {t}: lr/lr_max={1.0 / r ** (1.0 / BM_TTS_P):.4g} tau1={tau1:.1f} tau2={tau2:.1f} "
                      f"mean_age={w * tau1 + (1 - w) * tau2:.1f}", flush=True)
        out = []
        for g, buf in zip(gradients, momentum_buffers, strict=True):
            units = _UNITS.get(id(buf))
            st = None
            seeded_now = False
            if units is None:
                st = _param_state_of(buf)
                if st is not None and "bm_u0" in st:     # restart: the units come back with the optimizer state
                    units = _UNITS[id(buf)] = [st["bm_u0"], st["bm_u1"]]
                    _STATE["restored"] = _STATE.get("restored", 0) + 1
            if units is None:                         # switch step: HybridMuon's own update, then seed both units
                buf.lerp_(g, 1.0 - momentum)
                units = _UNITS[id(buf)] = [buf.clone(), buf.clone()]
                seeded_now = True
                if st is not None:                    # saved with the optimizer state from now on
                    st["bm_u0"], st["bm_u1"] = units
                else:
                    _STATE["unsaved"] = _STATE.get("unsaved", 0) + 1
            else:
                if PM and PM_GATE:                      # 3. gate statistics before the units take this gradient
                    _pm_gate_stats(g, units, id(buf))
                units[0].lerp_(g, 1.0 - b1)
                if not _STATE.get("gate"):              # spike gate: the slow unit skips outlier steps
                    units[1].lerp_(g, 1.0 - b2)
                if BM_MAGMA_HM:                         # buf stays HybridMuon's EMA (Magma reads it); the update uses the kernel
                    buf.lerp_(g, 1.0 - momentum)
                    out.append(torch.lerp(g, units[0] * w + units[1] * (1.0 - w), mu_nes))
                    continue
                if PM and PM_GATE:
                    s_ = _PMS[id(buf)]
                    w_t = (PM_WMAX * (1 - (1.0 - b1) / (1.0 + b1) * s_[0] / s_[1].clamp_min(1e-20)).clamp(0.0, 1.0)).to(buf.dtype)
                    buf.copy_(units[0] * w_t + units[1] * (1 - w_t))
                else:
                    buf.copy_(units[0]).mul_(w).add_(units[1], alpha=1.0 - w)
                if REJ_KAPPA > 0 and g.ndim >= 2:
                    gated, frac = _rej_gate(units[0] - units[1])
                    buf.add_(gated, alpha=(1.0 - w) if REJ_MODE == "pass" else -w)
                    _STATE.setdefault("rej", [0.0, 0])
                    if t % 1000 == 0:
                        _STATE["rej"][0] += float(frac); _STATE["rej"][1] += 1
            mu_use = momentum if seeded_now else mu_nes
            if PM and PM_PHI > 0 and not seeded_now:   # 5. fresh-share cap on the Nesterov mix
                mu_use = torch.maximum(torch.tensor(mu_use, dtype=torch.float32, device=g.device),
                                       1.0 / (1.0 + PM_PHI / (1 - PM_PHI) * buf.float().norm() / g.float().norm().clamp_min(1e-20))).to(g.dtype)
            out.append(torch.lerp(g, buf, mu_use))   # the switch step's update equals HybridMuon's
        if not _STATE.get("units_reported"):
            _STATE["units_reported"] = True
            print(f"[bm_hybrid] units at step {t}: {len(_UNITS)} buffers, restored from checkpoint "
                  f"{_STATE.get('restored', 0)}, not attached to optimizer state {_STATE.get('unsaved', 0)}", flush=True)
        return out

    if ISO:
        def bm_iso_updates(gradients, momentum_buffers, momentum, use_foreach=False):
            return _iso_pre(bm_updates(gradients, momentum_buffers, momentum, use_foreach), momentum_buffers)
        H._compute_muon_nesterov_updates = bm_iso_updates
    else:
        H._compute_muon_nesterov_updates = bm_updates
    print(f"[bm_hybrid] arm=bm phases={_STATE['phases']} tts_p={BM_TTS_P} tts_cap={BM_TTS_CAP} t0={_STATE['t']} "
          f"rej_kappa={REJ_KAPPA} rej_mode={REJ_MODE} grad_log={GRAD_LOG} iso={int(ISO)} gate_k={BM_GATE_K} "
          f"gate_win={BM_GATE_WIN} gate_med0={BM_GATE_MED0} nesterov_mu={BM_NESTEROV_MU or 'hm'} magma_hm={int(BM_MAGMA_HM)} "
          f"pm={int(PM)} pm_gate={int(PM_GATE)} pm_wmax={PM_WMAX} pm_osc_beta={PM_OSC_BETA} pm_shadow={int(PM_SHADOW)} pm_phi={PM_PHI}", flush=True)
