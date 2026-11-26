"""CPU tests: bitwise agreement with the GPT training scripts, Muon-equivalence where it must hold, the five switches,
the spike guard and state-dict round trips."""
import copy

import torch

import reference_nanogpt as ref
from physmuon import (PhysMuon, fast_weight_for_age, gpt_speedrun, interatomic_potential,
                      interatomic_potential_gated, kernel_age, language_model)

SHAPES = [(48, 16), (24, 48), (24, 24)]          # tall, wide, square


def _mats(seed=0):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s) / s[1] ** 0.5) for s in SHAPES]


def _targets(seed=1, steps=40, k=4):
    g = torch.Generator().manual_seed(seed)
    return [[(torch.randn(8, 16, generator=g), torch.randn(8, 24, generator=g)) for _ in range(k)] for _ in range(steps)]


def _loss(mats, x, y):
    h = torch.tanh(x @ mats[0].T)                # (8, 48)
    h = torch.tanh(h @ mats[1].T)                # (8, 24)
    return ((h @ mats[2].T - y) ** 2).sum()


def _run_package(opt, mats, data, observe):
    for batch in data:
        for x, y in batch:
            _loss(mats, x, y).backward()
            if observe:
                opt.observe_microbatch()
        opt.step()
        for p in mats:
            p.grad = None


def _run_reference(opt, mats, data, iso):
    for step, batch in enumerate(data):
        if iso:
            for p in mats:
                p.iso_rs2 = torch.zeros(p.size(0)); p.iso_cs2 = torch.zeros(p.size(1)); p.iso_prev = None
        for x, y in batch:
            _loss(mats, x, y).backward()
            if iso:
                with torch.no_grad():
                    for p in mats:
                        gk = p.grad.float() if p.iso_prev is None else p.grad.float() - p.iso_prev
                        gk2 = gk.square()
                        p.iso_rs2 += gk2.sum(1); p.iso_cs2 += gk2.sum(0)
                        p.iso_prev = p.grad.float().clone()
        if iso:
            for p in mats:
                p.iso_count = len(batch); p.iso_prev = None
        opt.cur_step = step
        opt.step()
        for p in mats:
            p.grad = None


def test_matches_reference_all_five():
    """All five changes: bitwise equal to the optimizer of the 3,085-step GPT script."""
    data = _targets()
    m1, m2 = _mats(), _mats()
    o1 = ref.RefAllFive(m1, lr=0.02, weight_decay=0.05, bm_start=10, fresh_cap=0.25)
    o2 = PhysMuon([dict(params=m2)], lr=0.02, weight_decay=0.05, ns="cubic", **dict(gpt_speedrun(), bm_start=10))
    _run_reference(o1, m1, data, iso=True)
    _run_package(o2, m2, data, observe=True)
    for p1, p2 in zip(m1, m2):
        assert torch.equal(p1, p2), f"max |diff| {float((p1 - p2).abs().max()):.3e}"
        for k in ("m_fast", "m_slow", "osc", "drift", "row_heat", "col_heat"):
            assert torch.equal(o1.state[p1][k], o2.state[p2][k]), k


def test_matches_reference_without_iso():
    """Changes 1, 3, 4, 5 (without change 2): bitwise equal to the optimizer of the 3,120-step GPT script."""
    data = _targets(seed=2)
    m1, m2 = _mats(), _mats()
    o1 = ref.RefNoIso(m1, lr=0.02, weight_decay=0.05, bm_start=10, fresh_cap=0.25)
    o2 = PhysMuon([dict(params=m2)], lr=0.02, weight_decay=0.05, ns="cubic", **dict(gpt_speedrun(), bm_start=10, iso=False))
    _run_reference(o1, m1, data, iso=False)
    _run_package(o2, m2, data, observe=False)
    for p1, p2 in zip(m1, m2):
        assert torch.equal(p1, p2), f"max |diff| {float((p1 - p2).abs().max()):.3e}"


def test_matches_reference_iso_only():
    """Change 2 alone (no memory change): bitwise equal to the reference step before its switch."""
    data = _targets(seed=3, steps=12)
    m1, m2 = _mats(), _mats()
    o1 = ref.RefAllFive(m1, lr=0.02, weight_decay=0.05, bm_start=10 ** 9)   # never switches: change 2 alone throughout
    o2 = PhysMuon([dict(params=m2)], lr=0.02, weight_decay=0.05, ns="cubic", iso=True)
    _run_reference(o1, m1, data, iso=True)
    _run_package(o2, m2, data, observe=True)
    for p1, p2 in zip(m1, m2):
        assert torch.equal(p1, p2), f"max |diff| {float((p1 - p2).abs().max()):.3e}"


def test_plain_muon_matches_reference():
    data = _targets(seed=4, steps=10)
    m1, m2 = _mats(), _mats()
    o1 = ref.RefNoIso(m1, lr=0.02, weight_decay=0.05, bm_start=10 ** 9)     # never switches: the baseline's Muon
    o2 = PhysMuon([dict(params=m2)], lr=0.02, weight_decay=0.05, ns="cubic")
    _run_reference(o1, m1, data, iso=False)
    _run_package(o2, m2, data, observe=False)
    for p1, p2 in zip(m1, m2):
        assert torch.equal(p1, p2)


def test_shadow_onset_is_muon_before_the_switch():
    data = _targets(seed=5, steps=12)
    m1, m2 = _mats(), _mats()
    o1 = PhysMuon([dict(params=m1)], lr=0.02)
    o2 = PhysMuon([dict(params=m2)], lr=0.02, bm_start=8, bm_beta_slow=0.985, osc_gate=True, bm_shadow=True, fresh_cap=0.25)
    _run_package(o1, m1, data[:8], observe=False)
    _run_package(o2, m2, data[:8], observe=False)
    for p1, p2 in zip(m1, m2):
        assert torch.equal(p1, p2), "before bm_start the update must be Muon's"
        assert o2.state[p2]["m_slow"].abs().sum() > 0, "with shadow onset the units must already follow the gradient"
    _run_package(o1, m1, data[8:], observe=False)
    _run_package(o2, m2, data[8:], observe=False)
    assert any(not torch.equal(p1, p2) for p1, p2 in zip(m1, m2)), "after bm_start the readout must change"


def test_snapshot_switch_equals_muon():
    """Without shadow onset the units are copies of the momentum at the switch, so that step is still Muon's."""
    data = _targets(seed=6, steps=21)
    m1, m2 = _mats(), _mats()
    o1 = PhysMuon([dict(params=m1)], lr=0.01)
    o2 = PhysMuon([dict(params=m2)], lr=0.01, bm_start=20, bm_beta_fast=0.9, bm_beta_slow=0.995, bm_w_fast=0.2579)
    _run_package(o1, m1, data, observe=False)
    _run_package(o2, m2, data, observe=False)
    for p1, p2 in zip(m1, m2):
        assert torch.equal(p1, p2)


def test_gate_weight_and_fresh_share_bounds():
    data = _targets(seed=7, steps=30)
    m = _mats()
    o = PhysMuon([dict(params=m)], lr=0.02, **dict(gpt_speedrun(), bm_start=5, iso=False))
    _run_package(o, m, data, observe=False)
    for p in m:
        st = o.state[p]
        rho_f = (1 - 0.85) / (1 + 0.85)
        w = float(0.6 * (1 - rho_f * st["osc"] / st["drift"].clamp_min(1e-20)).clamp(0.0, 1.0))
        assert 0.0 <= w <= 0.6
        assert float(st["osc"]) >= 0.0 and float(st["drift"]) > 0.0


def test_fresh_share_never_exceeds_cap():
    """mu_t = max(mu, 1/(1 + phi/(1-phi) r)) keeps (1-mu_t)|g| / ((1-mu_t)|g| + mu_t |m|) <= phi."""
    phi, mu = 0.25, 0.95
    for r in (0.01, 0.1, 0.5, 1.0, 3.0, 30.0):            # r = |m_eff| / |g|
        mu_t = max(mu, 1.0 / (1.0 + phi / (1 - phi) * r))
        share = (1 - mu_t) / ((1 - mu_t) + mu_t * r)
        assert share <= phi + 1e-12


def test_iso_without_noise_is_the_plain_polar_direction():
    """Identical micro-batches carry no noise: the metric is the identity, so change 2 steps along Muon's polar
    factor and only rescales it to the Frobenius norm sqrt(min(rows, cols)). A wide matrix keeps the shape factor at
    exactly 1 (Muon applies it in bfloat16, change 2 in float32)."""
    torch.manual_seed(0)
    w0 = torch.randn(16, 24)
    g = torch.randint(-8, 8, (16, 24)).float()            # small integers: the accumulated sums are exact
    p1, p2 = torch.nn.Parameter(w0.clone()), torch.nn.Parameter(w0.clone())
    o1 = PhysMuon([dict(params=[p1])], lr=0.02, ns="cubic", momentum=0.0, nesterov=False)
    o2 = PhysMuon([dict(params=[p2])], lr=0.02, ns="cubic", momentum=0.0, nesterov=False, iso=True)
    p1.grad = 4 * g
    o1.step()
    for k in range(4):                                    # four identical micro-batches
        p2.grad = g.clone() if k == 0 else p2.grad + g
        o2.observe_microbatch()
    o2.step()
    st = o2.state[p2]
    assert torch.equal(st["row_heat"], torch.full_like(st["row_heat"], 1e-30))
    assert torch.equal(st["col_heat"], torch.full_like(st["col_heat"], 1e-30))
    d1, d2 = (w0 - p1.detach()).flatten(), (w0 - p2.detach()).flatten()
    cos = float(d1 @ d2 / (d1.norm() * d2.norm()))
    assert cos > 1 - 1e-6, cos
    assert abs(float(d2.norm()) - 0.02 * 16 ** 0.5) < 1e-5


def test_spike_guard_freezes_slow_unit():
    m = _mats()
    o = PhysMuon([dict(params=m)], lr=0.01, bm_start=5, bm_beta_fast=0.9, bm_beta_slow=0.995, bm_w_fast=0.2579,
               spike_guard=10.0, spike_window=50)
    _run_package(o, m, _targets(seed=8, steps=150, k=1), observe=False)     # fill the window with ordinary norms
    slow_before = o.state[m[0]]["m_slow"].clone()
    frozen_before = o.spike_frozen_steps
    for x, y in _targets(seed=9, steps=1, k=1)[0]:
        _loss(m, x, y).backward()
    for p in m:
        p.grad.mul_(1e6)                                  # one step with a 1e6 x gradient
    o.step()
    assert o.spike_frozen_steps == frozen_before + 1
    assert torch.equal(o.state[m[0]]["m_slow"], slow_before), "the slow unit must skip the spike step"


def test_state_dict_round_trip_all_five():
    data = _targets(seed=10, steps=30)
    m1 = _mats()
    kw = dict(gpt_speedrun(), bm_start=8)
    o1 = PhysMuon([dict(params=m1)], lr=0.02, weight_decay=0.05, ns="cubic", **kw)
    _run_package(o1, m1, data[:20], observe=True)
    m2 = [torch.nn.Parameter(p.detach().clone()) for p in m1]
    o2 = PhysMuon([dict(params=m2)], lr=0.02, weight_decay=0.05, ns="cubic", **kw)
    o2.load_state_dict(copy.deepcopy(o1.state_dict()))   # as torch.save/torch.load would; a bare load aliases the live tensors
    _run_package(o1, m1, data[20:], observe=True)
    _run_package(o2, m2, data[20:], observe=True)
    for p1, p2 in zip(m1, m2):
        assert torch.equal(p1, p2), "units, gate statistics and fluctuation estimates must survive state_dict / load_state_dict"


def test_switches_need_the_memory():
    m = _mats()
    for bad in (dict(osc_gate=True), dict(bm_shadow=True), dict(fresh_cap=0.25)):
        try:
            PhysMuon([dict(params=m)], **bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} without bm_start must be rejected")


def test_schedule_helpers():
    assert abs(kernel_age(0.90, 0.995, 0.2579) - 150.0) < 1.0
    w = fast_weight_for_age(0.85, 0.98, 30.0)
    assert abs(kernel_age(0.85, 0.98, w) - 30.0) < 1e-9


def test_presets():
    assert language_model(3250) == gpt_speedrun(3250) and language_model(3250)["bm_start"] == 750
    pot = interatomic_potential(20000)
    assert pot == interatomic_potential_gated(20000)
    assert pot["bm_start"] == 6154 and pot["bm_beta_fast"] == 0.90 and pot["bm_beta_slow"] == 0.995
    assert pot["osc_gate"] is True and pot["osc_wmax"] == 0.3 and pot["bm_shadow"] is True
    assert interatomic_potential(69740)["bm_start"] == 21459
    for preset in (language_model(40), interatomic_potential(40),
                   interatomic_potential(40, beta_fast=0.85, beta_slow=0.98)):
        mats = _mats()
        before = [m.detach().clone() for m in mats]
        opt = PhysMuon([dict(params=mats)], lr=0.02, **preset)
        _run_package(opt, mats, _targets(), observe=True)
        assert all(torch.isfinite(m).all() for m in mats)
        assert all(not torch.equal(m, b) for m, b in zip(mats, before))
