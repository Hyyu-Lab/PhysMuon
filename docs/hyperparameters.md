# Hyperparameters

中文版：[hyperparameters_zh.md](hyperparameters_zh.md)。

## Recommended settings

PhysMuon is used on top of a tuned Muon setup. Keep Muon's learning rate, weight decay, schedule and momentum, and add
one of the presets of `physmuon`:

```python
opt = PhysMuon(groups, lr=..., weight_decay=..., **language_model(total_steps))
```

| | `language_model(T)` | `interatomic_potential(T)` |
|---|---|---|
| use it for | language models | interatomic potentials |
| changes switched on | ①②③④⑤ | ①②③④ |
| readout switches at | 23% of training | 30.77% of training |
| ① post-switch readout and mode rates | parallel fast + slow (gated), 0.85 / 0.985 | parallel fast + slow (gated), 0.90 / 0.995 |
| ② exponent λ, clamp, EMA rate | 0.5, [1/2, 2], 0.95 | the same |
| ③ ceiling `osc_wmax` | 0.6 | 0.3 |
| ④ | on | on |
| ⑤ `fresh_cap` φ | 0.25 | off |

Both presets use the shorter baseline momentum before `bm_start` and switch to parallel fast + slow blending via gated
readout ③ afterward. Five of the six interatomic-potential tasks in the paper share the exact defaults of
`interatomic_potential(T)` (`beta_fast=0.90, beta_slow=0.995, w_max=0.3`); the 8,000-step regular-batch SPICE2 task
uses `interatomic_potential(T, beta_fast=0.85, beta_slow=0.98)`. With deepmd-kit, the same settings are given as
environment variables; see [`integrations/deepmd/README.md`](../integrations/deepmd/README.md).

## Settings of every task in the paper

| task | baseline: learning rate (or multiplier), momentum | switch step | momentum before the switch | β_fast | β_slow | readout cap w_max | ② estimated from | ⑤ |
|---|---|---|---|---|---|---|---|---|
| GPT-2 124M (3,250 steps) | 0.025, 0.95 | 750 | 0.95 | 0.85 | 0.985 | ③, 0.6 | 8 micro-batches | φ = 0.25 |
| GPT-2 302M (3,250 steps) | 0.0175, 0.95 | 750 | 0.95 | 0.85 | 0.985 | ③, 0.6 | 16 micro-batches | φ = 0.25 |
| MPtrj, DPA-3 (20,000 steps) | ×8, 0.98 | 6,154 | 0.95 | 0.90 | 0.995 | ③, 0.3 | 4 micro-batches | off |
| SPICE2, DPA-3 (8,000 steps) | ×16, 0.95 | 2,461 | 0.95 | 0.85 | 0.98 | ③, 0.3 | 4 micro-batches | off |
| SPICE2 with a large batch, DPA-3 (20,000 steps) | ×8, 0.98 | 6,154 | 0.98 | 0.90 | 0.995 | ③, 0.3 | 4 micro-batches | off |
| ANI-1x, DPA-3 (20,000 steps) | ×8, 0.98 | 6,154 | 0.98 | 0.90 | 0.995 | ③, 0.3 | 4 micro-batches | off |
| MPtrj, DPA4-Mini (10 epochs) | official recipe, 0.95 | 21,459 | 0.95 | 0.90 | 0.995 | ③, 0.3 | the gradients of 2 GPUs | off |
| MPtrj, DPA4-Pro (2,000,000 steps) | official recipe, 0.95 | 615,400 | 0.95 | 0.90 | 0.995 | ③, 0.3 | the gradients of 4–16 GPUs | off |

- The switch step is where the update starts to read the two modes; before it, the update uses a single-mode
  momentum ("momentum before the switch"). ④ is on in every task.
- DPA-3 learning rates are multipliers of the exponential schedule 1e-3 → 3e-5 (both ends are multiplied); the
  multiplier is the same for Muon and PhysMuon. DPA4 uses the published recipes unchanged.
- DPA4-Pro also uses the spike guard with a threshold of 10 times the median of the last 1,000 steps.

## Choosing the settings for a new task

Across the tasks in the paper, the hyperparameters fall into three groups:

- **Fixed defaults across tasks:** ②'s constants (λ = 0.5, clamp [1/2, 2], EMA rate 0.95), ③'s gate EMA rate
  `osc_beta = 0.9` (nearly flat over 0.80–0.95; see the sensitivity table below), and ④ (`bm_shadow=True`, no constant).
- **Chosen by task family:** ③'s ceiling `osc_wmax` (0.6 on language models, 0.3 on interatomic potentials), ⑤
  `fresh_cap` (0.25 on language models, off on interatomic potentials; compare on/off on new domains), the switch step
  (placed after the steep early transient: 23% on GPT-2, 30.77% on potentials), and the spike guard when gradients are
  heavy-tailed.
- **Tuned per task and batch size:** ①'s memory length (`bm_beta_fast`, `bm_beta_slow`). The mean memory length of a
  mode with rate β is β/(1 − β) steps; for a two-mode mix it is w·β_fast/(1 − β_fast) + (1 − w)·β_slow/(1 − β_slow)
  (`physmuon.kernel_age`, `physmuon.beta_for_age`).

Step by step:

1. **Tune the baseline first.** Scan Muon's learning rate and momentum, and compare against the tuned Muon.
2. **Set ①'s mode rates.** Start from `0.85 / 0.985` on language models or `0.90 / 0.995` on interatomic potentials
   (shorter runs with smaller batches, such as 8,000-step regular-batch SPICE2, can use `0.85 / 0.98`). Switch the
   readout on at about 30% of training (30.77% on potentials, 23% on GPT-2); reading a long memory from step 0 was
   clearly worse on GPT-2 and unstable on potentials.
3. **Set ③'s ceiling `osc_wmax`.** Start from 0.6 on language models or 0.3 on interatomic potentials (keep
   `osc_beta = 0.9`, which is flat over 0.80–0.95).
4. **Keep ②'s constants** (λ = 0.5, clamp [1/2, 2], EMA rate 0.95). On MPtrj and SPICE2, changing them made the results
   worse; on GPT-2, λ and the EMA rate were best at these values and widening the clamp to [1/3, 3] made no difference.
   ② needs at least two independent gradient estimates per step: micro-batches, or the gradients of different GPUs.
5. **Switch ④ on.** It has no constant.
6. **Use ⑤ on language models** (φ = 0.25); leave it off on interatomic potentials (on a new domain, test φ = 0.25
   against 0).
7. **Heavy-tailed gradients.** On large crystal data sets with rare extreme gradient norms, switch the spike guard on:
   `spike_guard=10` in `PhysMuon`; `BM_GATE_K=10` in the DPA4 hook for deepmd-kit, where `ISO_SPIKE_SKIP=1` is also
   recommended.

Compare configurations across multiple seeds using matched seeds and data order.

## Sensitivity of the constants on GPT-2

Four paired seeds per value; the recommended values are in bold. The full table is in
[`experiments/language_model/ablations.md`](../experiments/language_model/ablations.md).

| constant | values tried | effect on the validation loss |
|---|---|---|
| switch step `bm_start` | 0, 500, **750**, 1,000 | 0 costs +0.0065; 500 is equal; 1,000 loses about 0.0005 |
| slow mode `bm_beta_slow` | 0.98, **0.985**, 0.99 | −0.0033, **−0.0039**, −0.0012 relative to ① with a fixed readout weight |
| fast mode `bm_beta_fast` | 0.80, **0.85**, 0.90 | +0.00104, **0**, +0.00060 |
| ceiling `osc_wmax` | 0.5, **0.6**, 0.7 | 0.5 and 0.7 lose about 0.0004 |
| fresh share `fresh_cap` | 0.15, **0.25**, 0.35, 0.45 | gains of 0.00037, **0.00052**, 0.00045, 0.00016 over leaving ⑤ off |
| ② exponent `iso_lam` | 0.25, **0.5**, 1.0 | 1.0 loses about 0.002; 0.25 keeps about 80% of the gain |
| ② clamp `iso_clamp` | **2**, 3 | no difference |
| EMA rate of the gate `osc_beta` | 0.80, **0.9**, 0.95 | +0.00013, **0**, +0.00011 |
| EMA rate of ② `iso_beta` | 0.90, **0.95** | +0.00031 at 0.90 |
