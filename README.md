# PhysMuon

PhysMuon is [Muon](https://github.com/KellerJordan/Muon) with five changes to its update, each derived from a physical
model of the optimizer. It is used like Muon, on the hidden weight matrices of a network, and reaches a lower loss in the
same number of steps:

- **GPT-2 124M** (modded-nanogpt speed benchmark): the target validation loss is reached at step 3,085; the tuned Muon
  baseline does not reach it within 3,250 steps.
- **Interatomic potentials** (DPA-3 and DPA4 on MPtrj, SPICE2 and ANI-1x): 7–21% lower force error than Muon at the same
  number of steps, and fewer unstable molecular-dynamics trajectories.
- **Matbench Discovery** (DPA4-Pro trained on MPtrj only): combined performance score 0.8757, against 0.8650 for the
  leaderboard entry with the same architecture.

Paper: Y. Hu, H. Xiang, X. Gong, H. Yu, *A Physical Response-and-Memory Model for Muon Optimization*,
[arXiv:2608.22994](https://arxiv.org/abs/2608.22994). 中文说明：[README_zh.md](README_zh.md)。

## Installation

```bash
pip install git+https://github.com/Hyyu-Lab/PhysMuon      # or, in a copy of the repository: pip install .
```

Requirements: Python ≥ 3.10 and PyTorch ≥ 2.1, on any operating system and hardware supported by PyTorch; there are no
other dependencies. Tested with Python 3.11 and PyTorch 2.14 on Linux, on CPUs and CUDA GPUs. With PyTorch already
installed, installation takes less than a minute. The optimizer is a single file, [`physmuon/optim.py`](physmuon/optim.py),
which can also be copied into a project.

## Quick start

```python
from physmuon import PhysMuon, language_model

# as with Muon: hidden weight matrices -> PhysMuon; embeddings, output head, gains and biases -> AdamW
hidden = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n and "head" not in n]
others = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n and "head" not in n)]

opt = PhysMuon([dict(params=hidden),
                dict(params=others, use_muon=False, lr=3e-3)],     # AdamW inside the same optimizer
               lr=0.02, weight_decay=0.01,                         # the values of your Muon setup
               **language_model(total_steps))                      # recommended settings, see below

for step in range(total_steps):
    for x, y in micro_batches(step):          # at least two micro-batches per step (gradient accumulation)
        loss_fn(model(x), y).backward()
        opt.observe_microbatch()              # change ② measures the gradient fluctuations here
    opt.step()
    opt.zero_grad(set_to_none=True)
```

Compared with a Muon training loop, there are two differences: the settings come from a preset that needs the total
number of steps, and `observe_microbatch()` is called after the backward pass of every micro-batch. Learning rate,
weight decay, schedule and parameter groups stay as in your Muon setup.

**Demo.** `python examples/quickstart.py` runs this loop on a small network and synthetic data. It needs no GPU and
no data set, takes 15–70 seconds on four CPU cores and prints:

```text
step    0  loss 0.0966
step   50  loss 0.0149
step  100  loss 0.0137
step  150  loss 0.0142
step  200  loss 0.0133
step  250  loss 0.0124
step  299  loss 0.0120
physmuon: held-out loss 0.0020
```

The last digits can differ between PyTorch versions and CPUs. The example checks the installation and shows the
complete loop; the experiments of the paper are reproduced as described in [`experiments/`](experiments).

## Recommended hyperparameters

Start from a tuned Muon setup and keep its learning rate, weight decay, schedule and momentum. Then use one of the
presets:

| | language models | interatomic potentials |
|---|---|---|
| preset | `language_model(T)` | `interatomic_potential(T)` |
| changes switched on | ①②③④⑤ | ①②③④ |
| readout switches at (`bm_start`) | 23% of training | 30.77% of training |
| post-switch readout and mode rates | parallel fast + slow (gated), 0.85 / 0.985 | parallel fast + slow (gated), 0.90 / 0.995 |
| ceiling of the gated readout (`osc_wmax`) | 0.6 | 0.3 |
| fresh-share cap (`fresh_cap`) | 0.25 | off |
| constants of ② | λ = 0.5, clamp 2, EMA 0.95 | the same |

`T` is the total number of training steps. Both presets use the shorter baseline momentum before `bm_start` and switch to
parallel fast + slow blending via gated readout ③ afterward; five of the six interatomic-potential tasks in the paper
share the exact defaults of `interatomic_potential(T)`.
[docs/hyperparameters.md](docs/hyperparameters.md) gives the values for each task and a step-by-step guide for new ones.

## The five changes

The physical model treats each weight matrix as a responsive medium with memory: semi-orthogonalization (Muon's polar
factor) is the response that lowers the loss fastest when the change of a layer's output per step is capped, and momentum
is an internal stress that relaxes on a single timescale. Replacing these two simplest forms with forms closer to a real
medium gives the five changes.

| | change | what it does | switch |
|---|---|---|---|
| ① | two relaxation modes | replaces the single momentum average by a fast and a slow mode, read as a weighted mix | `bm_start`, `bm_beta_fast`, `bm_beta_slow`, `bm_w_fast` |
| ② | fluctuation-calibrated output budget | takes the polar factor in a diagonal metric set by the gradient fluctuations of each row and column | `iso=True` |
| ③ | gated readout | sets the weight of the fast mode per matrix and per step from the measured rebound and drift | `osc_gate=True`, `osc_wmax` |
| ④ | tracking from step 0 | lets both modes follow the gradient from the first step; only the readout switches at `bm_start` | `bm_shadow=True` |
| ⑤ | fresh-share cap | keeps the share of the current gradient in the update input below φ when the memory is read deep | `fresh_cap=φ` |

With all five off, `PhysMuon` is Muon. The update rules are written out at the top of
[`physmuon/optim.py`](physmuon/optim.py); a Chinese summary of the physics is in [docs/physics_zh.md](docs/physics_zh.md).

## Using it in a training loop

**Parameter groups.** Hidden weight matrices (attention and MLP weights) go into groups with `use_muon=True` (the
default when every parameter in the group has at least two dimensions). Embeddings, the output head, gains and biases
go into `use_muon=False` groups (AdamW inside this optimizer) or into an optimizer of your own, as in the usual Muon
setups. Parameters with more than two dimensions are treated as stacks of matrices over their last two dimensions.

**Micro-batches for ②.** ② estimates the gradient fluctuations of every output and input channel from the spread
between the micro-batch gradients of one step. A step therefore needs at least two micro-batches, and
`observe_microbatch()` must be called after each backward pass, while `p.grad` is still accumulating. If it is never
called, the optimizer warns once and uses Muon's polar factor. To train without micro-batches, switch ② off
(`iso=False`); the other changes do not need them.

**Distributed training.** Every rank computes the full update, so all ranks stay identical; the fluctuation estimates
are averaged over ranks at `step()`. Either run each micro-batch backward inside `model.no_sync()` and all-reduce the
gradients after the last one (as `examples/nanogpt/train_gpt.py` does), or let DDP average the gradients at every
backward pass. Keep at least two micro-batches per rank.

**Learning rate and weight decay.** Set `group["lr"]` from your schedule before each `step()`. Weight decay is decoupled
and multiplied by the learning rate.

**Switch step.** The two modes are read from `bm_start` on; before that, the update is Muon's. Reading a long memory from
the first step was clearly worse on GPT-2 and unstable on the potentials, so the switch comes after the early, steep
part of training: 23% of the run on GPT-2, 30.77% on the potentials.

**Checkpoints.** The two modes, the gate statistics and the fluctuation estimates are optimizer state:
`opt.state_dict()` and `opt.load_state_dict()` resume exactly.

**Cost.** On GPT-2 124M a PhysMuon step takes about 2% longer than a Muon step, and the optimizer state grows from
0.75 to 1.38 GiB (the two modes are kept for every matrix).

**Heavy-tailed gradients.** On data with rare, extremely large gradients, set `spike_guard=10`: a step whose gradient
norm exceeds ten times the median of the last 1,000 steps leaves the slow mode unchanged. If you clip gradients, pass the
norm before clipping, `opt.step(grad_norm=...)`. This optional safeguard follows earlier spike-detection work
(SPAM, arXiv:2501.06842; ZClip, arXiv:2504.02507) and is used on DPA4-Pro.

## Options

| option | default | meaning |
|---|---|---|
| `lr`, `weight_decay` | 0.02, 0 | learning rate; decoupled weight decay |
| `momentum` | 0.95 | Nesterov coefficient, and the rate of the single momentum average before the switch |
| `nesterov`, `ns`, `ns_steps`, `scale` | True, "quintic", None, "shape" | Muon's own settings; `ns="cubic"` is the 12-step iteration of the GPT-2 benchmark baseline; `scale="shape"` multiplies the update by √max(1, rows/cols), a float `c` by c·√max(rows, cols) |
| `bm_start` | None | ①: step from which the two modes are read (None: Muon's single average throughout) |
| `bm_beta_fast`, `bm_beta_slow` | 0.85, 0.98 | ①: rates of the fast and the slow mode |
| `bm_w_fast` | 0.4385 | ①: fixed weight of the fast mode after the switch, used when ③ is off |
| `iso`, `iso_lam`, `iso_clamp`, `iso_beta`, `iso_sync` | False, 0.5, 2.0, 0.95, True | ②: on/off, exponent λ, metric bound [1/c, c], EMA rate of the fluctuation estimates, average them over ranks |
| `osc_gate`, `osc_wmax`, `osc_beta` | False, 0.6, 0.9 | ③: on/off, largest weight of the fast mode, EMA rate of the two gate statistics |
| `bm_shadow` | False | ④: both modes follow the gradient from step 0 (otherwise they start as copies of the momentum at `bm_start`) |
| `fresh_cap` | 0 | ⑤: largest share φ of the current gradient in the update input after the switch (0: off) |
| `betas`, `eps` | (0.9, 0.95), 1e-8 | AdamW groups |
| `spike_guard`, `spike_window` | 0, 1000 | optional safeguard for heavy-tailed gradients |

③, ④ and ⑤ act on the two modes and require `bm_start`. The option prefixes identify the change they configure: `bm_`
for ① and ④ (the two relaxation modes), `iso_` for ② (the fluctuation-calibrated output budget), and `osc_` for ③
(the gated readout).

## Interatomic potentials with deepmd-kit

[`integrations/deepmd/`](integrations/deepmd) adds PhysMuon to the training of DPA-3 and DPA4 models in deepmd-kit, with
the settings for MPtrj, SPICE2 and ANI-1x.

## Results

| task | baseline | PhysMuon |
|---|---|---|
| GPT-2 124M, modded-nanogpt speed benchmark: first step at which the 8-seed mean validation loss meets the benchmark criterion | tuned Muon: not reached within 3,250 steps | step 3,085 |
| the same benchmark, leaderboard record configuration with only the momentum replaced by ① | record: 2,690 | 2,635 |
| GPT 302M, validation loss after 3,250 steps | 3.2395 | 3.2216 |
| five interatomic-potential tasks: validation force error | Muon; HybridMuon for DPA4 | 7.4%–20.9% lower, on every seed |
| molecular dynamics with the SPICE2 models at 2,100 K: unstable trajectories | 66.7% | 48.8% |
| Matbench Discovery, DPA4-Pro trained on MPtrj only: combined performance score | 0.8650 (leaderboard entry) | 0.8757 |

All numbers, the training logs and the commands that reproduce them are in [`experiments/`](experiments).

## Repository layout

| path | content |
|---|---|
| [`physmuon/`](physmuon) | the optimizer (`optim.py`), the presets (`recipes.py`) and helpers for the memory length (`schedule.py`) |
| [`examples/`](examples) | `quickstart.py`, and `nanogpt/train_gpt.py`, the training script of the GPT-2 benchmark |
| [`integrations/deepmd/`](integrations/deepmd) | PhysMuon in deepmd-kit for DPA-3 and DPA4, with the settings of every task |
| [`docs/`](docs) | the hyperparameter guide; a summary of the physics (in Chinese) |
| [`experiments/`](experiments) | the experiments of the paper, one folder each for language models, interatomic potentials, molecular dynamics, Matbench Discovery and the memory probe: results, data, scripts that recompute the numbers of the paper, and the commands that reproduce them |
| [`tests/`](tests) | unit tests |

## Tests

```bash
python tests/run_tests.py    # or: pytest tests
```

The tests run on a CPU. They check that the optimizer reproduces, bit for bit, the optimizer code of the training
scripts behind the GPT-2 results, that it equals Muon wherever it should, and that training resumes exactly from a
state dict.

## Citation

```bibtex
@misc{hu2026physmuon,
  title         = {A Physical Response-and-Memory Model for Muon Optimization},
  author        = {Hu, Yinze and Xiang, Hongjun and Gong, Xingao and Yu, Hongyu},
  year          = {2026},
  eprint        = {2608.22994},
  archivePrefix = {arXiv}
}
```

## License and acknowledgements

MIT. `examples/nanogpt/train_gpt.py` and `tests/reference_nanogpt.py` derive from
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT, license text in [`examples/nanogpt/LICENSE`](examples/nanogpt/LICENSE)).
PhysMuon builds on Muon (K. Jordan et al., 2024) and, for the interatomic potentials, on
[deepmd-kit](https://github.com/deepmodeling/deepmd-kit).
