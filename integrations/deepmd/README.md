# PhysMuon in deepmd-kit

This directory integrates PhysMuon into [deepmd-kit](https://github.com/deepmodeling/deepmd-kit) for training DPA-3 and
DPA4 interatomic potentials. The model, data pipeline, loss, learning-rate schedule and evaluation are deepmd's; a hook
file replaces the optimizer.

| folder | model | what the hook changes |
|---|---|---|
| [`dpa3/`](dpa3) | DPA-3 | deepmd's Adam is replaced by Muon or PhysMuon on the matrix weights (Adam stays on the rest) |
| [`dpa4/`](dpa4) | DPA4 | the momentum of deepmd's HybridMuon is replaced by PhysMuon |

## Requirements

Requires deepmd-kit 3.2.0 with the PyTorch backend; the hooks patch trainer internals and are tied to this version. The
PyPI wheel of deepmd-kit 3.2.0 compiles its C++ PyTorch operators against PyTorch 2.11.0 and depends on `mpich` via the
`[torch]` extra. [`requirements.txt`](requirements.txt) pins a clean PyPI installation (Linux, CUDA 13):

```bash
pip install -r requirements.txt
```

If you need a different CUDA wheel index for PyTorch, install `torch==2.11.0` from that index first and then run
`pip install -r requirements.txt`. deepmd-kit supports DPA4 model compilation on PyTorch 2.11–2.13; PyTorch 2.11.0 is
required for the prebuilt PyPI wheel of deepmd-kit 3.2.0.

## Usage

1. Write the training input with one of the `make_input_*.py` scripts (or use your own `input.json`).
2. Start training with the entry script of the folder instead of `dp`.
3. Choose the optimizer with environment variables.

```bash
cd dpa3
BATCH=auto:1024 python make_input_mptrj.py 20000 0 $DATA input.json 8      # 20,000 steps, seed 0, learning rate × 8
ARM=bm BM_START=6154 MUON_MOMENTUM=0.95 BM_BETA_FAST=0.90 BM_BETA_SLOW=0.995 BM_W_FAST=0.30 ISO=1 PM=1 PM_GATE=1 PM_WMAX=0.3 PM_SHADOW=1 PM_PHI=0 \
NUM_WORKERS=0 python run_dp_pm.py --pt train input.json --skip-neighbor-stat
```

`$DATA` holds the data set in deepmd's mixed-type format, with `train/` and `valid/` subfolders. Models are tested as
usual with `dp --pt test`.

## Optimizer switches

| change | DPA-3 (`dpa3/muon_pm_dp.py`) | DPA4 (`dpa4/pm_hybrid.py`) |
|---|---|---|
| baseline | `ARM=muon`, `MUON_MOMENTUM` | `ARM=hmuon`; momentum from input.json (`HM_MOMENTUM` of `make_input_*.py`, default 0.95) |
| ① two relaxation modes | `ARM=bm`, `BM_START`, `BM_BETA_FAST`, `BM_BETA_SLOW`, `BM_W_FAST`; `MUON_MOMENTUM` before the switch | `ARM=bm`, `BM_SCHED=step:b_fast:b_slow:w`; HybridMuon's momentum before the switch |
| ② fluctuation-calibrated output budget | `ISO=1`; fluctuations from `ISO_K`=4 micro-batches of each step | `ISO=1`; fluctuations from the differences between GPUs, needs at least 2 GPUs |
| ③ gated readout | `PM=1 PM_GATE=1 PM_WMAX=…` | `PM=1 PM_GATE=1 PM_WMAX=…` |
| ④ tracking from step 0 | `PM=1 PM_SHADOW=1` | `PM=1 PM_SHADOW=1` |
| ⑤ fresh-share cap | `PM=1 PM_PHI=φ` (`PM_PHI=0`: off) | `PM=1 PM_PHI=φ` (`PM_PHI=0`: off) |
| spike guard | — | `BM_GATE_K=10` (window `BM_GATE_WIN`=1000 steps) |

- With `PM=1` the defaults are `PM_GATE=1`, `PM_WMAX=0.6`, `PM_SHADOW=1` and `PM_PHI=0.25`. On potentials always set
  `PM_WMAX=0.3` and `PM_PHI=0`.
- The constants of ② (`ISO_LAM`=0.5, `ISO_CLAMP`=2, `ISO_BETA`=0.95) and of ③ (`PM_OSC_BETA`=0.9) are the defaults and
  were the same on every task.

## Recommended settings

Set the switch step to 30.77% of the training steps and keep the learning rate of the Muon baseline. All six
interatomic-potential tasks in the paper share the same two-mode gated readout (`①–④`, `PM_WMAX=0.3`, `PM_PHI=0`).

**DPA-3** (exponential learning-rate decay, gradient clipping 5; one GPU). The last argument of `make_input_*.py`
multiplies both ends of the learning-rate schedule 1e-3 → 3e-5; `BATCH` is the number of atoms per step.

```bash
BATCH=auto:1024 python make_input_mptrj.py  20000 $SEED $DATA input.json 8    # MPtrj
BATCH=auto:256  python make_input_spice2.py  8000 $SEED $DATA input.json 16   # SPICE2
BATCH=auto:2048 python make_input_spice2.py 20000 $SEED $DATA input.json 8    # SPICE2, large batch
BATCH=auto:2048 python make_input_ani1x.py  20000 $SEED $DATA input.json 8    # ANI-1x
NUM_WORKERS=0 <optimizer variables> python run_dp_pm.py --pt train input.json --skip-neighbor-stat
```

| task | Muon baseline | PhysMuon |
|---|---|---|
| MPtrj | `ARM=muon MUON_MOMENTUM=0.98` | `ARM=bm BM_START=6154 MUON_MOMENTUM=0.95 BM_BETA_FAST=0.90 BM_BETA_SLOW=0.995 BM_W_FAST=0.30 ISO=1 PM=1 PM_GATE=1 PM_WMAX=0.3 PM_SHADOW=1 PM_PHI=0` |
| SPICE2 | `ARM=muon MUON_MOMENTUM=0.95` | `ARM=bm BM_START=2461 MUON_MOMENTUM=0.95 BM_BETA_FAST=0.85 BM_BETA_SLOW=0.98 BM_W_FAST=0.30 ISO=1 PM=1 PM_GATE=1 PM_WMAX=0.3 PM_SHADOW=1 PM_PHI=0` |
| SPICE2, large batch | `ARM=muon MUON_MOMENTUM=0.98` | `ARM=bm BM_START=6154 MUON_MOMENTUM=0.98 BM_BETA_FAST=0.90 BM_BETA_SLOW=0.995 BM_W_FAST=0.30 ISO=1 PM=1 PM_GATE=1 PM_WMAX=0.3 PM_SHADOW=1 PM_PHI=0` |
| ANI-1x | `ARM=muon MUON_MOMENTUM=0.98` | `ARM=bm BM_START=6154 MUON_MOMENTUM=0.98 BM_BETA_FAST=0.90 BM_BETA_SLOW=0.995 BM_W_FAST=0.30 ISO=1 PM=1 PM_GATE=1 PM_WMAX=0.3 PM_SHADOW=1 PM_PHI=0` |

For a new data set on DPA-3, start from `BM_BETA_FAST=0.90 BM_BETA_SLOW=0.995` with gated readout `PM_WMAX=0.3` and
adjust the mode rates if needed; see [`docs/hyperparameters.md`](../../docs/hyperparameters.md).

**DPA4-Mini on MPtrj** (the published recipe: 10 epochs = 69,740 steps on two GPUs).

```bash
cd dpa4
NGPU=2 python make_input_mini.py mini 10 $SEED $DATA input.json
NUM_WORKERS=4 <optimizer variables> torchrun --nproc_per_node=2 run_dp_dpa4_pm.py --pt train input.json --skip-neighbor-stat
```

| HybridMuon baseline | PhysMuon |
|---|---|
| `ARM=hmuon` | `ARM=bm BM_SCHED=21459:0.90:0.995:0.2579 ISO=1 PM=1 PM_GATE=1 PM_WMAX=0.3 PM_SHADOW=1 PM_PHI=0` |

With ③ on, the fixed weight in `BM_SCHED` (0.2579) is replaced by the gated readout with the ceiling `PM_WMAX`.

**DPA4-Pro on MPtrj** (the recipe of the Matbench Discovery entry DPA-4.0.1-Pro-MPtrj: 2,000,000 steps on 16 GPUs).

```bash
python make_input_pro.py pro_model_params.json $DATA input.json
```

Launch as for DPA4-Mini. With fewer than 16 GPUs, set `GRAD_ACC` so that GPUs × `GRAD_ACC` = 16. Train with
`ARM=hmuon` up to step 610,000, then continue from that checkpoint (`--restart`, `BM_T0=610000`) with

```bash
ARM=bm BM_SCHED=615400:0.90:0.995:0.2579 BM_GATE_K=10 ISO=1 PM=1 PM_GATE=1 PM_WMAX=0.3 PM_SHADOW=1 PM_PHI=0
```

From step 610,000 on, ② is active and the two modes follow the gradient (④); the update reads them from step 615,400
(30.77% of the run) on. `BM_GATE_K=10` is the spike guard: a step whose gradient norm before clipping exceeds 10 times
the median of the last 1,000 steps, or is not finite, leaves the slow mode unchanged. Setting `ISO_SPIKE_SKIP=1` also
excludes spike steps from ②'s fluctuation estimate and is recommended whenever `BM_GATE_K` is enabled.
