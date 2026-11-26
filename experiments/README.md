# Experiments of the paper

Everything that belongs to the experiments of the paper is in this folder, one subfolder per group of experiments: the
results, the data behind them, a script that recomputes the numbers of the paper from the data, and the scripts and
commands that produced the data. The settings of every task are in
[`docs/hyperparameters.md`](../docs/hyperparameters.md).

| | data | recompute the numbers | rerun |
|---|---|---|---|
| [`language_model/`](language_model) | 137 training logs of GPT-2 (124M and 302M) | `python language_model/score.py` | [`examples/nanogpt`](../examples/nanogpt) |
| [`potentials/`](potentials) | logs and results of the 46 training runs of the five tasks; force errors per element | `python potentials/summarize.py` | [`integrations/deepmd`](../integrations/deepmd), `potentials/data_preparation/` |
| [`molecular_dynamics/`](molecular_dynamics) | outcome of every trajectory; DFT energies and forces; bond scans | `python molecular_dynamics/analyze.py` | `molecular_dynamics/run_md.py` and the DFT scripts next to it |
| [`matbench/`](matbench) | Matbench Discovery metrics of the DPA4-Pro checkpoints; the trained models | — | the evaluation scripts in `matbench/` |
| [`memory_probe/`](memory_probe) | the read-only probe of the optimal memory in eight training runs | `python memory_probe/analyze.py` | — |

The scripts that recompute the numbers need Python with numpy only and run in about a minute; next to each script is the
output it gives on the published data (`expected_output.txt`).

## Language models (modded-nanogpt speed benchmark, GPT-2 124M)

The benchmark fixes the model, the data and the tokens per step, and counts the steps until the held-out validation
loss reaches 3.28. A configuration reaches the target at the first validation step where the mean loss over n seeds
satisfies (3.28 − mean)·√n ≥ 0.004. Adding the changes one by one to the tuned Muon baseline (8 seeds per row):

| configuration | first step at the target | mean validation loss at step 3,250 |
|---|---|---|
| Muon | not reached | 3.27940 |
| + ① | 3,205 | 3.27683 |
| + ①② | 3,160 | 3.27316 |
| + ①②③ † | 3,100 | 3.26986 |
| + ①②③④ | 3,095 | 3.26931 |
| + ①②③④⑤ (PhysMuon) | 3,085 | 3.26873 |
| without ②: ①③ † | 3,140 | 3.27284 |
| without ②: ①③④ | 3,130 | 3.27205 |
| without ②: ①③④⑤ | 3,120 | 3.27124 |

† From this row on, ① uses β_slow = 0.985 and switches at step 750 (before: 0.98 and step 1,000).

- **Leaderboard record configuration.** Keeping every component of the record configuration and replacing only its
  momentum by ① reaches the target at step 2,635 instead of the record's 2,690 (8 seeds); an independent repeat reached
  it at step 2,645. The scripts and the per-seed logs of this comparison are public:
  [bimaxwell-track3-reproduction](https://github.com/orange4664/bimaxwell-track3-reproduction) and pull request
  [#339](https://github.com/KellerJordan/modded-nanogpt/pull/339) of modded-nanogpt. Pull request
  [#340](https://github.com/KellerJordan/modded-nanogpt/pull/340) holds an independent set of 8 seeds with ① alone on
  the tuned Muon baseline (target reached at step 3,210).
- **302M model** (24 layers, width 1,024, 3,250 steps): validation loss 3.2395 with Muon and 3.2216 with PhysMuon,
  lower on each of the 3 seeds.
- **Cost.** A PhysMuon step takes about 2% longer than a Muon step (1.5%–2.9%), and the optimizer state grows from 0.75
  to 1.38 GiB. Because it requires fewer steps, PhysMuon reaches the target loss about 3% faster in wall-clock time.

Reproduce (FineWeb shards in `examples/nanogpt/data/fineweb10B`, prepared with `data/cached_fineweb10B.py` of
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)):

```bash
cd examples/nanogpt
torchrun --standalone --nproc_per_node=1 train_gpt.py --recipe all5 --seed 0      # muon | paper_bm | all5
python ../../experiments/language_model/score.py                                  # tables from the logs
```

`--recipe muon`, `paper_bm` and `all5` run the Muon baseline, the "+ ①" row and PhysMuon; a run of 3,250 steps takes
about 2.5 hours on one GPU. The other rows and the 302M model were run with earlier versions of the same script. Except
for the 28 earlier 302M logs in `gpt302m/` (from which the script source has been removed), the logs in
[`language_model/`](language_model) begin with the complete source of the script that wrote them, followed by the
settings of the run, so each 124M row and the 302M comparison in `gpt302m_physmuon/` can be rerun from its log (the
302M runs set `NUM_LAYERS=24 MODEL_DIM=1024 BATCH_MULT=0.5 LR_MULT=0.7`).

## Interatomic potentials

Validation errors, baseline → PhysMuon; in parentheses the mean seed-paired relative change and the number of seeds on
which PhysMuon is lower. Forces in meV/Å, energies and virials in meV per atom.

| task (model, length) | baseline | force MAE | energy MAE | force RMSE | virial MAE |
|---|---|---|---|---|---|
| MPtrj (DPA-3, 20,000 steps) | Muon | 123.5 → 97.6 (−20.9%, 4/4) | 41.8 → 32.2 (−22.9%, 4/4) | 295.5 → 215.0 (−26.9%, 4/4) | 58.4 → 49.5 (−15.2%, 4/4) |
| SPICE2 (DPA-3, 8,000 steps) | Muon | 69.6 → 63.7 (−8.3%, 6/6) | 18.5 → 16.2 (−12.0%, 6/6) | 120.9 → 111.9 (−7.4%, 6/6) | — |
| SPICE2, large batch (DPA-3, 20,000 steps) | Muon | 30.9 → 28.1 (−9.0%, 6/6) | 5.9 → 4.9 (−16.6%, 6/6) | 57.6 → 52.8 (−8.3%, 6/6) | — |
| ANI-1x (DPA-3, 20,000 steps) | Muon | 46.9 → 42.2 (−10.2%, 4/4) | 7.5 → 5.8 (−22.8%, 4/4) | 85.5 → 76.6 (−10.3%, 4/4) | — |
| MPtrj (DPA4-Mini, 10 epochs) | HybridMuon | 46.7 → 43.3 (−7.4%, 3/3) | 18.5 → 16.8 (−9.4%, 3/3) | 142.9 → 121.0 (−14.4%, 3/3) | 33.7 → 32.3 (−4.2%, 3/3) |

Virials exist only for the two MPtrj tasks. Both optimizers of a task use the same number of steps and the same seeds;
the values of every seed are in [`potentials/per_seed.csv`](potentials/per_seed.csv). The commands for each task are in
[`integrations/deepmd/README.md`](../integrations/deepmd/README.md); the training logs of every run, the preparation of the
three data sets and the force errors per element are in [`potentials/`](potentials).

## Molecular dynamics

All ANI-1x and SPICE2 (large batch) models were run on 20 molecules.

- **Accuracy on MD configurations.** 360 configurations drawn from the ANI-1x trajectories (120 each at 300, 1,000 and
  1,500 K) were recomputed with the data set's DFT level (ωB97x/6-31G(d)). Force MAE 38.0 → 34.2 meV/Å (−9.9%), every
  PhysMuon model better than every Muon model; per-atom energy error −5.1% after removing a constant offset per molecule.
- **Stability.** Unstable trajectories, Muon → PhysMuon (42-ps restricted mean survival time with completed trajectories
  capped at 42 ps):

| test | temperature | trajectories per optimizer | unstable | 42-ps RMST (ps) |
|---|---|---|---|---|
| ANI-1x, routine | 300 K (NVE), 1,000 K | 48, 144 | 0 → 0 | — |
| ANI-1x, routine | 1,500 K | 144 | 1 (0.7%) → 0 | — |
| SPICE2, routine | 300 K (NVE), 600 K, 1,000 K | 48, 144, 144 | 0 → 0 | — |
| SPICE2, stress test | 1,200 K | 336 | 2 (0.6%) → 1 (0.3%) | 41.9 → 41.9 |
| SPICE2, stress test | 1,500 K | 336 | 15 (4.5%) → 6 (1.8%) | 41.0 → 41.6 |
| SPICE2, stress test | 1,800 K | 336 | 88 (26.2%) → 58 (17.3%) | 35.0 → 37.8 |
| SPICE2, stress test | 2,100 K | 336 | 224 (66.7%) → 164 (48.8%) | 23.4 → 29.4 |

Across all four SPICE2 stress-test temperatures combined, total unstable trajectories drop by 30% (329 → 229); at
2,100 K PhysMuon lowers instabilities across all 8 test molecules and extends 42-ps RMST from 23.4 to 29.4 ps. PhysMuon
reaches Muon's 1,500 and 1,800 K instability fractions only at 1,552 and 1,885 K (+50–85 K).

The outcome of every trajectory, the DFT results, the bond-stretch scans and the scripts of the tests are in
[`molecular_dynamics/`](molecular_dynamics).

## Matbench Discovery (DPA4-Pro, trained on MPtrj only)

Comparison with the leaderboard entry DPA-4.0.1-Pro-MPtrj, the same architecture trained with HybridMuon for 2,000,000
steps. PhysMuon values are the exponential moving average of the weights at 1,465,000 steps.

| metric | leaderboard entry | PhysMuon |
|---|---|---|
| F1 (unique prototypes) | 0.8615 | 0.8712 |
| κ_SRME (lower is better) | 0.2114 | 0.1860 |
| RMSD_AUC (higher is better) | 0.7651 | 0.7734 |
| CPS | 0.8650 | 0.8757 |

- PhysMuon already surpasses the 2,000,000-step leaderboard entry (CPS 0.8650) before half the training budget: every
  evaluated checkpoint from 900,000 steps onward exceeds it (0.8654 at 900,000 steps, 0.8698 at 1,100,000 steps, 0.8727
  at 1,300,000 steps, and 0.8757 at 1,465,000 steps).
- Of the 0.0107 gain in CPS, F1 and κ_SRME each contribute nearly half.

The training command is in [`integrations/deepmd/README.md`](../integrations/deepmd/README.md); the evaluation scripts, the
metrics of the reported checkpoints and the trained models are in [`matbench/`](matbench).

## Read-only probe of the optimal memory

During an unmodified training run, a read-only probe inspects the gradients and optimizer state to estimate how many
steps the momentum should average. In eight independent runs this optimal memory grows from the early to the late part of
training for every group of directions (by a factor of 1.13 to 2.43), and more for the weaker directions. The
measurements are in [`memory_probe/`](memory_probe).
