# Matbench Discovery

DPA4-Pro trained on MPtrj with PhysMuon, evaluated on [Matbench Discovery](https://github.com/janosh/matbench-discovery).
The training command is in [`integrations/deepmd`](../../integrations/deepmd); the data split is described in
[`../potentials`](../potentials).

| | |
|---|---|
| [`metrics.csv`](metrics.csv) | the metrics of the checkpoints reported in the paper, and of the public checkpoint of the leaderboard entry evaluated with the same scripts |
| [`checkpoints/`](checkpoints) | the trained models |
| [`freeze.py`](freeze.py), [`relax_wbm.py`](relax_wbm.py), [`metrics.py`](metrics.py) | stability metrics and RMSD on the WBM test set |
| [`kappa_shards.py`](kappa_shards.py), [`kappa_metrics.py`](kappa_metrics.py) | thermal conductivity (κ_SRME) |

## Results

Exponential moving average of the weights; F1, precision, recall, R², MAE and RMSE on the 215,488 unique prototypes of
the 256,963 WBM structures. CPS = 0.5 F1 + 0.4 (1 − κ_SRME / 2) + 0.1 RMSD_AUC, where
RMSD_AUC = mean(max(1 − RMSD / 0.05, 0)).

| model | steps | F1 | precision | recall | R² | MAE (eV/atom) | RMSE (eV/atom) | RMSD_AUC | κ_SRME | CPS |
|---|---|---|---|---|---|---|---|---|---|---|
| PhysMuon | 900,000 | 0.8607 | 0.8459 | 0.8760 | 0.8536 | 0.0283 | 0.0705 | 0.7719 | 0.2106 | 0.8654 |
| PhysMuon | 1,100,000 | 0.8650 | 0.8586 | 0.8715 | 0.8517 | 0.0279 | 0.0710 | 0.7735 | 0.2003 | 0.8698 |
| PhysMuon | 1,300,000 | 0.8686 | 0.8627 | 0.8746 | 0.8492 | 0.0276 | 0.0716 | 0.7739 | 0.1948 | 0.8727 |
| PhysMuon | 1,465,000 | 0.8712 | 0.8631 | 0.8794 | 0.8535 | 0.0272 | 0.0705 | 0.7734 | 0.1860 | 0.8757 |
| leaderboard entry DPA-4.0.1-Pro-MPtrj (HybridMuon) | 2,000,000 | 0.8615 | 0.8596 | 0.8635 | 0.8410 | 0.0284 | 0.0735 | 0.7651 | 0.2114 | 0.8650 |

Evaluating the official public DPA-4.0.1-Pro-MPtrj checkpoint with the same relaxation and phono3py scripts reproduces
its leaderboard scores.

## Evaluation

The steps follow the DPA-4.0.1 entries of Matbench Discovery; `matbench_discovery` is used at commit `bae6e10`.
Inference runs in single precision without TF32 and without mixed precision (`DP_TF32_INFER=0 DP_AMP_INFER=0`).

```bash
# 1. freeze the checkpoint (once per type of GPU)
python freeze.py model_ema.ckpt-1465000.pt model                       # writes model.pt2

# 2. relax the 256,963 WBM structures: FIRE with FrechetCellFilter, fmax 0.02 eV/A, at most 500 steps
for i in $(seq 0 127); do
  python relax_wbm.py --model model.pt2 --wbm-zip 2024-08-04-wbm-initial-atoms.extxyz.zip \
      --wbm-summary 2023-12-13-wbm-summary.csv.gz --out-dir runs/TAG --n-shards 128 --shard-index $i --mod 1
done                                                                    # 8 processes per GPU on 16 GPUs

# 3. stability metrics (MP2020 corrections, energies above the hull, F1 on the unique prototypes) and RMSD
WBM_SUMMARY=2023-12-13-wbm-summary.csv.gz WBM_CSE=2022-10-19-wbm-computed-structure-entries.jsonl.gz \
python metrics.py runs TAG                                              # writes runs/subset_metrics_TAG.csv

# 4. thermal conductivity of the 103 structures, with models/deepmd/test_dpa4_kappa.py of matbench-discovery:
#    FIRE to fmax 1e-4 eV/A (at most 300 steps) with fixed symmetry, displacements of 0.03 A, phono3py 3.30.0 at 300 K
python kappa_shards.py 2024-11-09-phononDB-PBE-103-structures.extxyz 4 shards
for k in 00 01 02 03; do
  DP_ACT_INFER=1 python test_dpa4_kappa.py --checkpoint-path model_ema.ckpt-1465000.pt \
      --output-dir kappa_runs/TAG/shard-$k --structures-path shards/shard-$k.extxyz
done                                                                    # one shard per GPU
python kappa_metrics.py 2024-11-09-kappas-phononDB-PBE-noNAC.json.gz TAG=kappa_runs/TAG
```

The WBM and phonon files are the data files of Matbench Discovery. `relax_wbm.py` can be resumed and retries a
structure when the GPU runs out of memory; the structures that still end with an error are counted in the column
`relaxation_errors` of `metrics.csv` (0 for the 1,465,000-step model).
