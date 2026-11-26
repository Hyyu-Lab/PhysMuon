# Molecular dynamics

The molecular-dynamics tests of the paper: the trajectory statistics, the DFT results, the script that recomputes the
numbers of the paper from them, and the scripts that produced them.

| | |
|---|---|
| [`analyze.py`](analyze.py) | recomputes the numbers of the paper from `data/`; its output is [`expected_output.txt`](expected_output.txt) |
| [`data/routine/`](data/routine), [`data/stress/`](data/stress) | one file per task (model × molecule) with the outcome of its seven trajectories |
| [`data/dft_ani1x/`](data/dft_ani1x), [`data/model_predictions/`](data/model_predictions) | 360 configurations from the ANI-1x trajectories with their DFT energies and forces, and the predictions of the eight ANI-1x models |
| [`data/bond_scan/`](data/bond_scan) | bond-stretch scans: geometries, energies of the 12 SPICE2 models, DFT energies |
| [`run_md.py`](run_md.py) | the molecular-dynamics test (routine and stress) |
| [`evaluate_snapshots.py`](evaluate_snapshots.py), [`dft_ani1x.py`](dft_ani1x.py) | model predictions and DFT reference on the saved configurations |
| [`bond_scan.py`](bond_scan.py), [`dft_spice2.py`](dft_spice2.py) | bond-stretch scans with the models and their DFT reference |

## The tests

- **Models.** The final checkpoints (20,000 steps) of the ANI-1x runs (Muon and PhysMuon, 4 seeds each) and of the
  SPICE2 large-batch runs (Muon and PhysMuon, 6 seeds each). In file names and fields the optimizers are `muon` and
  `physmuon`. The training settings of every run are in [`../potentials/runs.tsv`](../potentials/runs.tsv).
- **Molecules.** One configuration per number of atoms from the validation set of each data set (see `run_md.py` for
  the rule): 12 molecules for ANI-1x and 8 for SPICE2.

  | | m0 | m1 | m2 | m3 | m4 | m5 | m6 | m7 | m8 | m9 | m10 | m11 | m12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | ANI-1x | C2H7N | C5H4O3 | C5H8 | C3H7N3O2 | C3H7N5O | C7H6N4 | C8H6N2O2 | — | C9H8N2O2 | C7H11N3O2 | C9H12O4 | C10H15NO | C8H13N |
  | SPICE2 | C5H10Cl2Si | C8H13NO2 | C10H17NOS | C17H18Si | C14H23NO3Si | C14H32N2 | C21H25N3O5 | C25H27N5O3 | | | | | |

  Index 7 of ANI-1x is not part of the test (the molecule with 22 atoms, m12, was run in its place).
- **Routine test.** Every model on every molecule: one trajectory at 300 K that runs without thermostat after 2 ps, and
  six trajectories at two temperatures (ANI-1x 1,000 and 1,500 K; SPICE2 600 and 1,000 K), 42 ps each.
- **Stress test.** Seven trajectories of 42 ps at one temperature on SPICE2 at 1,200, 1,500, 1,800 and 2,100 K (336
  trajectories per optimizer and temperature).
- **Accuracy on MD configurations.** From the routine trajectories of the two seed-0 ANI-1x models, one configuration
  every 8 ps at each of the three temperatures: 360 configurations, recomputed at ωB97x/6-31G(d).
- **Bond-stretch scans.** Three covalent bonds frequently broken under thermal stress (Si–Cl, C–H, Si–C), stretched
  rigidly by up to 3 Å, with the 12 SPICE2 models and with ωB97M-D3(BJ)/def2-TZVPPD.

## Recomputing the numbers of the paper

```bash
python analyze.py > output.txt          # numpy only
diff output.txt expected_output.txt
```

| section of the output | in the paper |
|---|---|
| 1. Routine test | the routine rows of the table of unstable trajectories; energy conservation |
| 2. SPICE2 stress tests | the stress rows of that table and the stability panel of the Matbench/MD figure; the supplementary tables of the stress test (fractions over time, seed-level statistics, per-molecule counts); the equivalent tolerated temperature shift (50–85 K) |
| 3. Temperature scaling | the shared-prefactor Arrhenius effective barrier ratio |
| 4. Accuracy on MD configurations | the "ANI-1x MD snapshots" row of the table of potential errors; the three temperatures |
| 5. Bond-stretch scans | the supplementary table and text on bond scans |

Seed-level permutation p values and seed-cluster 95% confidence intervals in section 2 are computed by exact enumeration
over all 2⁶ = 64 sign assignments and all 6⁶ = 46,656 seed draws.

## Data

- `data/routine/<data set>_<optimizer>_s<seed>_m<molecule>.json`, `data/stress/..._T<temperature>.json`: the output
  of `run_md.py` for one task. `temps` is the temperature of each trajectory; `replica_failures` holds, for each
  trajectory, `null` or the record of its instability (`t_ps`: time of the first check that found it, `fragments`,
  `fragment_formulas`, `broken_bonds`, and for the stress test `confirmed_ps`); `transient_splits` counts the splits
  that healed within 100 fs; `nve_drift_meV_atom_ps` is the drift of the total energy of the trajectory without
  thermostat. `frame` is the index of the starting configuration in the validation set.
- `data/dft_ani1x/ani1x_<optimizer>_s0_m<molecule>.npz`: the 15 configurations taken from the trajectories of one
  model on one molecule (`coords`, `types`, `type_map`, `labels`: the temperature, `t_ps`) with the DFT results (`E` in
  eV, `F` in eV/Å, `converged`). `data/model_predictions/` holds, under the same file names, `E_<optimizer>_s<seed>`
  and `F_<optimizer>_s<seed>` of the eight models. `validation.json`: the DFT setup against the labels of the data set
  on 12 validation configurations.
- `data/bond_scan/geometries.npz` (per scan: `X` the geometries, `r` the bond lengths, `frag` the atoms that move),
  `model_energies.npz` (`DS`: the stretches; `<scan>|<optimizer>|s<seed>`: energies in eV),
  `dft_restricted_<scan>.json` and `dft_unrestricted_<scan>.json` (energies in eV at ten stretches; the reference of the
  paper is the lower of `E_RKS_eV` and `E_UKS_eV`), `dft_validation.json` (the DFT setup against the labels of the data
  set on the three starting configurations).

## Running the tests

Requires deepmd-kit 3.2.0 (PyTorch backend) for the models and PySCF with pyscf-dispersion for the DFT reference.
`MODELS` is a folder with the trained models as `<data set>_<optimizer>_s<seed>.pt` (the `model.ckpt-20000.pt` of each
run, see [`integrations/deepmd`](../../integrations/deepmd)); `DATA` is the data set in deepmd's mixed-type format (see
[`../potentials`](../potentials)).

```bash
# one task of the routine test and one of the stress test: <data set> <optimizer> <seed> <molecule>
python run_md.py ani1x  physmuon 0 3 --models MODELS --valid ANI1X/valid  --out results/routine
python run_md.py spice2 physmuon 0 3 --models MODELS --valid SPICE2/valid --out results/stress --stress 1500
python run_md.py spice2 physmuon 0 3 --models MODELS --valid SPICE2/valid --out test --smoke      # 0.6 ps, a few minutes

# accuracy on the configurations saved by the seed-0 ANI-1x tasks of the routine test
python evaluate_snapshots.py --models MODELS --snapshots results/routine/snapshots --out model_predictions
python dft_ani1x.py snapshots results/routine/snapshots dft_ani1x
python dft_ani1x.py validate ANI1X/valid dft_ani1x/validation.json

# bond-stretch scans
python bond_scan.py --models MODELS --valid SPICE2/valid --out bond_scan
python dft_spice2.py validate SPICE2/valid bond_scan/dft_validation.json
python dft_spice2.py restricted   m0_Si7-Cl1 bond_scan -0.2 0 0.2 0.5 0.7 1 1.5 2 2.5 3
python dft_spice2.py unrestricted m0_Si7-Cl1 bond_scan -0.2 0 0.2 0.5 0.7 1 1.5 2 2.5 3
```

`python analyze.py --data DIR` reads results arranged as in `data/`. A task of seven 42 ps trajectories takes between
half an hour and four hours, with four tasks sharing one GPU. The dynamics are chaotic: a repeated task does not follow
the same trajectories, and single instabilities differ from the stored ones; the fractions over the hundreds of
trajectories of a test are what the paper compares.
