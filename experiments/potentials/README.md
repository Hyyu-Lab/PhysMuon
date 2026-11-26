# Interatomic potentials

The five training tasks of the paper (DPA-3 on MPtrj, SPICE2, SPICE2 with a large batch and ANI-1x; DPA4-Mini on
MPtrj): the results and training logs of every seed, the preparation of the data sets, and the force errors per
element. The training commands are in [`integrations/deepmd`](../../integrations/deepmd).

| | |
|---|---|
| [`per_seed.csv`](per_seed.csv) | validation errors of every seed |
| [`summarize.py`](summarize.py) | recomputes the tables of the paper from `logs/` and `per_element/`; its output is [`expected_output.txt`](expected_output.txt) |
| [`runs.tsv`](runs.tsv) | the 46 runs: task, optimizer, seed, the command that writes `input.json`, the optimizer variables, the log folder |
| [`logs/`](logs) | per run: `lcurve.out`, `train.log` and the result of `dp test` on the validation set |
| [`data_preparation/`](data_preparation) | conversion of the three data sets to deepmd's mixed-type format, with the validation split |
| [`per_element/`](per_element) | force error per element of the SPICE2 large-batch models |

```bash
python summarize.py > output.txt          # numpy only
diff output.txt expected_output.txt
```

## Logs

`logs/<task>/<optimizer>/seed<k>/` holds the files of one run as deepmd-kit wrote them:

- `lcurve.out`: training and validation errors during training;
- `train.log`: the log of the run, with the versions of deepmd-kit and PyTorch, the GPU, the data summary, the timing
  and the lines printed by the optimizer;
- `test.log`: the end of the output of `dp --pt test` on the whole validation set after training; the block after
  "weighted average of errors" is what the paper reports. For DPA4-Mini the paper reports `test_ema.log`, the same
  test with the weights averaged over training.

Local paths and host names were stripped from `train.log`; all training and evaluation records are verbatim.

## Data sets

| data set | source | training / validation configurations | validation split |
|---|---|---|---|
| MPtrj | `2024-09-03-mp-trj.extxyz.zip`, the extended-XYZ files of MPtrj distributed with [Matbench Discovery](https://github.com/janosh/matbench-discovery) (145,923 materials, 1,580,395 configurations) | 1,564,344 / 16,051 | 1% of the materials (1,459), drawn with random seed 0; all configurations of a material are in the same set |
| SPICE2 | [`InstaDeepAI/SPICE2-curated`](https://huggingface.co/datasets/InstaDeepAI/SPICE2-curated) | 1,734,158 / 87,764 | the split of the source (by molecule) |
| ANI-1x | [`roitberg-group/ani1x`](https://huggingface.co/datasets/roitberg-group/ani1x) (`ANI-1x-wB97X-631Gd.h5`) | 4,852,676 / 103,329 | 2% of the molecules (56 of 3,114), chosen by a hash of the element sequence; all conformers of a molecule are in the same set |

```bash
cd data_preparation
python ani1x_to_deepmd.py  ANI-1x-wB97X-631Gd.h5 ANI1X          # writes ANI1X/train, ANI1X/valid
python spice2_to_deepmd.py SPICE2-curated.zip SPICE2
python mptrj_to_deepmd.py  EXTXYZ_DIR MPTRJ_SYSTEMS 0 1         # needs ase and dpdata
python mptrj_split.py      MPTRJ_SYSTEMS MPTRJ ../../../integrations/deepmd/dpa4/pro_model_params.json
```

[`data_preparation/mptrj_validation_ids.txt`](data_preparation/mptrj_validation_ids.txt) lists the 1,459 validation
materials of MPtrj. The same MPtrj split is used by the DPA-3 task, by DPA4-Mini and by DPA4-Pro
([`../matbench`](../matbench)).

## Force error per element

`per_element/<optimizer>.json` holds, for each of the six SPICE2 large-batch models of an optimizer, the summed absolute
force error and the number of atoms of every element on the validation set; `training_set_elements.json` holds the
element counts of the training set. They were written by

```bash
cd per_element
python element_errors.py physmuon --models MODELS --valid SPICE2/valid       # muon | physmuon
python element_frequency.py SPICE2/train training_set_elements.json
```

where `MODELS` holds the trained models as `spice2_<optimizer>_s<seed>.pt` (the `model.ckpt-20000.pt` of each run).
