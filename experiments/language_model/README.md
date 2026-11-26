# Language models: training logs

中文说明：[README_zh.md](README_zh.md)。

The training logs of the language-model runs: 131 logs behind [`ablations.md`](ablations.md) and the 6 logs
of the 302M comparison of PhysMuon with Muon, 137 in all (about 3.3 MB). A log is the record written by the training
script: the full source of the script, the settings of the run, and `step:N/T val_loss:X` at every validation step.
The 28 logs in `gpt302m/` omit the embedded script source; the run settings and validation losses are unchanged.

- `python experiments/language_model/score.py` recomputes the tables of `ablations.md` and the 302M comparison in
  `gpt302m_physmuon/` from the logs. A group reaches the target at the first
  validation step where the mean loss over its n seeds satisfies (3.28 − mean)·√n ≥ 0.004, using only steps recorded
  by every seed.
- [`manifest.tsv`](manifest.tsv) lists, for every log, the group, the seed, the run name, the number of steps, the
  final validation loss and the SHA-256; `sha256sum -c SHA256SUMS` checks all files.

GPT-2 124M on the modded-nanogpt benchmark, 3,250 steps, seeds 0–7:

| folder | configuration |
|---|---|
| `gpt124m/muon` | the tuned Muon baseline |
| `gpt124m/bm_paper` | + ① |
| `gpt124m/bm_iso` | + ①② |
| `gpt124m/bm_iso_gate` | + ①②③ |
| `gpt124m/bm_iso_gate_shadow` | + ①②③④ |
| `gpt124m/all_five` | all five changes (PhysMuon) |
| `gpt124m/bm_gate`, `bm_gate_shadow`, `bm_gate_shadow_fresh` | without ②: ①③, ①③④ and ①③④⑤ |
| `gpt124m/bm_w16`, `bm_gate_switch1000` | ① with the fixed weight 1/6; ①③ switched on at step 1,000 |
| `gpt124m/iso` | ② alone |
| `gpt124m/muon_unmodified` | the unmodified baseline script, seeds 0–2 |
| `gpt124m/physmuon_all_five`, `physmuon_all_five_prerename` | all five changes rerun with this package (`examples/nanogpt/train_gpt.py --recipe all5`), seeds 0–1, two runs of each seed |

302M GPT (24 layers, width 1,024, 3,250 steps):

| folder | configuration |
|---|---|
| `gpt302m_physmuon/muon`, `gpt302m_physmuon/all_five` | the comparison of the paper: Muon and PhysMuon in one script, seeds 0–2 |
| `gpt302m/` | an earlier version of the script: Muon, ①, ②, ①②, ①②③, ①③, and ①② with the schedule shortened to 2,600, 2,800 and 3,000 steps |

The 302M runs pass 3.28 well before step 3,250, so only their final validation losses are compared; `gpt302m/` and
`gpt302m_physmuon/` come from different scripts and their final losses are not directly comparable. Logs for the
constant sensitivity scans (4 seeds each) in `ablations.md` are not included here; the independent 8-seed runs are in
PR #340 of modded-nanogpt and Appendix E of arXiv:2608.22994.

In the logs of `bm_w16` and `bm_gate_switch1000`, the `recipe:` summary line right after the PyTorch header prints the
defaults of the script; the settings actually in effect are on the `diag:` line that follows.
