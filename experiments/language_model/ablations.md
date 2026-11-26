# GPT-2: the effect of each change and the scans of the constants

The training logs behind the tables of this page are in this folder; `python experiments/language_model/score.py`
recomputes the tables from them.

All numbers come from the nanoGPT optimization track ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), track 3):
a fixed GPT on FineWeb, a fixed batch of 524,288 tokens per step and a fixed number of steps; only the optimizer may
change. The reference is the track's tuned baseline (#36: Muon, learning rate 0.025, weight decay 0.05, Nesterov
μ = 0.95, cubic Newton–Schulz with 12 steps, AdamW for embeddings, head and scalars, constant learning rate followed
by a linear cooldown over the last 70% of the steps). The score of an ensemble of n seeds is the first validation
step at which `(3.28 − mean validation loss) · sqrt(n) >= 0.004`.

Every configuration below is the baseline script with only the hidden-matrix optimizer changed. Comparisons are paired:
the same seeds 0–7 and the same data order. "Paired" is the mean
difference of the validation loss at step 3,250 against the row named in brackets, with the number of seeds on
which the change was better.

## 124M GPT, 3,250 steps, n = 8

Adding one change at a time:

| optimizer | loss at 3,250 | paired | first step meeting the score |
|---|---|---|---|
| Muon (baseline) | 3.27940 | — | not reached by 3,250 |
| + ① two relaxation modes (fixed weight 0.4385) | 3.27683 | −0.00257 (8/8) [Muon] | 3,205 |
| + ② fluctuation-calibrated output budget | 3.27316 | −0.00366 (8/8) [row above] | 3,160 |
| + ③ gated readout | 3.26986 | −0.00331 (8/8) [row above] | 3,100 |
| + ④ tracking from step 0 | 3.26931 | −0.00054 (8/8) [row above] | 3,095 |
| + ⑤ fresh-share cap (φ = 0.25) | **3.26873** | −0.00058 (8/8) [row above] | **3,085** |

Rows 1–2 use the slow mode 0.98, the fixed fast weight 0.4385 and the switch at step 1,000; from row 3 on the
slow mode is 0.985 and the switch is at step 750, the best values with the gate (see the scans below). To isolate
the diagonal metric of ②, the Muon and + ① rows use the same final normalization as ② (λ = 0); on seeds 0–2 this
is slightly stronger than the unmodified baseline script (3.27949).

The same sequence without ②:

| optimizer | loss at 3,250 | paired | first step meeting the score |
|---|---|---|---|
| ① two relaxation modes, fixed weight 1/6 (best fixed kernel) | 3.27588 | — | 3,200 |
| ① + ③ gated readout, switch at 1,000 | 3.27337 | −0.00251 (8/8) [row above] | 3,150 |
| ① + ③, switch at 750 | 3.27284 | −0.00053 (7/8) [row above] | 3,140 |
| ① + ③ + ④ tracking from step 0 | 3.27205 | −0.00079 (8/8) [row above] | 3,130 |
| ① + ③ + ④ + ⑤ fresh-share cap | **3.27124** | −0.00081 (8/8) [row above] | **3,120** |

Single changes on Muon: ② alone 3.27545 (−0.00395, 8/8; score 3,190). Two relaxation modes alone with the
constants of row 1: 3,205 (3,210 in an independent set of 8 seeds, PR #340); with the fast-mode weight 1/6
(a mean memory of 42 steps) 3,200 (3,190 in the independent set). ② and ① are close to additive (interaction
+0.00028, 95% interval [−0.00017, +0.00073]).

Reproduction with this package: `examples/nanogpt/train_gpt.py --recipe all5` (the baseline script with `PhysMuon`)
ended at 3.26932 and 3.26792 on seeds 0 and 1 (mean 3.26862), matching the original script on the same seeds (3.26856
and 3.26939, mean 3.26898); a second pair of runs in `gpt124m/physmuon_all_five_prerename` gave 3.26867 and 3.26800.

## 302M GPT, 3,250 steps, n = 3

24 layers of width 1,024, the same data, schedule and number of steps, half the batch, the Muon learning rate
retuned (0.7× the 124M value, best of 0.5×, 0.7×, 1.0×). These runs used an earlier version of the script (its source
is not included in the logs); the paper's
302M comparison (all five changes against Muon in one script) is in [README.md](README.md) and
[`gpt302m_physmuon/`](gpt302m_physmuon), and final losses of the two sets are not comparable.

| optimizer | loss at 3,250 | paired vs Muon |
|---|---|---|
| Muon | 3.23809 | — |
| ① two relaxation modes | 3.23520 | −0.00289 (3/3) |
| ② fluctuation-calibrated output budget | 3.22499 | −0.01310 (3/3) |
| ① + ② | 3.22251 | −0.01558 (3/3) |
| ① + ② + ③ gated readout | 3.22120 | −0.01689 (3/3) |

The gated readout on ① alone: −0.00181 (5/5 seeds). Shortening the schedule shows that ① + ② reaches
Muon's 3,250-step loss in about 3,036 steps (6.6% fewer).

## Constants and how sensitive they are

Scans on the 124M GPT, n = 4 paired seeds unless stated; the recipe's values are in bold.

| constant | values tried | result |
|---|---|---|
| slow mode `bm_beta_slow` (with the gate) | 0.98, **0.985**, 0.99 | −0.0033, **−0.0039**, −0.0012 against ① with the fixed weight 0.4385 |
| largest fast weight `osc_wmax` | 0.45, 0.5, **0.6**, 0.7 | 0.6 best; 0.45 loses 0.0006 (① alone), 0.5 and 0.7 lose ≈0.0004 (all five changes) |
| switch step `bm_start` | 0, 500, **750**, 1,000 | 0 costs +0.0065 (a permanent offset from the early transient); 500 ≈ 750; 1,000 loses about 0.0005 |
| fresh share `fresh_cap` φ | 0.15, **0.25**, 0.35, 0.45 | −0.00037, **−0.00052**, −0.00045, −0.00016 on all five changes; −0.00035, **−0.00077**, −0.00058, −0.00054 without ② |
| ② exponent `iso_lam` | 0.25, **0.5**, 1.0 | 0.5 best; 1.0 loses ≈0.002, 0.25 keeps ≈80% of the gain; at 302M 0.375 and 0.75 lose 0.0002 and 0.0014 |
| fast mode `bm_beta_fast` (all five changes) | 0.80, **0.85**, 0.90 | +0.00104 (0/4), **0**, +0.00060 (0/4) |
| gate EMA rate `osc_beta` (all five changes) | 0.80, **0.9**, 0.95 | +0.00013, **0**, +0.00011: flat |
| EMA rate of ②'s fluctuation estimate `iso_beta` (all five changes) | 0.90, **0.95** | +0.00031 (0/4) at 0.90 |
| ② bound `iso_clamp` (all five changes) | **2**, 3 | no difference (−0.00000, 2/4) |

Not changed from the baseline: learning rate, weight decay, schedule, Nesterov μ, batch.
