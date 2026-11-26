"""Recompute the tables of ablations.md and the 302M comparison of PhysMuon with Muon (gpt302m_physmuon/) from the
training logs in this folder.

    python experiments/language_model/score.py

A log is the record written by the training script itself: the script's own source code (removed from the logs in
gpt302m/), then the run (PyTorch version and GPU, arguments) and the validation losses "step:N/T val_loss:X". Only
the part after the source code is parsed. The score of an ensemble of n seeds is the first validation step, among the steps every seed
logged, at which (3.28 - mean) * sqrt(n) >= 0.004.
"""
import math
import pathlib
import re
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def curve(path):
    text = path.read_text()
    run = text[text.index("Running PyTorch"):]
    return {int(s): float(v) for s, v in re.findall(r"step:(\d+)/\d+ val_loss:([0-9.]+)", run)}


def group(name):
    return {int(p.stem[4:]): curve(p) for p in sorted((HERE / name).glob("seed*.txt"))}


def common_steps(g):
    return sorted(set.intersection(*(set(c) for c in g.values())))


def final(g):
    t = common_steps(g)[-1]
    return t, {s: c[t] for s, c in g.items()}


def score_step(g):
    n = len(g)
    for t in common_steps(g):
        if (3.28 - st.mean(c[t] for c in g.values())) * math.sqrt(n) >= 0.004:
            return t
    return None


def paired(a, b):
    """a - b at the final step, over the seeds both groups have: (mean, wins, n)."""
    fa, fb = final(a)[1], final(b)[1]
    d = [fa[s] - fb[s] for s in sorted(set(fa) & set(fb))]
    return st.mean(d), sum(x < 0 for x in d), len(d)


def fmt_pair(p):
    return f"{p[0]:+.5f} ({p[1]}/{p[2]})"


def table(title, rows, with_score=True):
    """One row per group; with a reference group, n and the mean are over the seeds the two groups share."""
    print(f"\n## {title}\n")
    print("| optimizer | n | loss at final step | paired [against] |" + (" first step meeting the score |" if with_score else ""))
    print("|---|---|---|---|" + ("---|" if with_score else ""))
    for label, name, ref in rows:
        g = group(name)
        t, f = final(g)
        seeds, pair = sorted(f), "—"
        if ref:
            gr = group(ref)
            seeds = sorted(set(f) & set(final(gr)[1]))
            pair = f"{fmt_pair(paired(g, gr))} [{ref.split('/')[-1]}]"
        line = f"| {label} | {len(seeds)} | {st.mean(f[s] for s in seeds):.5f} (step {t}) | {pair} |"
        if with_score:
            s = score_step(g)
            line += f" {s if s else 'not reached'} |"
        print(line)


table("124M GPT, 3,250 steps: the stack", [
    ("Muon (baseline, run as ② with exponent 0)", "gpt124m/muon", None),
    ("+ ① two relaxation modes (fixed weight 0.4385)", "gpt124m/bm_paper", "gpt124m/muon"),
    ("+ ② fluctuation-calibrated output budget", "gpt124m/bm_iso", "gpt124m/bm_paper"),
    ("+ ③ gated readout", "gpt124m/bm_iso_gate", "gpt124m/bm_iso"),
    ("+ ④ tracking from step 0", "gpt124m/bm_iso_gate_shadow", "gpt124m/bm_iso_gate"),
    ("+ ⑤ fresh-share cap", "gpt124m/all_five", "gpt124m/bm_iso_gate_shadow"),
])
table("124M GPT, 3,250 steps: the stack without ②", [
    ("① two relaxation modes, fixed weight 1/6", "gpt124m/bm_w16", None),
    ("① + ③ gated readout, switch at 1,000", "gpt124m/bm_gate_switch1000", "gpt124m/bm_w16"),
    ("① + ③, switch at 750", "gpt124m/bm_gate", "gpt124m/bm_gate_switch1000"),
    ("① + ③ + ④ tracking from step 0", "gpt124m/bm_gate_shadow", "gpt124m/bm_gate"),
    ("① + ③ + ④ + ⑤ fresh-share cap", "gpt124m/bm_gate_shadow_fresh", "gpt124m/bm_gate_shadow"),
])
table("124M GPT, 3,250 steps: single changes and controls", [
    ("② fluctuation-calibrated output budget alone", "gpt124m/iso", "gpt124m/muon"),
    ("unmodified baseline script", "gpt124m/muon_unmodified", "gpt124m/muon"),
])
rep, orig, pre = group("gpt124m/physmuon_all_five"), group("gpt124m/all_five"), group("gpt124m/physmuon_all_five_prerename")
first = lambda c: min((t for t, v in c.items() if v <= 3.28), default=None)
print("\nreproduction with this package (examples/nanogpt/train_gpt.py --recipe all5) against the original script, same seed:")
for s in sorted(rep):
    print(f"  seed {s}: final {final(rep)[1][s]:.5f} vs {final(orig)[1][s]:.5f} ({final(rep)[1][s] - final(orig)[1][s]:+.5f}); "
          f"first validation step at or below 3.28: {first(rep[s])} vs {first(orig[s])}; "
          f"the same code before the rename: {final(pre)[1][s]:.5f}")
A, B, C, D = (group(f"gpt124m/{k}") for k in ("muon", "bm_paper", "iso", "bm_iso"))
fA, fB, fC, fD = (final(x)[1] for x in (A, B, C, D))
inter = [fD[s] - fB[s] - fC[s] + fA[s] for s in sorted(fA)]
m, se = st.mean(inter), st.stdev(inter) / math.sqrt(len(inter))
h = T975[len(inter) - 1] * se
print(f"\ninteraction of ① and ② (bm_iso - bm_paper - iso + muon): {m:+.5f}, 95% interval [{m - h:+.5f}, {m + h:+.5f}], n={len(inter)}")

table("302M GPT, 3,250 steps", [
    ("Muon", "gpt302m/muon", None),
    ("① two relaxation modes", "gpt302m/bm", "gpt302m/muon"),
    ("② fluctuation-calibrated output budget", "gpt302m/iso", "gpt302m/muon"),
    ("① + ②", "gpt302m/bm_iso", "gpt302m/muon"),
    ("① + ② + ③ gated readout", "gpt302m/bm_iso_gate", "gpt302m/muon"),
    ("① + ③ gated readout (against the memory alone)", "gpt302m/bm_gate", "gpt302m/bm"),
], with_score=False)   # the 302M model passes 3.28 long before step 3,250; only the final loss is compared
target = st.mean(final(group("gpt302m/muon"))[1].values())
pts = [(t, st.mean(final(group(f"gpt302m/bm_iso_T{t}"))[1].values())) for t in (2600, 2800, 3000)]
pts.append((3250, st.mean(final(group("gpt302m/bm_iso"))[1].values())))
print("\n302M, ① + ② with the whole schedule compressed to T steps (final loss): "
      + ", ".join(f"T={t}: {v:.5f}" for t, v in pts) + f"; Muon at 3,250: {target:.5f}")
(t0, v0), (t1, v1) = pts[-2], pts[-1]
t_star = t0 + (v0 - target) * (t1 - t0) / (v0 - v1)
print(f"steps for ① + ② to reach Muon's 3,250-step loss (linear between T={t0} and T={t1}): {t_star:.0f} "
      f"({100 * (1 - t_star / 3250):.1f}% fewer)")

table("302M GPT, 3,250 steps: PhysMuon against Muon (gpt302m_physmuon/, one script)", [
    ("Muon", "gpt302m_physmuon/muon", None),
    ("PhysMuon, all five changes (φ = 0.25)", "gpt302m_physmuon/all_five", "gpt302m_physmuon/muon"),
], with_score=False)
