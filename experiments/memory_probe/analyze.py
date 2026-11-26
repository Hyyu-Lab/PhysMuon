"""Recompute the numbers of the read-only probe (Supplementary Note 6) from the measurements.

    python analyze.py                          # from data/buckets_s<seed>.csv
    python analyze.py --raw slim_s0.jsonl.gz   # in addition: spread and ordering over the 360 measuring units of seed 0

data/buckets_s<seed>.csv holds, for every training step of one run, ln K* of the five groups of directions (the median
over the 72 hidden matrices; b1_top is the strongest 5% of the directions, b5_tail the weakest 35%). The optimal memory
is n* = (sqrt(1 + 4 / K*^2) - 1) / 2. Early window: steps 700-1,500; late window: steps 2,200-2,700.
The file for --raw holds ln K* of every unit at every step and is part of the public record of the probe runs
(kstar_probe/slim_s0.jsonl.gz in https://github.com/orange4664/bimaxwell-track3-reproduction).
"""
import argparse, csv, gzip, json, math, os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["b1_top", "b2", "b3", "b4", "b5_tail"]
EARLY, LATE = (700, 1500), (2200, 2700)


def nstar(lnk):
    tau = math.exp(-lnk)
    return (math.sqrt(1 + 4 * tau * tau) - 1) / 2


def window_medians(seed):
    rows = [r for r in csv.DictReader(open(f"{HERE}/data/buckets_s{seed}.csv")) if int(r["step"]) >= 250]
    med = lambda c, w: st.median(nstar(float(r[c])) for r in rows if r[c] != "nan" and w[0] <= int(r["step"]) <= w[1])
    return [med(c, EARLY) for c in COLS], [med(c, LATE) for c in COLS]


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--raw", default=None)
a = ap.parse_args()
e, l = window_medians(0)
print("seed 0, optimal memory n* (steps), median of each group of directions in the early and in the late window")
for j in range(5):
    print(f"  group {j + 1}: {e[j]:.1f} -> {l[j]:.1f}   (late / early {l[j] / e[j]:.2f})")
print("\nlate / early ratio of n* in the eight runs")
ratios = []
for seed in range(8):
    e, l = window_medians(seed); r = [l[j] / e[j] for j in range(5)]; ratios += r
    print(f"  seed {seed}: " + "  ".join(f"{v:.2f}" for v in r))
print(f"  {sum(v > 1 for v in ratios)} of {len(ratios)} ratios above 1; smallest {min(ratios):.2f}, largest {max(ratios):.2f}")
print("  mean over the runs, by group: " + "  ".join(f"{st.mean(ratios[j::5]):.2f}" for j in range(5)))

if a.raw:
    import numpy as np
    win = {EARLY: {}, LATE: {}}                           # window -> (matrix, group) -> values of ln K* in the window
    with gzip.open(a.raw, "rt") as f:
        for line in f:
            r = json.loads(line)
            for w in win:
                if w[0] <= r["step"] <= w[1]:
                    for key, arr in r["iprobe_logK"].items():
                        for b, v in enumerate(arr[:5]):
                            if v is not None and math.isfinite(v): win[w].setdefault((key, b), []).append(float(v))
    m1, m2 = ({k: float(np.median(v)) for k, v in win[w].items()} for w in (EARLY, LATE)); common = sorted(set(m1) & set(m2))

    def ranks(x):
        order = np.argsort(x, kind="mergesort"); out = np.empty(len(x)); i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]: j += 1
            out[order[i:j + 1]] = 0.5 * (i + j) + 1.0; i = j + 1
        return out

    ra, rb = ranks(np.array([m1[c] for c in common])), ranks(np.array([m2[c] for c in common])); ra -= ra.mean(); rb -= rb.mean()
    iqr = lambda m: float(np.percentile(list(m.values()), 75) - np.percentile(list(m.values()), 25))
    print(f"\nseed 0, {len(common)} measuring units (matrix x group): interquartile range of ln K* {iqr(m1):.3f} (early) -> {iqr(m2):.3f} (late); "
          f"Spearman rank correlation between the two windows {float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum())):.2f}")
