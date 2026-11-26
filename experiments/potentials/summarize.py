"""Recompute the interatomic-potential tables of the paper from the files in this folder.

    python summarize.py            # prints the tables; --write-csv rewrites per_seed.csv from the logs

1. Validation errors of the five tasks, from the output of `dp test` in logs/ (test.log; test_ema.log, the weights
   averaged over training, for DPA4-Mini): baseline -> PhysMuon, the mean of the seed-paired relative changes and the
   number of seeds on which PhysMuon is lower.
2. Force MAE per element on SPICE2 (large batch), from per_element/.
"""
import csv, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = (("force_mae_meV_per_A", "Force  MAE"), ("energy_mae_meV_per_atom", "Energy MAE/Natoms"),
           ("force_rmse_meV_per_A", "Force  RMSE"), ("virial_mae_meV_per_atom", "Virial MAE/Natoms"))


def read_test(path):
    """The weighted averages over the validation set: the last value of each quantity in the file, in meV."""
    text = open(path, encoding="utf-8").read(); out = {}
    for key, label in METRICS:
        v = re.findall(re.escape(label) + r"\s*:\s*([-+0-9.eE]+)", text)
        out[key] = 1000.0 * float(v[-1]) if v else None
    return out


runs = list(csv.DictReader(open(f"{HERE}/runs.tsv"), delimiter="\t"))
rows = []
for r in runs:
    d = f"{HERE}/{r['folder']}"; t = f"{d}/test_ema.log" if os.path.exists(f"{d}/test_ema.log") else f"{d}/test.log"
    rows.append({**{k: r[k] for k in ("data_set", "model", "optimizer", "seed")}, **read_test(t)})
if "--write-csv" in sys.argv:
    with open(f"{HERE}/per_seed.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["data_set", "model", "optimizer", "seed"] + [k for k, _ in METRICS])
        for r in rows:
            w.writerow([r["data_set"], r["model"], r["optimizer"], r["seed"]] + ["" if r[k] is None else f"{r[k]:.2f}" for k, _ in METRICS])

print("##### 1. Validation errors: baseline -> PhysMuon (mean seed-paired change, seeds on which PhysMuon is lower)")
print("      forces in meV/A, energies and virials in meV per atom\n")
tasks = list(dict.fromkeys((r["data_set"], r["model"]) for r in rows))
for ds, model in tasks:
    q = [r for r in rows if (r["data_set"], r["model"]) == (ds, model)]
    base = next(o for o in ("Muon", "HybridMuon") if any(r["optimizer"] == o for r in q))
    get = lambda o, k: np.array([r[k] for r in sorted((r for r in q if r["optimizer"] == o), key=lambda r: int(r["seed"]))], float)
    cells = []
    for k, _ in METRICS:
        if q[0][k] is None: continue
        b, p = get(base, k), get("PhysMuon", k)
        cells.append(f"{k.split('_meV')[0].replace('_', ' ')} {b.mean():.1f} -> {p.mean():.1f} ({100 * np.mean(p / b - 1):+.1f}%, {int((p < b).sum())}/{len(b)})")
    print(f"  {ds} ({model}), baseline {base}: " + " | ".join(cells))
    for o in dict.fromkeys(r["optimizer"] for r in q):
        print(f"      force MAE per seed, {o:>14}: " + "  ".join(f"{v:.2f}" for v in get(o, "force_mae_meV_per_A")) + f"   mean {get(o, 'force_mae_meV_per_A').mean():.2f}")

print("\n##### 2. SPICE2 (large batch): force MAE per element on the validation set (meV/A, mean of 6 seeds)")
fq = json.load(open(f"{HERE}/per_element/training_set_elements.json")); tmap = fq["type_map"]
share = np.array(fq["atoms_by_elem"]) / np.sum(fq["atoms_by_elem"])
A = {a: json.load(open(f"{HERE}/per_element/{a}.json"))["models"] for a in ("muon", "physmuon")}
per = {a: np.array([1000 * np.array(m["force_abs_sum_by_elem"]) / np.maximum(3 * np.array(m["atoms_by_elem"]), 1) for m in A[a].values()]) for a in A}   # [seed, element]
print(f"  {'element':>8} {'share of training atoms':>24} {'Muon':>7} {'PhysMuon':>9} {'PhysMuon vs Muon':>17}")
rel = []
for k, el in enumerate(tmap):
    r = float(np.mean(per["physmuon"][:, k] / per["muon"][:, k] - 1)); rel.append(r)
    print(f"  {el:>8} {100 * share[k]:>23.2f}% {per['muon'][:, k].mean():>7.1f} {per['physmuon'][:, k].mean():>9.1f} {100 * r:>+16.1f}%")
tot = {a: np.array([m["force_mae_meV_A"] for m in A[a].values()]) for a in A}
print(f"  {'all':>8} {100.0:>23.2f}% {tot['muon'].mean():>7.2f} {tot['physmuon'].mean():>9.2f} {100 * np.mean(tot['physmuon'] / tot['muon'] - 1):>+16.1f}%")
big = [k for k in range(len(tmap)) if share[k] > 0.01]
print(f"  change over the {len(tmap)} elements: {100 * max(rel):+.1f}% to {100 * min(rel):+.1f}% | elements with more than 1% of the training atoms "
      f"({', '.join(tmap[k] for k in big)}; {100 * share[big].sum():.1f}% together): {100 * max(rel[k] for k in big):+.1f}% to {100 * min(rel[k] for k in big):+.1f}%")
