#!/usr/bin/env python
"""Relax the WBM test set with a frozen model, with the settings of the DPA-4.0.1 entries of Matbench Discovery:
FIRE with FrechetCellFilter, fmax 0.02 eV/A, at most 500 steps (as relax_atoms of matbench_discovery/discovery.py).
Energies are the raw totals of the model; cells, positions and atomic numbers of the relaxed structures are stored.

    python relax_wbm.py --model model.pt2 --wbm-zip 2024-08-04-wbm-initial-atoms.extxyz.zip \
        --wbm-summary 2023-12-13-wbm-summary.csv.gz --out-dir runs/TAG --n-shards 128 --shard-index 0 --mod 1

The structures are split into --n-shards shards with about the same number of atoms; run every shard index. A run can
be resumed. --mod 1 is the whole set of 256,963 structures; --mod N keeps the fixed subset md5(material_id) % N == 0.
"""
import argparse, hashlib, io, json, os, time, zipfile
import numpy as np, pandas as pd
from ase.io import read
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

FMAX, MAX_STEPS = 0.02, 500


def in_subset(mid, mod):
    return int(hashlib.md5(mid.encode()).hexdigest(), 16) % mod == 0


def partition(sizes, n_shards):
    shards, load = [[] for _ in range(n_shards)], [0] * n_shards
    for mid, n in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        k = min(range(n_shards), key=lambda i: (load[i], i))
        shards[k].append(mid); load[k] += n
    return shards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--wbm-zip", required=True)
    p.add_argument("--wbm-summary", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--mod", type=int, default=25)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    from deepmd.calculator import DP

    df = pd.read_csv(a.wbm_summary, usecols=["material_id", "n_sites"])
    df = df[df.material_id.map(lambda m: in_subset(m, a.mod))]
    ids = partition(dict(zip(df.material_id, df.n_sites)), a.n_shards)[a.shard_index]
    if a.limit:
        ids = ids[-a.limit:]                     # smallest cells of this shard for the pilot
    os.makedirs(a.out_dir, exist_ok=True)
    out = f"{a.out_dir}/shard-{a.shard_index:03d}-of-{a.n_shards:03d}.jsonl"
    done = set()
    if os.path.isfile(out):
        for line in open(out):
            try: done.add(json.loads(line)["material_id"])
            except (json.JSONDecodeError, KeyError): pass
    calc, t0, n_new = DP(a.model), time.perf_counter(), 0
    with zipfile.ZipFile(a.wbm_zip) as zf, open(out, "a") as fout:
        for mid in ids:
            if mid in done: continue
            rec = {"material_id": mid}
            for attempt in range(4):   # a CUDA out-of-memory is usually transient (8 processes share the GPU): wait and retry
                try:
                    atoms = read(io.StringIO(zf.read(f"{mid}.extxyz").decode()), format="extxyz")
                    atoms.calc = calc
                    ts = time.perf_counter()
                    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
                    rec["converged"] = bool(opt.run(fmax=FMAX, steps=MAX_STEPS))
                    rec["n_steps"] = int(opt.get_number_of_steps())
                    energy = float(atoms.get_potential_energy())
                    if not np.isfinite(energy): raise ValueError(f"non-finite energy {energy}")
                    rec["energy"] = energy
                    rec["cell"] = atoms.cell.array.tolist(); rec["positions"] = atoms.positions.tolist()
                    rec["numbers"] = atoms.numbers.tolist(); rec["seconds"] = round(time.perf_counter() - ts, 3)
                    rec.pop("error", None)
                    break
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    if "OutOfMemory" not in type(exc).__name__ or attempt == 3: break
                    time.sleep(30 * (attempt + 1))
            fout.write(json.dumps(rec) + "\n"); fout.flush(); n_new += 1
    print(f"shard {a.shard_index}: {n_new} new structures in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
