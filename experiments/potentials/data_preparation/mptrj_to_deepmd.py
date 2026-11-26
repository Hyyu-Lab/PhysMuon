"""MPtrj, step 1: the per-material extended-XYZ files -> one deepmd system per material and composition.

    python mptrj_to_deepmd.py EXTXYZ_DIR DEEPMD_DIR SHARD NSHARD [LIMIT]      # e.g. 8 shards in parallel: SHARD = 0..7

Each mp-XXXX.extxyz holds the frames of one material. Needs ase and dpdata.
"""
import os, sys, time, traceback

import numpy as np
from ase.io import read
import dpdata

SRC = sys.argv[1]
DST = sys.argv[2]
SHARD = int(sys.argv[3])
NSHARD = int(sys.argv[4])
LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else 0

files = sorted(f for f in os.listdir(SRC) if f.endswith(".extxyz"))
mine = files[SHARD::NSHARD]
if LIMIT:
    mine = mine[:LIMIT]
print(f"shard {SHARD}/{NSHARD}: {len(mine)} of {len(files)} files", flush=True)

os.makedirs(DST, exist_ok=True)
t0 = time.time()
nsys = nframe = nfail = 0

for i, fn in enumerate(mine):
    mid = fn[: -len(".extxyz")]
    out = os.path.join(DST, mid)
    if os.path.isdir(out):
        continue
    try:
        frames = read(os.path.join(SRC, fn), index=":", format="extxyz")
        # A single material file can hold frames at different supercell sizes
        # (e.g. Ca1Fe1O3 and Ca4Fe4O12). MultiSystems groups them by formula
        # instead of dropping the file.
        ms = dpdata.MultiSystems()
        nf = 0
        for a in frames:
            ms.append(dpdata.LabeledSystem(a, fmt="ase/structure"))
            nf += 1
        if nf == 0:
            nfail += 1
            continue
        ms.to_deepmd_npy(out)
        nsys += len(ms.systems)
        nframe += nf
    except Exception:
        nfail += 1
        if nfail <= 3:
            traceback.print_exc()
    if (i + 1) % 200 == 0:
        dt = time.time() - t0
        print(f"  {i+1}/{len(mine)} files | {nsys} sys | {nframe} frames | "
              f"{nfail} fail | {dt:.1f}s | {(i+1)/dt:.1f} files/s", flush=True)

dt = time.time() - t0
print(f"SHARD {SHARD} DONE: {nsys} systems, {nframe} frames, {nfail} failures, "
      f"{dt:.1f}s, {len(mine)/max(dt,1e-9):.1f} files/s", flush=True)
