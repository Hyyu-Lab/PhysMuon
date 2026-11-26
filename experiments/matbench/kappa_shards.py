"""Split the 103 structures of the thermal-conductivity test into N files, alternating over the space groups so that the
costly structures are spread evenly.

    python kappa_shards.py 2024-11-09-phononDB-PBE-103-structures.extxyz 4 shards
"""
import os
import sys

from ase.io import read, write
from moyopy import MoyoDataset
from moyopy.interface import MoyoAdapter

src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
atoms = read(src, format="extxyz", index=":")
spg = [MoyoDataset(MoyoAdapter.from_atoms(a), symprec=1e-5).number for a in atoms]
order = sorted(range(len(atoms)), key=lambda i: (spg[i], i))
os.makedirs(out, exist_ok=True)
for k in range(n):
    part = [atoms[i] for i in order[k::n]]
    write(f"{out}/shard-{k:02d}.extxyz", part, format="extxyz")
    print(f"shard {k}: {len(part)} structures, space groups {sorted(set(spg[i] for i in order[k::n]))}")
