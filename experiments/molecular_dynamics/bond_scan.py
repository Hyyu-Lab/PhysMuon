"""Rigid bond-stretch scans with the 12 SPICE2 large-batch models (Muon and PhysMuon; 6 seeds each), on the
three bonds that broke most often at 1,500 K: Si-Cl of molecule 0, C-H of molecule 1 and Si-C of molecule 3.
Starting from the configuration of the molecular-dynamics test, the atom or the group on one side of the bond is moved
along the bond axis from -0.3 to +3.0 A in steps of 0.1 A; all other atoms stay where they are.

    python bond_scan.py --models MODELS --valid DATA/valid --out bond_scan

Writes bond_scan/geometries.npz (the scanned geometries, used by dft_spice2.py) and bond_scan/model_energies.npz
(DS: the stretches in A; "<scan>|<optimizer>|s<seed>": the energies in eV).
"""
import argparse, os
import numpy as np
import run_md

SCANS = [("m0_Si7-Cl1", 0, 7, 1), ("m1_C10-H22", 1, 10, 22), ("m3_Si17-C16", 3, 17, 16)]   # (name, molecule, fixed atom, moving atom)
DS = np.round(np.arange(-0.3, 3.0001, 0.1), 3)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--models", required=True); ap.add_argument("--valid", required=True); ap.add_argument("--out", default="bond_scan")
a = ap.parse_args()
from deepmd.infer import DeepPot
cfg = run_md.CFG["spice2"]; geo = {}
for name, mi, i, j in SCANS:
    n = cfg["sizes"][mi]; tmap = open(f"{a.valid}/{n}/type_map.raw").read().split()
    x0, t, frame = run_md.pick(a.valid, n, tmap); sym = [tmap[k] for k in t]
    rc = np.array([run_md.ELEM[s][1] for s in sym])
    b, _ = run_md.bonds(x0, rc); adj = {k: set() for k in range(n)}
    for p, c in b:
        if {p, c} != {i, j}: adj[p].add(c); adj[c].add(p)
    frag, st = set(), [j]
    while st:
        k = st.pop()
        if k in frag: continue
        frag.add(k); st += list(adj[k] - frag)
    assert i not in frag, name
    u = (x0[j] - x0[i]) / np.linalg.norm(x0[j] - x0[i]); r0 = float(np.linalg.norm(x0[j] - x0[i]))
    X = np.array([x0 + np.where(np.isin(np.arange(n), sorted(frag))[:, None], d * u, 0.0) for d in DS])
    geo[name] = dict(X=X, types=t, type_map=np.array(tmap), r=r0 + DS, frag=np.array(sorted(frag)), frame=frame)
    print(name, "frame", frame, "bond", sym[i], i, sym[j], j, "r0 %.3f A" % r0, "moving", run_md.formula([sym[k] for k in sorted(frag)]), flush=True)
os.makedirs(a.out, exist_ok=True)
np.savez(f"{a.out}/geometries.npz", **{f"{k}|{q}": v for k, g in geo.items() for q, v in g.items()})
E = {}
for opt in cfg["optimizers"]:
    for s in range(cfg["seeds"]):
        dp = DeepPot(run_md.model_path(a.models, "spice2", opt, s))
        for name, g in geo.items():
            e, _, _ = dp.eval(g["X"].reshape(len(g["X"]), -1), None, g["types"].tolist())
            E[f"{name}|{opt}|s{s}"] = np.asarray(e).reshape(-1)
        print("done", opt, s, flush=True)
np.savez(f"{a.out}/model_energies.npz", DS=DS, **E)
