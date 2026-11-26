"""Energies and forces of the eight ANI-1x models (Muon and PhysMuon, seeds 0-3) on the configurations saved by the
routine test of run_md.py (the seed-0 tasks write them to <results>/snapshots).

    python evaluate_snapshots.py --models MODELS --snapshots results/snapshots --out model_predictions

Writes one file per snapshot file, with E_<optimizer>_s<seed> (eV) and F_<optimizer>_s<seed> (eV/A).
"""
import argparse, glob, os
import numpy as np
from run_md import model_path

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--models", required=True); ap.add_argument("--snapshots", required=True); ap.add_argument("--out", default="model_predictions")
a = ap.parse_args()
from deepmd.infer import DeepPot
models = {f"{o}_s{s}": DeepPot(model_path(a.models, "ani1x", o, s)) for o in ("muon", "physmuon") for s in range(4)}
os.makedirs(a.out, exist_ok=True)
for src in sorted(glob.glob(f"{a.snapshots}/ani1x_*_s0_m*.npz")):
    tag = os.path.basename(src)[:-4]; z = np.load(src); X = z["coords"]; n = X.shape[1]
    out = {}
    for k, dp in models.items():
        e, f, _ = dp.eval(X.reshape(len(X), -1), None, z["types"].tolist())
        out[f"E_{k}"] = np.asarray(e).reshape(-1); out[f"F_{k}"] = np.asarray(f).reshape(len(X), n, 3)
    np.savez(f"{a.out}/{tag}.npz", **out); print("evaluated", tag, len(X), flush=True)
