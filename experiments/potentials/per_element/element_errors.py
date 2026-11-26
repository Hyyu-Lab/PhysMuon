"""Force error per element of the six SPICE2 large-batch models of one optimizer, on the whole validation set.

    python element_errors.py physmuon --models MODELS --valid DATA/valid        # muon | physmuon

MODELS holds the trained models as spice2_<optimizer>_s<seed>.pt (the model.ckpt-20000.pt of each run). Writes
<optimizer>.json: for every model the summed absolute force error and the number of atoms of each element, the force
MAE over all atoms and the energy MAE per atom. summarize.py turns these files into the table of the paper.
"""
import argparse, glob, json, time
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("optimizer", choices=("muon", "physmuon")); ap.add_argument("--models", required=True)
ap.add_argument("--valid", required=True); ap.add_argument("--out", default=None); ap.add_argument("--seeds", type=int, nargs="*", default=list(range(6)))
a = ap.parse_args()
from deepmd.infer import DeepPot
V = a.valid
tmap = open(f"{V}/18/type_map.raw").read().split(); ne = len(tmap)
dirs = sorted(glob.glob(f"{V}/*/set.000"), key=lambda d: int(d.split("/")[-2]))
out = {"optimizer": a.optimizer, "type_map": tmap, "models": {}}
for seed in a.seeds:
    t0 = time.time(); dp = DeepPot(f"{a.models}/spice2_{a.optimizer}_s{seed}.pt")
    assert list(dp.get_type_map()) == tmap
    sabs = np.zeros(ne); cnt = np.zeros(ne); se = 0.0; nfr = 0; tot_abs = 0.0; tot_comp = 0
    for d in dirs:
        X = np.load(f"{d}/coord.npy"); F = np.load(f"{d}/force.npy"); E = np.load(f"{d}/energy.npy").reshape(-1)
        T = np.load(f"{d}/real_atom_types.npy").astype(int); nf, n = T.shape
        e, f, _ = dp.eval(X.reshape(nf, -1), None, T, mixed_type=True)
        err = np.abs(np.asarray(f).reshape(nf, n, 3) - F.reshape(nf, n, 3)).sum(-1)      # summed over x, y, z per atom
        np.add.at(sabs, T.reshape(-1), err.reshape(-1)); np.add.at(cnt, T.reshape(-1), 1)
        tot_abs += err.sum(); tot_comp += nf * n * 3
        se += float((np.abs(np.asarray(e).reshape(-1) - E) / n).sum()); nfr += nf
    out["models"][f"seed{seed}"] = {"force_abs_sum_by_elem": sabs.tolist(), "atoms_by_elem": cnt.tolist(), "force_mae_meV_A": 1000 * tot_abs / tot_comp,
                                    "energy_mae_meV_atom": 1000 * se / nfr, "frames": nfr}
    print(f"seed {seed}: force MAE %.2f meV/A, energy MAE %.3f meV/atom, %d frames, %.0f s" % (1000 * tot_abs / tot_comp, 1000 * se / nfr, nfr, time.time() - t0), flush=True)
with open(a.out or f"{a.optimizer}.json", "w") as fh:
    json.dump(out, fh, indent=1); fh.write("\n")
