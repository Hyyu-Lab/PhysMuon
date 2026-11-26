"""DFT reference for the ANI-1x molecular-dynamics configurations: wB97x/6-31G(d) single points with PySCF, the level of
the ANI-1x data set (Cartesian d functions, as in the Gaussian default; density fitting).

    python dft_ani1x.py snapshots results/snapshots dft        # energies and forces of every saved configuration
    python dft_ani1x.py validate DATA/valid validation.json    # the same setup against the labels of the data set

`snapshots` writes one file per snapshot file: the configurations with E (eV), F (eV/A) and the convergence flag.
`validate` computes two random validation configurations for each of six molecule sizes (12 in all) and compares the
forces with the labels of the data set.
"""
import glob, json, os, sys, time
import numpy as np
from pyscf import gto, dft

HARTREE_EV, BOHR_A = 27.211386245988, 0.529177210903
ZNUM = {"H": 1, "C": 6, "N": 7, "O": 8}


def single_point(symbols, coords):
    nelec = sum(ZNUM[s] for s in symbols)
    mol = gto.M(atom=[(s, tuple(map(float, x))) for s, x in zip(symbols, coords)], basis="6-31g*", unit="Angstrom",
                charge=0, spin=nelec % 2, cart=True, verbose=0)
    mf = (dft.RKS(mol) if nelec % 2 == 0 else dft.UKS(mol)).density_fit(auxbasis="def2-universal-jkfit")
    mf.xc = "wb97x"; mf.grids.level = 4; mf.conv_tol = 1e-10; mf.max_cycle = 200
    e = mf.kernel()
    if not mf.converged:
        mf = mf.newton(); e = mf.kernel()
    g = mf.nuc_grad_method().kernel()
    return float(e) * HARTREE_EV, -np.asarray(g) * HARTREE_EV / BOHR_A, bool(mf.converged)


def snapshots(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for src in sorted(glob.glob(f"{src_dir}/ani1x_*_s0_m*.npz")):
        tag = os.path.basename(src)[:-4]
        if os.path.exists(f"{out_dir}/{tag}.npz"): continue
        z = np.load(src); tm = list(z["type_map"]); sym = [tm[j] for j in z["types"]]
        E, F, C = [], [], []
        for k, x in enumerate(z["coords"]):
            t0 = time.time(); e, f, conv = single_point(sym, x); E.append(e); F.append(f); C.append(conv)
            print(tag, k, z["labels"][k], "%.1fs" % (time.time() - t0), "converged" if conv else "NOT CONVERGED", flush=True)
        np.savez(f"{out_dir}/tmp_{tag}.npz", E=np.array(E), F=np.array(F), converged=np.array(C), labels=z["labels"], t_ps=z["t_ps"],
                 coords=z["coords"], types=z["types"], type_map=z["type_map"])
        os.replace(f"{out_dir}/tmp_{tag}.npz", f"{out_dir}/{tag}.npz")


def validate(valid, out_json):
    rng = np.random.default_rng(7); out = []
    for n in (10, 12, 15, 18, 21, 25):
        d = f"{valid}/{n}/set.000"; tm = open(f"{valid}/{n}/type_map.raw").read().split()
        X = np.load(f"{d}/coord.npy"); F = np.load(f"{d}/force.npy"); E = np.load(f"{d}/energy.npy").reshape(-1); T = np.load(f"{d}/real_atom_types.npy").astype(int)
        for i in rng.choice(len(X), 2, replace=False):
            sym = [tm[j] for j in T[i]]; t0 = time.time()
            e, f, conv = single_point(sym, X[i].reshape(n, 3))
            ref = F[i].reshape(n, 3)
            r = {"natoms": n, "frame": int(i), "converged": conv, "sec": round(time.time() - t0, 1), "E_pyscf_eV": e, "E_label": float(E[i]),
                 "force_MAE_vs_label_meV_A": float(np.abs(f - ref).mean() * 1000), "force_max_dev_meV_A": float(np.abs(f - ref).max() * 1000)}
            out.append(r); print(json.dumps(r), flush=True)
    de = np.array([r["E_pyscf_eV"] - r["E_label"] for r in out])
    print("energy offset pyscf-label: mean %.4f eV, spread (std) %.4f eV" % (de.mean(), de.std()))
    print("force MAE vs label over all frames: %.3f meV/A" % np.mean([r["force_MAE_vs_label_meV_A"] for r in out]))
    json.dump(out, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    {"snapshots": snapshots, "validate": validate}[sys.argv[1]](sys.argv[2], sys.argv[3])
