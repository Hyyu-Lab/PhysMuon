"""DFT reference for the bond-stretch scans, at the level of the SPICE2 data set: wB97M-D3(BJ)/def2-TZVPPD with PySCF
(density fitting).

    python dft_spice2.py validate DATA/valid bond_scan/dft_validation.json
    python dft_spice2.py restricted   m0_Si7-Cl1 bond_scan [d ...]
    python dft_spice2.py unrestricted m0_Si7-Cl1 bond_scan [d ...]

validate       energies and forces of the three starting configurations against the labels of the data set
restricted     restricted Kohn-Sham energies along one scan, as in SPICE2 -> <folder>/dft_restricted_<scan>.json
unrestricted   each point is solved with restricted Kohn-Sham and then spin-unrestricted, started from a
               broken-symmetry guess (HOMO and LUMO mixed by +-45 degrees in the alpha and beta spaces) and followed
               through internal instabilities -> <folder>/dft_unrestricted_<scan>.json (E_RKS, E_UKS, <S^2>)

The geometries are read from <folder>/geometries.npz (written by bond_scan.py); d ... selects stretches in A, the
default is every point of the scan. A restricted calculation cannot describe the two radicals of a homolytically broken
bond, and its energy keeps rising with the stretch; the reference of the paper is the lower of the restricted and the
unrestricted energy at each point.
"""
import json, sys, time
import numpy as np
from pyscf import gto, dft

HARTREE_EV, BOHR_A = 27.211386245988, 0.529177210903
ZNUM = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}


# ---------------------------------------------------------------- restricted Kohn-Sham, as in SPICE2
def make_mf(symbols, coords, charge=0):
    nel = sum(ZNUM[s] for s in symbols) - charge
    mol = gto.M(atom=[(s, tuple(map(float, x))) for s, x in zip(symbols, coords)], basis="def2-tzvppd", unit="Angstrom",
                charge=charge, spin=nel % 2, verbose=0, max_memory=60000)
    try:
        mf = dft.RKS(mol, xc="wb97m-d3bj")
    except Exception as exc:
        print("xc string not accepted (%s); using wB97M-V without VV10 plus D3(BJ)" % exc, flush=True)
        mf = dft.RKS(mol, xc="wb97m-v"); mf.nlc = False; mf.disp = "d3bj"
    mf = mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.grids.level = 3; mf.conv_tol = 1e-9; mf.max_cycle = 200
    return mf


def energy(symbols, coords, grad=False):
    mf = make_mf(symbols, coords); e = mf.kernel()
    if not mf.converged:
        mf = mf.newton(); e = mf.kernel()
    out = {"E_eV": float(e) * HARTREE_EV, "converged": bool(mf.converged), "xc": str(mf.xc), "nlc": str(getattr(mf, "nlc", "")), "disp": str(getattr(mf, "disp", ""))}
    if grad:
        out["F_eVA"] = (-np.asarray(mf.nuc_grad_method().kernel()) * HARTREE_EV / BOHR_A).tolist()
    return out


# ---------------------------------------------------------------- spin-unrestricted, broken-symmetry start
def make_mol(symbols, coords):
    nel = sum(ZNUM[s] for s in symbols)
    return gto.M(atom=[(s, tuple(map(float, x))) for s, x in zip(symbols, coords)], basis="def2-tzvppd", unit="Angstrom",
                 charge=0, spin=nel % 2, verbose=0, max_memory=80000)


def setup(mf):
    mf = mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.grids.level = 3; mf.conv_tol = 1e-9; mf.max_cycle = 300
    return mf


def run(mf, dm0=None):
    e = mf.kernel(dm0)
    if not mf.converged:                      # second-order solver from the last orbitals (its kernel takes mo, occ)
        mo, occ = mf.mo_coeff, mf.mo_occ
        mf = mf.newton(); e = mf.kernel(mo, occ)
    return mf, float(e)


def point(symbols, coords):
    mol = make_mol(symbols, coords)
    rks, e_r = run(setup(dft.RKS(mol, xc="wb97m-d3bj")))
    mo, occ = rks.mo_coeff, rks.mo_occ
    homo = int(np.where(occ > 0)[0][-1]); lumo = homo + 1
    ca, cb = mo.copy(), mo.copy()
    c = np.sqrt(0.5)
    ca[:, homo] = c * (mo[:, homo] + mo[:, lumo]); cb[:, homo] = c * (mo[:, homo] - mo[:, lumo])
    oa = (occ > 0).astype(float)
    uks = setup(dft.UKS(mol, xc="wb97m-d3bj"))
    uks, e_u = run(uks, uks.make_rdm1((ca, cb), (oa, oa)))
    n_stab = 0
    for _ in range(4):                        # follow internal instabilities to the lowest UKS solution
        try:
            mo_new, _, stable, _ = uks.stability(return_status=True)
        except Exception as exc:              # older PySCF: no return_status, or no stability for DF objects
            print("stability check skipped:", exc, flush=True); break
        if stable:
            break
        n_stab += 1
        uks, e_u = run(uks, uks.make_rdm1(mo_new, uks.mo_occ))
    s2 = float(uks.spin_square()[0])
    return {"E_RKS_eV": e_r * HARTREE_EV, "E_UKS_eV": e_u * HARTREE_EV, "S2_UKS": s2, "conv_RKS": bool(rks.converged),
            "conv_UKS": bool(uks.converged), "stability_reruns": n_stab}


# ---------------------------------------------------------------- command line
def validate(valid, out_json):
    import run_md
    cfg = run_md.CFG["spice2"]; res = []
    for mi in (0, 1, 3):
        n = cfg["sizes"][mi]; tmap = open(f"{valid}/{n}/type_map.raw").read().split()
        x0, t, frame = run_md.pick(valid, n, tmap); sym = [tmap[k] for k in t]
        d = f"{valid}/{n}/set.000"
        Fl = np.load(f"{d}/force.npy")[frame].reshape(n, 3); El = float(np.load(f"{d}/energy.npy").reshape(-1)[frame])
        t0 = time.time(); o = energy(sym, x0, grad=True); F = np.array(o.pop("F_eVA"))
        r = {"mol": mi, "frame": frame, "natoms": n, "sec": round(time.time() - t0), "E_pyscf_eV": o["E_eV"], "E_label_eV": El,
             "force_MAE_vs_label_meV_A": float(np.abs(F - Fl).mean() * 1000), "force_max_dev_meV_A": float(np.abs(F - Fl).max() * 1000), **o}
        res.append(r); print(json.dumps(r), flush=True)
    json.dump(res, open(out_json, "w"), indent=1)


def scan(kind, name, folder, want):
    g = np.load(f"{folder}/geometries.npz"); X = g[f"{name}|X"]; r = g[f"{name}|r"]; tm = list(g[f"{name}|type_map"]); sym = [tm[k] for k in g[f"{name}|types"]]
    d = np.round(r - r[3], 3); idx = [int(np.argmin(np.abs(d - w))) for w in want] if want else list(range(len(r)))
    out = []
    for k in idx:
        t0 = time.time(); o = energy(sym, X[k]) if kind == "restricted" else point(sym, X[k])
        o.update({"d_A": float(d[k]), "r_A": float(r[k]), "sec": round(time.time() - t0)})
        out.append(o); print(name, json.dumps(o), flush=True)
        json.dump(out, open(f"{folder}/dft_{kind}_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate(sys.argv[2], sys.argv[3])
    else:
        assert sys.argv[1] in ("restricted", "unrestricted"), sys.argv[1]
        scan(sys.argv[1], sys.argv[2], sys.argv[3], [float(v) for v in sys.argv[4:]])
