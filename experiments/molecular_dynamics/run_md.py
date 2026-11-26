"""Molecular-dynamics test of DPA-3 models trained with Muon and with PhysMuon (ANI-1x and SPICE2, large batch).

One task is one model on one molecule: (data set, optimizer, seed, molecule index). The seven trajectories of a task are
advanced together, and all of them are evaluated in one batched call of the model.

  routine test             trajectory 0: Langevin at 300 K for 2 ps, then no thermostat for 40 ps (energy conservation);
                           trajectories 1-6: Langevin at two temperatures, three trajectories each, for 42 ps
                           (ANI-1x: 1,000 and 1,500 K; SPICE2: 600 and 1,000 K)
  stress test (--stress T) all seven trajectories Langevin at T for 42 ps

Integration: time step 0.5 fs, BAOAB Langevin with friction 0.01 /fs. Every model sees the same initial velocities and
the same random forces for a given molecule, so two models differ only through their forces.

A trajectory is unstable when the molecule falls into two or more pieces (two atoms are bonded when they are closer than
1.5 times the sum of their covalent radii), when two atoms come closer than 0.6 A, or when the energy is not finite.
It is checked every 10 fs. With --persist N a split counts only if it lasts N consecutive checks; a split that heals
earlier is counted in "transient_splits". The routine test of the paper used N = 1 and the stress tests N = 10 (100 fs),
because at these temperatures a bond can pass the bonding threshold for a moment without breaking. An unstable
trajectory is frozen, and its record holds the time, the fragments and the bonds that broke.

The molecule of an index is taken from the validation set of the data set: among the configurations with that number
of atoms that are one molecule, contain C and H and have no bond longer than 1.15 times the sum of the covalent radii,
the one with the smallest root-mean-square force.

Units: A, fs, eV, amu.

    python run_md.py spice2 physmuon 0 3 --stress 1500 --models MODELS --valid DATA/valid --out results

MODELS holds the trained models as <data set>_<optimizer>_s<seed>.pt (the model.ckpt-20000.pt of each training run);
DATA/valid is the validation set in deepmd's mixed-type format. Writes results/<task>.json, for the routine test the
energies of trajectory 0 (results/nve_<task>.npy: time in ps, potential and kinetic energy in eV), and for the seed-0
ANI-1x tasks of the routine test the configurations used for the DFT comparison (results/snapshots/<task>.npz).
"""
import argparse, json, os, time
import numpy as np

ELEM = {"H": (1.008, 0.31), "B": (10.81, 0.84), "C": (12.011, 0.76), "N": (14.007, 0.71), "O": (15.999, 0.66), "F": (18.998, 0.57),
        "Si": (28.085, 1.11), "P": (30.974, 1.07), "S": (32.06, 1.05), "Cl": (35.45, 1.02), "Br": (79.904, 1.20), "I": (126.904, 1.39)}   # mass, covalent radius
# Number of atoms of each molecule index, and the two temperatures of the routine test. Index 7 of ANI-1x is not used.
CFG = {
    "ani1x": {"sizes": {0: 10, 1: 12, 2: 13, 3: 15, 4: 16, 5: 17, 6: 18, 8: 21, 9: 23, 10: 25, 11: 27, 12: 22}, "hot": [1000.0] * 3 + [1500.0] * 3,
              "optimizers": ("muon", "physmuon"), "seeds": 4},
    "spice2": {"sizes": {0: 18, 1: 24, 2: 30, 3: 36, 4: 42, 5: 48, 6: 54, 7: 60}, "hot": [600.0] * 3 + [1000.0] * 3,
               "optimizers": ("muon", "physmuon"), "seeds": 6},
}
ACC, KE_EV, KB, DT, GAMMA = 0.0096485, 103.6427, 8.617333e-5, 0.5, 0.01
CHECK = 20                                                              # steps between two checks (10 fs)


def components(x, rc, scale):
    """Number of bonded fragments, the fragment of every atom, and the smallest distance between two atoms."""
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    adj = (d < scale * (rc[:, None] + rc[None])) & ~np.eye(len(rc), dtype=bool)
    lab = -np.ones(len(rc), int); n = 0
    for s in range(len(rc)):
        if lab[s] >= 0: continue
        st = [s]
        while st:
            i = st.pop()
            if lab[i] >= 0: continue
            lab[i] = n; st += list(np.where(adj[i] & (lab < 0))[0])
        n += 1
    return n, lab, float(d[np.triu_indices(len(rc), 1)].min())


def bonds(x, rc, scale=1.25):
    d = np.linalg.norm(x[:, None] - x[None], axis=-1); s = rc[:, None] + rc[None]
    adj = np.triu((d < scale * s), 1)
    return [(int(i), int(j)) for i, j in zip(*np.where(adj))], d / s


def formula(sym):
    c = {e: sym.count(e) for e in set(sym)}
    hill = [k for k in ("C", "H") if "C" in c and k in c]                 # Hill order; a C fragment may have no H
    keys = hill + sorted(k for k in c if k not in hill)
    return "".join(k + (str(c[k]) if c[k] > 1 else "") for k in keys)


def pick(valid, natoms, tmap):
    """Starting configuration with `natoms` atoms: coordinates, atom types and the index of the frame in the validation set."""
    d = f"{valid}/{natoms}/set.000"
    X = np.load(f"{d}/coord.npy"); F = np.load(f"{d}/force.npy"); T = np.load(f"{d}/real_atom_types.npy").astype(int)
    rms = np.sqrt((F.reshape(len(F), -1) ** 2).mean(1))
    for i in np.argsort(rms):
        x, t = X[i].reshape(natoms, 3), T[i]
        sym = [tmap[j] for j in t]; rc = np.array([ELEM[s][1] for s in sym])
        if "C" in sym and "H" in sym and components(x, rc, 1.25)[0] == 1:
            b, ratio = bonds(x, rc)
            if max(ratio[i_, j_] for i_, j_ in b) <= 1.15:
                return x, t, int(i)
    raise RuntimeError(f"no unstrained single molecule with {natoms} atoms")


def model_path(models, ds, opt, seed):
    return f"{models}/{ds}_{opt}_s{seed}.pt"


def main(ds, opt, seed, mi, models, valid, out, stress=0.0, persist=None, smoke=False):
    from deepmd.infer import DeepPot
    cfg = CFG[ds]; natoms = cfg["sizes"][mi]
    persist = (10 if stress else 1) if persist is None else persist       # checks (x 10 fs) a split must last to count
    neq, nnve = (200, 1000) if smoke else (4000, 80000)
    ntot = neq + nnve
    tmap = open(f"{valid}/{natoms}/type_map.raw").read().split()
    dp = DeepPot(model_path(models, ds, opt, seed))
    assert list(dp.get_type_map()) == tmap, (dp.get_type_map(), tmap)
    x0, t, frame = pick(valid, natoms, tmap); sym = [tmap[j] for j in t]
    m = np.array([ELEM[s][0] for s in sym])[None, :, None]; rc = np.array([ELEM[s][1] for s in sym])
    bonds0, _ = bonds(x0, rc)
    temps = np.array([stress] * 7) if stress else np.array([300.0] + cfg["hot"]); nrep = len(temps); Tset = temps[:, None, None]
    rng = np.random.default_rng(1000 + 17 * mi)                         # identical initial velocities / noise for every model
    x = np.repeat(x0[None], nrep, 0)
    v = rng.normal(size=x.shape) * np.sqrt(KB * Tset / m / KE_EV); v -= (m * v).sum(1, keepdims=True) / m.sum()
    noise = np.random.default_rng(5000 + 17 * mi)
    c1 = np.exp(-GAMMA * DT); c2 = np.sqrt(1 - c1 ** 2)
    calls, tcall = 0, 0.0

    def ef(x):
        nonlocal calls, tcall
        t0 = time.time(); e, f, _ = dp.eval(x.reshape(nrep, -1), None, t.tolist()); tcall += time.time() - t0; calls += 1
        return np.asarray(e).reshape(nrep), np.asarray(f).reshape(nrep, natoms, 3)

    e, f = ef(x)
    alive = np.ones(nrep, bool); xlast = x.copy(); fail = [None] * nrep
    split = [None] * nrep; transient = [0] * nrep; longest = [0] * nrep     # open split (first check, record), healed splits, longest healed (checks)

    def record(xr, lab, nc, dmin, k):
        dr = np.linalg.norm(xr[:, None] - xr[None], axis=-1)
        return {"t_ps": round(k * DT / 1000.0, 3), "fragments": int(nc), "dmin": round(dmin, 3), "finite": True,
                "fragment_formulas": sorted((formula([sym[i] for i in range(natoms) if lab[i] == c]) for c in range(nc)), key=len),
                "broken_bonds": [f"{sym[i]}{i}-{sym[j]}{j} {dr[i, j]:.2f}A" for i, j in bonds0 if lab[i] != lab[j]]}

    nve, snaps = [], []
    snap_steps = set() if stress else {neq + k * (nnve // 5) for k in range(1, 6)}   # 8, 16, 24, 32, 40 ps after equilibration
    k_end = ntot
    for k in range(1, ntot + 1):
        lang = np.ones((nrep, 1, 1), bool); lang[0] = bool(stress) or k <= neq   # trajectory 0 runs without thermostat after equilibration
        v += 0.5 * DT * f / m * ACC
        x += 0.5 * DT * v
        v = np.where(lang, c1 * v + c2 * np.sqrt(KB * Tset / m / KE_EV) * noise.normal(size=x.shape), v)
        x += 0.5 * DT * v
        dead = ~alive; x[dead] = xlast[dead]; v[dead] = 0.0
        e, f = ef(x); f[dead] = 0.0
        v += 0.5 * DT * f / m * ACC
        v[dead] = 0.0
        if k % CHECK == 0:
            for r in range(nrep):
                if not alive[r]: continue
                if not (np.isfinite(x[r]).all() and np.isfinite(e[r])):
                    fail[r] = {"t_ps": round(k * DT / 1000.0, 3), "finite": False, "during_split": split[r] is not None}
                    alive[r] = False; x[r] = xlast[r]; v[r] = 0.0; continue
                nc, lab, dmin = components(x[r], rc, 1.5)
                if dmin < 0.6:
                    fail[r] = record(x[r], lab, nc, dmin, k); fail[r]["collision"] = True
                    alive[r] = False; x[r] = xlast[r]; v[r] = 0.0
                elif nc > 1:
                    if split[r] is None: split[r] = (k, record(x[r], lab, nc, dmin, k))
                    if (k - split[r][0]) // CHECK + 1 >= persist:
                        fail[r] = split[r][1]; fail[r]["confirmed_ps"] = round(k * DT / 1000.0, 3)
                        fail[r]["broken_bonds_confirmed"] = record(x[r], lab, nc, dmin, k)["broken_bonds"]
                        alive[r] = False; x[r] = xlast[r]; v[r] = 0.0; split[r] = None
                else:
                    if split[r] is not None:
                        transient[r] += 1; longest[r] = max(longest[r], (k - split[r][0]) // CHECK); split[r] = None
                    xlast[r] = x[r]
            if not stress and k > neq and alive[0]:
                ke0 = 0.5 * float((m[0] * v[0] ** 2).sum()) * KE_EV
                nve.append(((k - neq) * DT / 1000.0, float(e[0]), ke0))
            if not alive.any():
                k_end = k; break
        if seed == 0 and ds == "ani1x" and k in snap_steps:
            for r, lab_ in ((0, "300K_NVE"), (1, f"{temps[1]:.0f}K"), (4, f"{temps[4]:.0f}K")):
                if alive[r]: snaps.append((lab_, round(k * DT / 1000.0, 2), x[r].copy()))
    nve = np.array(nve)
    res = {"dataset": ds, "opt": opt, "seed": seed, "mol": mi, "natoms": natoms, "frame": frame,
           "formula": {s: sym.count(s) for s in dict.fromkeys(sym)}, "temps": temps.tolist(), "stress_T": stress or None,
           "replica_failures": fail, "transient_splits": transient, "longest_transient_checks": longest, "persist_checks": persist,
           "open_split_at_end_ps": [round(sp[0] * DT / 1000.0, 3) if sp else None for sp in split],
           "ms_per_call": round(1000.0 * tcall / calls, 2), "ps_per_replica": ntot * DT / 1000.0, "stopped_at_ps": round(k_end * DT / 1000.0, 3)}
    if len(nve) > 10:
        et = (nve[:, 1] + nve[:, 2] - nve[0, 1] - nve[0, 2]) / natoms * 1000.0
        res.update({"nve_ps": float(nve[-1, 0]), "nve_drift_meV_atom_ps": float(np.polyfit(nve[:, 0], et, 1)[0]),
                    "nve_std_meV_atom": float(et.std()), "nve_maxdev_meV_atom": float(np.abs(et).max()),
                    "nve_T_mean": float(2 * nve[:, 2].mean() / ((3 * natoms - 6) * KB))})
    tag = f"{ds}_{opt}_s{seed}_m{mi}" + (f"_T{stress:.0f}" if stress else "")
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/{tag}.json", "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1); fh.write("\n")
    if len(nve): np.save(f"{out}/nve_{tag}.npy", nve.astype(np.float32))
    if snaps:
        os.makedirs(f"{out}/snapshots", exist_ok=True)
        np.savez(f"{out}/snapshots/{tag}.npz", labels=np.array([s[0] for s in snaps]), t_ps=np.array([s[1] for s in snaps]),
                 coords=np.array([s[2] for s in snaps]), types=t, type_map=np.array(tmap), x0=x0)
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", choices=sorted(CFG)); ap.add_argument("optimizer"); ap.add_argument("seed", type=int); ap.add_argument("molecule", type=int)
    ap.add_argument("--models", required=True, help="folder with <data set>_<optimizer>_s<seed>.pt")
    ap.add_argument("--valid", required=True, help="validation set of the data set (deepmd mixed-type format)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--stress", type=float, default=0.0, help="temperature of the stress test in K (default: routine test)")
    ap.add_argument("--persist", type=int, default=None, help="checks a split must last to count (default: 1, or 10 with --stress)")
    ap.add_argument("--smoke", action="store_true", help="0.6 ps instead of 42 ps, to check the installation")
    a = ap.parse_args()
    main(a.dataset, a.optimizer, a.seed, a.molecule, a.models, a.valid, a.out, a.stress, a.persist, a.smoke)
