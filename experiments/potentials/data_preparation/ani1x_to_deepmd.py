"""ANI-1x (wB97X/6-31G(d); HDF5 grouped by the number of atoms) -> deepmd-kit mixed-type systems, non-periodic.

    python ani1x_to_deepmd.py ANI-1x-wB97X-631Gd.h5 DATA [max_frames_per_group]

Energies and forces are converted from Hartree and Hartree/A to eV and eV/A. Validation set: 2% of the molecules, a
molecule being one sequence of elements, so all conformers of a molecule are in the same set (chosen by a hash of the
sequence). Writes DATA/train, DATA/valid and DATA/SUMMARY.json.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import h5py

HA = 27.211386245988
Z2S = {1: "H", 6: "C", 7: "N", 8: "O"}
TYPE_MAP = ["H", "C", "N", "O"]
ZIDX = {1: 0, 6: 1, 7: 2, 8: 3}


def is_valid(spec):
    h = hashlib.sha1(np.asarray(spec, dtype=np.int64).tobytes()).digest()
    return int.from_bytes(h[:4], "little") % 100 < 2


def write(out: Path, nat, coord, force, energy, types):
    d = out / str(nat); s = d / "set.000"; s.mkdir(parents=True, exist_ok=True)
    (d / "type.raw").write_text("0\n" * nat)
    (d / "type_map.raw").write_text("\n".join(TYPE_MAP) + "\n")
    (d / "nopbc").write_text("")
    n = len(energy)
    np.save(s / "coord.npy", coord.reshape(n, -1).astype(np.float64))
    np.save(s / "force.npy", force.reshape(n, -1).astype(np.float64))
    np.save(s / "energy.npy", energy.astype(np.float64))
    np.save(s / "real_atom_types.npy", types.astype(np.int32))
    np.save(s / "box.npy", np.tile(np.eye(3).reshape(-1) * 100.0, (n, 1)))


def main(h5, out, cap=0):
    out = Path(out); f = h5py.File(h5, "r")
    unit = HA   # ANI-1x stores Hartree and Hartree/A (checked: CO -113.28, N2 -109.50, O2 -150.22; max |F| ~0.15)
    summ = {"source": str(h5), "energy_scale_to_eV": unit, "type_map": TYPE_MAP, "train": {}, "valid": {}, "dropped_nan": 0,
            "molecules": {"train": 0, "valid": 0}}
    for key in sorted(f.keys(), key=int):
        g = f[key]; nat = int(key)
        spec = g["species"][:]; E = g["energies"][:]; F = g["forces"][:]; R = g["coordinates"][:]
        if cap:
            spec, E, F, R = spec[:cap], E[:cap], F[:cap], R[:cap]
        ok = np.isfinite(E) & np.isfinite(F).reshape(len(E), -1).all(1) & np.isfinite(R).reshape(len(E), -1).all(1)
        summ["dropped_nan"] += int((~ok).sum())
        spec, E, F, R = spec[ok], E[ok], F[ok], R[ok]
        uniq, inv = np.unique(spec, axis=0, return_inverse=True)
        vmask_mol = np.array([is_valid(u) for u in uniq]); vmask = vmask_mol[inv.reshape(-1)]
        summ["molecules"]["valid"] += int(vmask_mol.sum()); summ["molecules"]["train"] += int((~vmask_mol).sum())
        types = np.vectorize(ZIDX.get)(spec)
        for split, m in (("train", ~vmask), ("valid", vmask)):
            if m.sum():
                write(out / split, nat, R[m], F[m] * unit, E[m] * unit, types[m])
                summ[split][nat] = int(m.sum())
        print(f"  natoms {nat}: {len(E)} frames, {len(uniq)} molecules, valid {int(vmask.sum())}", flush=True)
    summ["train_frames"] = sum(summ["train"].values()); summ["valid_frames"] = sum(summ["valid"].values())
    (out / "SUMMARY.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("energy_scale_to_eV", "train_frames", "valid_frames", "molecules", "dropped_nan")}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
