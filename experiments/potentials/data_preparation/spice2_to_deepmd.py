"""SPICE2-curated (extended XYZ) -> deepmd-kit mixed-type systems (plain numpy, streaming), non-periodic.

    python spice2_to_deepmd.py SPICE2-curated.zip DATA [max_train_frames] [max_valid_frames]

One system per number of atoms. Energies in eV, forces in eV/A, coordinates in A. The type map is the set of elements
in the data, ordered by atomic number. The training and validation sets are the two files of the archive
(SPICE2_curated_train.xyz and SPICE2_curated_val.xyz). Writes DATA/train, DATA/valid and DATA/SUMMARY.json.
"""
import io, json, re, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

Z = {s: i + 1 for i, s in enumerate(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe".split())}
E_RE = re.compile(r'(?:^|\s)energy=(-?[\d.]+(?:[eE][-+]?\d+)?)')
CT_RE = re.compile(r'config_type=(\S+)')


def frames(stream, limit):
    """yield (species list, numeric (n,6) array [pos, force], energy, config_type) from an extxyz text stream"""
    k = 0
    while limit <= 0 or k < limit:
        line = stream.readline()
        if not line:
            return
        if not line.strip():
            continue
        n = int(line)
        hdr = stream.readline()
        assert "Properties=species:S:1:pos:R:3:forces:R:3" in hdr, hdr[:200]
        e = float(E_RE.search(hdr).group(1))
        m = CT_RE.search(hdr)
        rows = [stream.readline() for _ in range(n)]
        spec = [r.split(None, 1)[0] for r in rows]
        num = np.array(" ".join(r.split(None, 1)[1] for r in rows).split(), dtype=np.float64).reshape(n, 6)
        yield spec, num, e, (m.group(1) if m else "")
        k += 1


def convert(zip_path, member, out, limit, index):
    buckets = defaultdict(lambda: {"coord": [], "force": [], "energy": [], "types": []})
    ctypes = Counter()
    proc = subprocess.Popen(["unzip", "-p", zip_path, member], stdout=subprocess.PIPE)
    stream = io.TextIOWrapper(proc.stdout, encoding="utf-8")
    n_frames = 0
    for spec, num, e, ct in frames(stream, limit):
        for s in spec:
            if s not in index:
                index[s] = len(index)
        b = buckets[len(spec)]
        b["coord"].append(num[:, :3].reshape(-1)); b["force"].append(num[:, 3:].reshape(-1))
        b["energy"].append(e); b["types"].append(np.array([index[s] for s in spec], dtype=np.int32))
        ctypes[ct] += 1; n_frames += 1
        if n_frames % 200000 == 0:
            print(f"  {member}: {n_frames} frames", flush=True)
    proc.stdout.close(); proc.wait()
    return buckets, ctypes, n_frames


def write(buckets, out: Path, remap, type_map):
    sizes = {}
    for nat, b in sorted(buckets.items()):
        d = out / str(nat); s = d / "set.000"; s.mkdir(parents=True, exist_ok=True)
        (d / "type.raw").write_text("0\n" * nat)
        (d / "type_map.raw").write_text("\n".join(type_map) + "\n")
        (d / "nopbc").write_text("")
        nf = len(b["energy"])
        np.save(s / "coord.npy", np.stack(b["coord"]))
        np.save(s / "force.npy", np.stack(b["force"]))
        np.save(s / "energy.npy", np.array(b["energy"], dtype=np.float64))
        np.save(s / "real_atom_types.npy", remap[np.stack(b["types"])].astype(np.int32))
        np.save(s / "box.npy", np.tile(np.eye(3).reshape(-1) * 100.0, (nf, 1)))
        sizes[nat] = nf
    return sizes


def main(zip_path, out, max_train=0, max_valid=0):
    out = Path(out); index = {}
    tr, ct_tr, n_tr = convert(zip_path, "SPICE2-curated/SPICE2_curated_train.xyz", out, max_train, index)
    va, ct_va, n_va = convert(zip_path, "SPICE2-curated/SPICE2_curated_val.xyz", out, max_valid, index)
    type_map = sorted(index, key=lambda s: Z[s])
    remap = np.zeros(len(index), dtype=np.int32)
    for s, i in index.items():
        remap[i] = type_map.index(s)
    summary = {"source": str(zip_path), "type_map": type_map,
               "train": {"frames": n_tr, "buckets": write(tr, out / "train", remap, type_map), "config_types": dict(ct_tr)},
               "valid": {"frames": n_va, "buckets": write(va, out / "valid", remap, type_map), "config_types": dict(ct_va)}}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: (v if k != "train" and k != "valid" else {"frames": v["frames"], "n_buckets": len(v["buckets"])})
                      for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 0, int(a[4]) if len(a) > 4 else 0)
