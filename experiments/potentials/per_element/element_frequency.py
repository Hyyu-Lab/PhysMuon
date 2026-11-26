"""How often each element appears in the SPICE2 training set: atoms of the element and frames that contain it.

    python element_frequency.py DATA/train training_set_elements.json
"""
import glob, json, sys
import numpy as np

T0, out_path = sys.argv[1], sys.argv[2]
tmap = open(f"{T0}/18/type_map.raw").read().split(); ne = len(tmap)
atoms = np.zeros(ne); frames = np.zeros(ne); nfr = 0
for f in glob.glob(f"{T0}/*/set.*/real_atom_types.npy"):
    T = np.load(f).astype(int); nfr += len(T)
    np.add.at(atoms, T.reshape(-1), 1)
    pres = np.zeros((len(T), ne), bool); rows = np.repeat(np.arange(len(T)), T.shape[1]); pres[rows, T.reshape(-1)] = True
    frames += pres.sum(0)
out = {"type_map": tmap, "frames_total": nfr, "atoms_by_elem": atoms.tolist(), "frames_with_elem": frames.tolist()}
json.dump(out, open(out_path, "w"), indent=1)
for k, e in enumerate(tmap): print(f"{e:>2}: share of atoms {atoms[k] / atoms.sum():.5f}, frames with it {frames[k] / nfr:.4f}")
