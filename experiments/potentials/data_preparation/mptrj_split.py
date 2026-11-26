"""MPtrj, step 2: the systems of step 1 -> deepmd-kit mixed-type systems (one per number of atoms), with a validation set.

    python mptrj_split.py DEEPMD_DIR DATA MODEL_PARAMS.json [LIMIT]

Validation set: 1% of the materials (all frames of a material are in the same set), drawn with random seed 0 from the
sorted list of material ids; the 1,459 ids are listed in mptrj_validation_ids.txt. The type map is taken from the
model parameters (e.g. integrations/deepmd/dpa4/pro_model_params.json). Writes DATA/train, DATA/valid and
DATA/SUMMARY.json. Needs dpdata.
"""
import os, sys, json, time, random
import numpy as np
import dpdata

SRC, DST = sys.argv[1], sys.argv[2]
TYPE_MAP = json.load(open(sys.argv[3]))["type_map"]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 0

mpids = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))  # includes the few ids that do not start with mp-
if LIMIT:
    mpids = mpids[:LIMIT]
rng = random.Random(0)
valid_ids = set(rng.sample(mpids, max(1, len(mpids) // 100)))

train = dpdata.MultiSystems(type_map=TYPE_MAP)
valid = dpdata.MultiSystems(type_map=TYPE_MAP)
t0 = time.time(); nsys = nfr = nfail = 0; fails = []
for i, mid in enumerate(mpids):
    for f in sorted(os.listdir(os.path.join(SRC, mid))):
        p = os.path.join(SRC, mid, f)
        if not os.path.isdir(p):
            continue
        try:
            s = dpdata.LabeledSystem(p, fmt="deepmd/npy", type_map=TYPE_MAP)
        except Exception as e:
            nfail += 1; fails.append((p, repr(e)[:200])); continue
        (valid if mid in valid_ids else train).append(s)
        nsys += 1; nfr += s.get_nframes()
    if (i + 1) % 5000 == 0:
        print(f"{i+1}/{len(mpids)} mpids | {nsys} sys | {nfr} frames | {nfail} fail | {time.time()-t0:.0f}s", flush=True)
print(f"loaded: {nsys} systems, {nfr} frames, {nfail} failures, {time.time()-t0:.0f}s", flush=True)


def dump(ms, out):
    ms.to_deepmd_npy_mixed(out, set_size=200000)
    buckets = sorted(os.listdir(out))
    fr = 0
    for b in buckets:
        for st in sorted(os.listdir(os.path.join(out, b))):
            if st.startswith("set."):
                fr += np.load(os.path.join(out, b, st, "energy.npy")).shape[0]
    return {"buckets": len(buckets), "frames": fr, "bucket_names": buckets}


os.makedirs(DST, exist_ok=True)
tr = dump(train, os.path.join(DST, "train")); print("train:", tr["buckets"], "buckets", tr["frames"], "frames", flush=True)
va = dump(valid, os.path.join(DST, "valid")); print("valid:", va["buckets"], "buckets", va["frames"], "frames", flush=True)
summary = {"source": SRC, "limit": LIMIT, "mpids": len(mpids), "valid_mpids": len(valid_ids), "valid_seed": 0,
           "systems_loaded": nsys, "frames_loaded": nfr, "failures": nfail, "fail_list": fails[:50],
           "train": tr, "valid": va, "type_map_len": len(TYPE_MAP), "seconds": round(time.time() - t0)}
json.dump(summary, open(os.path.join(DST, "SUMMARY.json"), "w"), indent=1)
print("SUMMARY written", flush=True)
