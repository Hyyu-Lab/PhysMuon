"""Merge the conductivity.json.gz files of the shards and compute kappa_SRME with the metric functions of
matbench_discovery, against the PBE reference without non-analytical correction. A material without a result counts
as a failure (SRME 2).

    python kappa_metrics.py 2024-11-09-kappas-phononDB-PBE-noNAC.json.gz TAG=kappa_runs/TAG
"""
import glob
import sys

import numpy as np
import pandas as pd
from pymatviz.enums import Key

from matbench_discovery.metrics.phonons import calc_kappa_metrics_from_dfs
from matbench_discovery.phonons import read_kappa_json

MID = str(Key.mat_id)          # pandas 3 does not match StrEnum members against string column labels
ref = read_kappa_json(sys.argv[1])              # already indexed by material_id
rows = []
for arg in sys.argv[2:]:
    tag, d = arg.split("=", 1)
    parts = sorted(glob.glob(f"{d}/shard-*/conductivity.json.gz"))
    df = pd.concat([read_kappa_json(p) for p in parts])
    missing = sorted(set(ref.index) - set(df.index))
    out = calc_kappa_metrics_from_dfs(df, ref)
    srme = out[str(Key.srme)].reindex(ref.index).fillna(2.0)        # missing materials count as failures (SRME 2)
    sre = out[str(Key.sre)].reindex(ref.index)
    imag = df[str(Key.has_imag_ph_modes)].fillna(False).astype(bool) if str(Key.has_imag_ph_modes) in df else pd.Series(dtype=bool)
    broken = df["broken_symmetry"].fillna(False).astype(bool) if "broken_symmetry" in df else pd.Series(dtype=bool)
    rows.append(dict(tag=tag, shards=len(parts), n=len(df), missing=len(missing), kappa_SRME=float(srme.mean()),
                     kappa_SRE=float(sre.mean()), imaginary=int(imag.sum()), broken_symm=int(broken.sum()),
                     srme_eq_2=int((srme >= 2 - 1e-9).sum())))
    print(rows[-1], flush=True)
    out.to_json(f"{d}/kappa_metrics_per_material.json.gz")
print(pd.DataFrame(rows).set_index("tag").round(4).to_string())
