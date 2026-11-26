#!/usr/bin/env python
"""Matbench Discovery metrics of one or more relaxation runs, with the steps of the official pipeline: raw model energy
-> the WBM ComputedStructureEntry with the relaxed structure -> MaterialsProject2020Compatibility (clean=True) ->
formation energy with the MP elemental reference energies -> energy above the hull, shifted from the DFT hull ->
stable_metrics (on all structures and on the unique prototypes); RMSD with StructureMatcher(stol=1.0, scale=False)
against the DFT-relaxed WBM structure (unmatched structures count as 1.0).

    python metrics.py runs TAG [TAG ...]          # reads runs/TAG/shard-*.jsonl, writes runs/subset_metrics_TAG.csv

Needs pymatgen and matbench_discovery. The WBM summary and the computed structure entries are taken from the
environment variables WBM_SUMMARY and WBM_CSE, or downloaded by matbench_discovery.
"""
import glob, gzip, json, os, sys
from multiprocessing import get_context
import numpy as np, pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from matbench_discovery.enums import DataFiles
from matbench_discovery.energy import calc_energy_from_e_refs, mp_elemental_ref_energies
from matbench_discovery.metrics.discovery import stable_metrics

runs, tags = sys.argv[1], sys.argv[2:]
SM = StructureMatcher(stol=1.0, scale=False)


def load(tag):
    recs = {}
    for path in sorted(glob.glob(f"{runs}/{tag}/shard-*.jsonl")):
        for line in open(path):
            r = json.loads(line); recs[r["material_id"]] = r
    return recs


def to_struct(r):
    return Structure(Lattice(r["cell"]), r["numbers"], r["positions"], coords_are_cartesian=True)


def rms(args):
    mid, s_ml, s_dft = args
    res = SM.get_rms_dist(Structure.from_dict(s_ml), Structure.from_dict(s_dft))
    return mid, (res[0] if res else np.nan)


all_recs = {t: load(t) for t in tags}
ids = set.intersection(*[set(v) for v in all_recs.values()])
WBM_SUMMARY = os.environ.get("WBM_SUMMARY") or DataFiles.wbm_summary.path
WBM_CSE = os.environ.get("WBM_CSE") or DataFiles.wbm_computed_structure_entries.path
print("summary:", WBM_SUMMARY, "| cse:", WBM_CSE, flush=True)
df = pd.read_csv(WBM_SUMMARY).set_index("material_id")
print(f"models {tags}; common subset {len(ids)} structures (of {len(df)})", flush=True)
cse_by_id = {}
with gzip.open(WBM_CSE, "rt") as f:
    for line in f:
        d = json.loads(line)
        if d["material_id"] in ids:
            cse_by_id[d["material_id"]] = d["computed_structure_entry"]
sub = df.loc[sorted(ids)]
e_dft = sub["e_form_per_atom_mp2020_corrected"]; each_true = sub["e_above_hull_mp2020_corrected_ppd_mp"]
uniq_col = "unique_prototype" if "unique_prototype" in sub.columns else "uniq"
uniq = sub[uniq_col].astype(bool)
compat = MaterialsProject2020Compatibility()
rows = []
for tag in tags:
    recs, e_form, pairs = all_recs[tag], {}, []
    n_err = sum("error" in recs[m] for m in ids); n_conv = sum(bool(recs[m].get("converged")) for m in ids)
    for mid in sub.index:
        r = recs[mid]
        if r.get("energy") is None:
            e_form[mid] = np.nan; continue
        cse = ComputedStructureEntry.from_dict(cse_by_id[mid])
        s_ml = to_struct(r)
        cse._energy = r["energy"]; cse._structure = s_ml
        proc = compat.process_entry(cse, clean=True)
        e_form[mid] = np.nan if proc is None else calc_energy_from_e_refs(
            sub.loc[mid, "formula"], ref_energies=mp_elemental_ref_energies, total_energy=proc.energy)
        pairs.append((mid, s_ml.as_dict(), cse_by_id[mid]["structure"]))
    pred = pd.Series(e_form).reindex(sub.index).round(4)
    p = pred.mask((pred - e_dft).abs() > 5.0).round(3)
    each_pred = each_true.round(3) + p - e_dft.round(3)
    m_full = stable_metrics(each_true.round(3), each_pred, fillna=True)
    m_uniq = stable_metrics(each_true.round(3)[uniq], each_pred[uniq], fillna=True)
    with get_context("fork").Pool(32) as pool:   # py3.14 defaults to forkserver, which re-imports this unguarded script
        rr = dict(pool.imap_unordered(rms, pairs, chunksize=64))
    rmsd_s = pd.Series(rr).astype(float).fillna(1.0)
    rmsd = rmsd_s.mean()
    rmsd_auc = np.maximum(1.0 - rmsd_s / 0.05, 0.0).mean()
    e_mae = (p - e_dft).abs().mean()
    rows.append(dict(tag=tag, n=len(sub), n_uniq=int(uniq.sum()), errors=n_err, converged=n_conv, F1_uniq=m_uniq["F1"], F1_full=m_full["F1"],
                     Prec_uniq=m_uniq["Precision"], Rec_uniq=m_uniq["Recall"], MAE_ehull_uniq=m_uniq["MAE"], RMSE_ehull_uniq=m_uniq["RMSE"],
                     R2_uniq=m_uniq["R2"], MAE_eform=e_mae, RMSD=rmsd, RMSD_AUC=rmsd_auc))
    print(rows[-1], flush=True)
out = pd.DataFrame(rows).set_index("tag")
pd.set_option("display.width", 250); print(out.round(4).to_string())
out.to_csv(f"{runs}/subset_metrics_{'_'.join(tags)[:80]}.csv")
