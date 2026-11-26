"""Recompute the molecular-dynamics numbers of the paper from the files in data/.

    python analyze.py                  # prints everything; the output of the published data is expected_output.txt

Sections: 1 routine test, 2 SPICE2 stress tests (with exact 2^6 seed-level permutation tests and exhaustive 6^6
seed-cluster bootstrap intervals), 3 Arrhenius temperature scaling of bond breaking, 4 accuracy on ANI-1x MD
configurations, 5 SPICE2 bond-stretch scans. Needs numpy only.
"""
import argparse, collections, glob, itertools, json, os, re
from math import comb
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ap.add_argument("--draws", type=int, default=4000, help="molecule-stratified bootstrap draws for Arrhenius fit")
args = ap.parse_args()
D, NB = args.data, args.draws
NAME = {"muon": "Muon", "physmuon": "PhysMuon"}
TS = np.array([1200, 1500, 1800, 2100]); ARMS = ("muon", "physmuon"); BT = ("C-H", "C-Si", "Cl-Si")
KB, TEND = 8.617333e-5, 42.0


def fisher_p(a, b, c, d):
    """Two-sided Fisher exact test of the 2 x 2 table [[a, b], [c, d]]."""
    n1, n2, m1 = a + b, c + d, a + c; N = n1 + n2
    p = lambda x: comb(n1, x) * comb(n2, m1 - x) / comb(N, m1)
    p0 = p(a); return min(1.0, sum(p(x) for x in range(max(0, m1 - n2), min(m1, n1) + 1) if p(x) <= p0 * (1 + 1e-9)))


def paired_p(diff):
    """Exact two-sided sign-flip test of paired differences over all sign assignments."""
    dist = {0: 1.0}
    for x in (abs(int(v)) for v in diff if v != 0):
        new = collections.defaultdict(float)
        for s, p in dist.items(): new[s + x] += p / 2; new[s - x] += p / 2
        dist = new
    obs = abs(int(sum(diff)))
    return sum(p for s, p in dist.items() if abs(s) >= obs)


def bondtypes(z):
    out = set()
    for s in z.get("broken_bonds_confirmed") or z.get("broken_bonds") or []:
        a, b = s.split()[0].split("-"); out.add("-".join(sorted([re.match(r"[A-Za-z]+", a).group(), re.match(r"[A-Za-z]+", b).group()])))
    return out


def hill(c):
    return "".join(k + (str(c[k]) if c[k] > 1 else "") for k in [e for e in ("C", "H") if e in c] + sorted(e for e in c if e not in ("C", "H")))


# ============================================================================ 1. routine test
print("##### 1. Routine test: trajectory 0 at 300 K without thermostat after 2 ps, six trajectories at two temperatures; 42 ps each")
R = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/routine/*.json"))]
for ds in ("ani1x", "spice2"):
    rr = [r for r in R if r["dataset"] == ds]; temps = sorted(set(t for r in rr for t in r["temps"][1:]))
    print(f"\n=== {ds}: {len(set(r['mol'] for r in rr))} molecules, {len(set(r['seed'] for r in rr))} seeds ===")
    for opt in ARMS:
        q = [r for r in rr if r["opt"] == opt]; line = [f"{NAME[opt]:>8}: {len(q)} tasks"]
        for T in temps:
            z = [r["replica_failures"][k] for r in q for k, Tk in enumerate(r["temps"]) if k > 0 and Tk == T]
            line.append(f"{T:.0f} K unstable {sum(1 for x in z if x)}/{len(z)}")
        d = np.array([abs(r["nve_drift_meV_atom_ps"]) for r in q if "nve_drift_meV_atom_ps" in r])
        line.append(f"300 K without thermostat: unstable {sum(1 for r in q if r['replica_failures'][0])}/{len(q)}, "
                    f"energy drift mean |d| {d.mean():.4f} meV/atom/ps, above 1e-3 in {int((d > 1e-3).sum())} trajectories, largest {d.max():.4f}")
        print("  " + " | ".join(line))
    for r in rr:
        for k, z in enumerate(r["replica_failures"]):
            if z: print(f"    unstable: {NAME[r['opt']]} seed {r['seed']} molecule m{r['mol']} ({hill(r['formula'])}) at {r['temps'][k]:.0f} K after {z['t_ps']} ps, {z['fragments']} fragments")
dr = {o: np.array([abs(r["nve_drift_meV_atom_ps"]) for r in R if r["opt"] == o]) for o in ARMS}
print("\n  both data sets: energy drift above 1e-3 meV/atom/ps in " + ", ".join(f"{int((dr[o] > 1e-3).sum())} of {len(dr[o])} {NAME[o]} trajectories (largest {dr[o].max():.4f})" for o in dr))

# ============================================================================ 2. SPICE2 stress tests
print("\n##### 2. SPICE2 stress tests: seven trajectories at one temperature, 42 ps; a split counts when it lasts 100 fs")
S = {}
for f in sorted(glob.glob(f"{D}/stress/*.json")):
    r = json.load(open(f)); S[(r["dataset"], int(round(r["stress_T"])), r["opt"], r["seed"], r["mol"])] = r

R2 = {(T, a, s, m): r for (ds, T, a, s, m), r in S.items() if ds == "spice2"}
miss = [(T, a, s, m) for T in TS for a in ARMS for s in range(6) for m in range(8) if (int(T), a, s, m) not in R2]
assert not miss, f"missing runs: {miss[:5]} ... ({len(miss)})"
FORM = {m: hill(R2[(1500, "muon", 0, m)]["formula"]) for m in range(8)}


def arrays(a):
    """(4 temperatures, 6 seeds, 8 molecules): unstable trajectories, ps at risk, healed splits, the same by broken bond, times."""
    nf, X, H = np.zeros((4, 6, 8)), np.zeros((4, 6, 8)), np.zeros((4, 6, 8)); B = {b: np.zeros((4, 6, 8)) for b in BT + ("other",)}; tf = [[] for _ in TS]
    for i, T in enumerate(TS):
        for s in range(6):
            for m in range(8):
                r = R2[(int(T), a, s, m)]
                for k, z in enumerate(r["replica_failures"]):
                    X[i, s, m] += min(z["t_ps"], TEND) if z else TEND; H[i, s, m] += r["transient_splits"][k]
                    tf[i].append(z["t_ps"] if z else np.inf)
                    if not z: continue
                    nf[i, s, m] += 1; bt = bondtypes(z)
                    for b in BT:
                        if b in bt: B[b][i, s, m] += 1
                    if not bt & set(BT): B["other"][i, s, m] += 1
    return nf, X, H, B, [np.array(t) for t in tf]


A = {a: arrays(a) for a in ARMS}
print("\n=== SPICE2: 8 molecules x 6 seeds x 7 trajectories = 336 trajectories per optimizer and temperature ===")
for i, T in enumerate(TS):
    for a in ARMS:
        nf, X, H, B, tf = A[a]; t = tf[i]; n = int(nf[i].sum())
        print(f"  {T} K {NAME[a]:>10}: unstable {n}/336 ({100 * n / 336:.1f}%) | stable at 10/20/30/42 ps: "
              + "/".join(f"{100 * (t > p).mean():.0f}" for p in (10, 20, 30)) + f"/{100 * (1 - n / 336):.0f}% | 42-ps RMST {np.minimum(t, TEND).mean():.1f} ps"
              f" | healed splits per trajectory {H[i].sum() / 336:.2f} | per seed {[int(v) for v in nf[i].sum(axis=1)]}")

print("\n  Seed-level comparison (n = 6 seeds, exact 2^6 sign-flip test and exhaustive 6^6 seed-cluster bootstrap 95% CIs)")
idx = np.array(list(itertools.product(range(6), repeat=6)))


def seed_cluster_ci(fm, fp, tm, tp):
    rr0 = (fm.sum() / tm.sum()) / (fp.sum() / tp.sum())
    fm_b, fp_b = fm[idx].sum(1), fp[idx].sum(1)
    tm_b, tp_b = tm[idx].sum(1), tp[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr_b = (fm_b / tm_b) / (fp_b / tp_b)
        rr_b = rr_b[~np.isnan(rr_b)]
        lo, hi = np.quantile(rr_b, [0.025, 0.975])
    rmst_b = (tp_b - tm_b) / 336.0
    rlo, rhi = np.quantile(rmst_b, [0.025, 0.975])
    return rr0, (lo, hi), (tp.sum() - tm.sum()) / 336.0, (rlo, rhi)


for i, T in enumerate(TS):
    fm, tm = A["muon"][0][i].sum(axis=1), A["muon"][1][i].sum(axis=1)
    fp, tp = A["physmuon"][0][i].sum(axis=1), A["physmuon"][1][i].sum(axis=1)
    dp = (A["muon"][0][i] - A["physmuon"][0][i]).ravel()
    ds = fm - fp
    rr0, (rlo, rhi), drmst, (mlo, mhi) = seed_cluster_ci(fm, fp, tm, tp)
    hi_s = "inf" if not np.isfinite(rhi) else f"{rhi:.2f}"
    print(f"  {T} K: seeds fewer/more {int((ds > 0).sum())}/{int((ds < 0).sum())} (p = {paired_p(ds):.4f}) | "
          f"pairs {int((dp > 0).sum())}/{int((dp < 0).sum())} | rate ratio {rr0:.2f} [{rlo:.2f}, {hi_s}] | "
          f"RMST gain {drmst:+.2f} ps [{mlo:+.2f}, {mhi:+.2f}]")

fm_all, tm_all = A["muon"][0].sum(axis=(0, 2)), A["muon"][1].sum(axis=(0, 2))
fp_all, tp_all = A["physmuon"][0].sum(axis=(0, 2)), A["physmuon"][1].sum(axis=(0, 2))
ds_all = fm_all - fp_all
dp_all = (A["muon"][0].sum(axis=0) - A["physmuon"][0].sum(axis=0)).ravel()
rr_all, (rlo, rhi), _, _ = seed_cluster_ci(fm_all, fp_all, tm_all, tp_all)
print(f"  All 4 combined: Muon {int(fm_all.sum())} vs PhysMuon {int(fp_all.sum())} | "
      f"seeds {int((ds_all > 0).sum())}/{int((ds_all < 0).sum())} (p = {paired_p(ds_all):.4f}) | "
      f"pairs {int((dp_all > 0).sum())}/{int((dp_all < 0).sum())} | rate ratio {rr_all:.2f} [{rlo:.2f}, {rhi:.2f}]")

print("\n  unstable trajectories per molecule (of 42), Muon / PhysMuon")
cells = {"fewer": 0, "equal": 0, "more": 0}
for m in range(8):
    row = []
    for i, T in enumerate(TS):
        mu, pm = int(A["muon"][0][i][:, m].sum()), int(A["physmuon"][0][i][:, m].sum()); row.append(f"{T} K {mu}/{pm}")
        cells["fewer" if pm < mu else "equal" if pm == mu else "more"] += 1
    print(f"  m{m} {FORM[m]:<14} " + " | ".join(row))
print(f"  of the 32 molecule-temperature groups, PhysMuon has fewer in {cells['fewer']}, equal in {cells['equal']}, more in {cells['more']}")
print("  broken bonds, all temperatures: " + ", ".join(f"{NAME[a]} " + " ".join(f"{b} {int(A[a][3][b].sum())}" for b in BT + ("other",)) for a in ARMS))

# Exhaustive 6^6 seed-cluster bootstrap for equivalent tolerated temperature
def teq_one(mu_f, pm_curve):
    if mu_f <= pm_curve[0]: return float(TS[0])
    if mu_f >= pm_curve[-1]: return float(TS[-1])
    for k in range(3):
        if pm_curve[k] <= mu_f <= pm_curve[k + 1]:
            return float(TS[k]) if pm_curve[k + 1] == pm_curve[k] else float(TS[k] + (TS[k + 1] - TS[k]) * (mu_f - pm_curve[k]) / (pm_curve[k + 1] - pm_curve[k]))
    return np.nan


print("\n  equivalent tolerated temperature (linearly interpolated; exact 6^6 seed-cluster bootstrap 95% CIs)")
mu_st = A["muon"][0].sum(axis=2).T       # (6, 4)
pm_st = A["physmuon"][0].sum(axis=2).T   # (6, 4)
pm_boot = pm_st[idx].sum(axis=1)         # (46656, 4)
for ti, T in enumerate(TS[:3]):
    pt = teq_one(mu_st[:, ti].sum(), pm_st.sum(axis=0))
    mu_boot = mu_st[idx, ti].sum(axis=1)
    vals = np.array([teq_one(m, p) for m, p in zip(mu_boot, pm_boot)])
    lo, hi = np.quantile(vals, [0.025, 0.975])
    print(f"  Muon at {T} K ({100 * mu_st[:, ti].sum() / 336:.1f}% unstable) -> PhysMuon at {pt:.0f} K (+{pt - T:.0f} K) [{lo:.0f}, {hi:.0f}]")

# ============================================================================ 3. Arrhenius temperature scaling
print("\n##### 3. Temperature scaling: shared-prefactor Arrhenius fit, rate = A exp(-E_opt / (kB T))")
print("  lam = E_PhysMuon / E_Muon: PhysMuon at T breaks a bond as often as Muon at T / lam")
SCOPE = {"all instabilities": ("any", range(8)), "C-H, all molecules": ("C-H", range(8)), "C-Si, m0 m3 m4": ("C-Si", (0, 3, 4)),
         "Si-Cl, m0": ("Cl-Si", (0,)), "C-H, m1": ("C-H", (1,)), "Si-C, m3": ("C-Si", (3,))}
A2 = {}
for a in ARMS:
    nf, X, H, B, _ = A[a]; A2[a] = (X, {"any": nf, **{b: B[b] for b in BT}})


def pglm(n, off, Xd):
    """Poisson regression n ~ exp(off + Xd @ beta)."""
    beta = np.linalg.lstsq(Xd, np.log((n + 0.5) / np.exp(off)), rcond=None)[0]
    for _ in range(200):
        eta = off + Xd @ beta
        if not np.all(np.isfinite(eta)) or np.abs(eta).max() > 50.0: return None
        mu = np.exp(eta); g = Xd.T @ (n - mu); Hs = Xd.T @ (Xd * mu[:, None])
        try: st = np.linalg.solve(Hs, g)
        except np.linalg.LinAlgError: return None
        beta = beta + st
        if not np.all(np.isfinite(beta)): return None
        if np.abs(st).max() < 1e-9: return beta
    return None


def fit(tot):
    n = np.concatenate([tot[a][0] for a in ARMS]); X = np.concatenate([tot[a][1] for a in ARMS]); x = np.tile(-1.0 / (KB * TS), 2)
    im, ip = np.r_[np.ones(4), np.zeros(4)], np.r_[np.zeros(4), np.ones(4)]
    b = pglm(n, np.log(X), np.column_stack([np.ones(8), x * im, x * ip]))
    return (np.nan, np.nan, np.nan) if b is None else (b[1], b[2], b[2] / b[1])


def totals(ib=None):
    out = {}
    for sc, (b, mols) in SCOPE.items():
        out[sc] = {}
        for a in ARMS:
            X, N = A2[a]
            pick = (lambda v: sum(v[:, :, m].sum(axis=1) for m in mols)) if ib is None else (lambda v: sum(v[:, ib[m], m].sum(axis=1) for m in mols))
            out[sc][a] = (pick(N[b]), pick(X))
    return out


rng = np.random.default_rng(20261031)
T0 = totals(); F0 = {sc: fit(T0[sc]) for sc in SCOPE}; FB = []
for _ in range(NB):
    tb = totals(rng.integers(0, 6, size=(8, 6))); FB.append({sc: fit(tb[sc])[2] for sc in SCOPE})
for sc in SCOPE:
    lo, hi = np.nanpercentile([b[sc] for b in FB], [2.5, 97.5]); em, ep, lam = F0[sc]
    print(f"  {sc:<20} lam = {lam:.3f} [{lo:.3f}, {hi:.3f}] | E_Muon {em:.2f}, E_PhysMuon {ep:.2f} eV | broken at 1200/1500/1800/2100 K: "
          + ", ".join(f"{NAME[a]} " + "/".join(f"{v:.0f}" for v in T0[sc][a][0]) for a in ARMS))

# ============================================================================ 4. accuracy on ANI-1x MD configurations
print("\n##### 4. ANI-1x: accuracy on configurations from the trajectories, against wB97x/6-31G(d)")
keys = [f"{o}_s{s}" for o in ARMS for s in range(4)]; rows = []
for f in sorted(glob.glob(f"{D}/dft_ani1x/ani1x_*.npz")):
    tag = os.path.basename(f)[:-4]; z = np.load(f); e = np.load(f"{D}/model_predictions/{tag}.npz"); ok = z["converged"]; n = z["F"].shape[1]
    ferr = {k: np.abs(e[f"F_{k}"] - z["F"]).mean(axis=(1, 2)) * 1000.0 for k in keys}; eerr = {}
    for k in keys:
        de = e[f"E_{k}"] - z["E"]; eerr[k] = (de - de[ok].mean()) / n * 1000.0        # constant offset of the molecule removed
    for i in range(len(ok)):
        if ok[i]: rows.append((str(z["labels"][i]), {k: float(ferr[k][i]) for k in keys}, {k: float(abs(eerr[k][i])) for k in keys}))
for g, lab in (("all", "all"), ("300K_NVE", "300 K"), ("1000K", "1,000 K"), ("1500K", "1,500 K")):
    sel = [r for r in rows if g == "all" or r[0] == g]
    mu = [np.mean([r[1][f"muon_s{s}"] for r in sel]) for s in range(4)]; pm = [np.mean([r[1][f"physmuon_s{s}"] for r in sel]) for s in range(4)]
    emu = np.mean([np.mean([r[2][f"muon_s{s}"] for r in sel]) for s in range(4)]); epm = np.mean([np.mean([r[2][f"physmuon_s{s}"] for r in sel]) for s in range(4)])
    print(f"  {lab:>8} ({len(sel):>3} configurations): force MAE Muon {np.mean(mu):.1f} [{min(mu):.1f}-{max(mu):.1f}] -> PhysMuon {np.mean(pm):.1f} [{min(pm):.1f}-{max(pm):.1f}] meV/A "
          f"({100 * (np.mean(pm) / np.mean(mu) - 1):+.1f}%; PhysMuon better in {sum(p < m for p in pm for m in mu)}/16 model pairs) | "
          f"energy MAE {emu:.2f} -> {epm:.2f} meV/atom ({100 * (epm / emu - 1):+.1f}%)")
v = json.load(open(f"{D}/dft_ani1x/validation.json"))
print(f"  the DFT setup against the labels of the data set: force difference {np.mean([r['force_MAE_vs_label_meV_A'] for r in v]):.1f} meV/A on {len(v)} validation configurations")

# ============================================================================ 5. SPICE2 bond-stretch scans
print("\n##### 5. SPICE2: bond-stretch scans against wB97M-D3(BJ)/def2-TZVPPD (the lower of the restricted and the unrestricted energy)")
z = np.load(f"{D}/bond_scan/model_energies.npz"); DS = z["DS"]; i0 = int(np.argmin(np.abs(DS)))
PTS = [1.0, 1.5, 2.0, 2.5, 3.0]; frac_all = {a: [] for a in ARMS}; mae_gain = []


def at(d):
    k = int(np.argmin(np.abs(DS - d))); assert abs(DS[k] - d) < 1e-6; return k


for name, lab in (("m0_Si7-Cl1", "Si-Cl (m0)"), ("m1_C10-H22", "C-H (m1)"), ("m3_Si17-C16", "Si-C (m3)")):
    u = {round(x["d_A"], 2): x for x in json.load(open(f"{D}/bond_scan/dft_unrestricted_{name}.json"))}
    dft = {d: min(x["E_UKS_eV"], x["E_RKS_eV"]) - u[0.0]["E_RKS_eV"] for d, x in u.items()}
    Sd = {a: np.array([z[f"{name}|{a}|s{s}"] - z[f"{name}|{a}|s{s}"][i0] for s in range(6)]) for a in ARMS}
    fr = {a: np.mean([Sd[a][:, at(p)].mean() / dft[p] for p in PTS]) for a in ARMS}
    for a in ARMS: frac_all[a].append(fr[a])
    up = sum(Sd["physmuon"][:, at(p)].mean() for p in PTS) / sum(Sd["muon"][:, at(p)].mean() for p in PTS) - 1
    near = max(abs(Sd[a][:, at(d)].mean() - dft[d]) for a in ARMS for d in (-0.2, 0.2))
    pts9 = sorted(d for d in dft if d != 0.0)
    mae = {a: np.mean([abs(Sd[a][:, at(d)].mean() - dft[d]) for d in pts9]) for a in ARMS}; mae_gain.append(1 - mae["physmuon"] / mae["muon"])
    print(f"  {lab}: model energy / DFT energy at +1.0 to +3.0 A: " + ", ".join(f"{NAME[a]} {100 * fr[a]:.1f}%" for a in ARMS) + f" | PhysMuon above Muon by {100 * up:+.1f}%")
    print(f"      within 0.2 A of the start: largest |model - DFT| {near:.3f} eV | 9-point MAE: "
          + ", ".join(f"{NAME[a]} {mae[a]:.3f}" for a in ARMS) + f" eV ({100 * (mae['physmuon'] / mae['muon'] - 1):+.1f}%)")
print("  ranges over the three bonds: " + ", ".join(f"{NAME[a]} {100 * min(v):.1f}%-{100 * max(v):.1f}%" for a, v in frac_all.items())
      + f" | full-curve MAE lower by {100 * min(mae_gain):.1f}%-{100 * max(mae_gain):.1f}%")
v = json.load(open(f"{D}/bond_scan/dft_validation.json"))
print(f"  the DFT setup against the labels of the data set: force difference {np.mean([r['force_MAE_vs_label_meV_A'] for r in v]):.1f} meV/A on {len(v)} starting configurations")
