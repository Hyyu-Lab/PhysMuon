"""input.json for DPA4 on MPtrj with the paper's Matbench Discovery recipe (arXiv 2606.02419, Supplementary Tables
sm_model_config / sm_matbench_config): model = deepmd preset dpa4-<grade>-v20260820 (Bessel radial basis, post-norm on
the SO(2) branch and pre-norm on the FFN, compiled, bf16 AMP, TF32), WSD cosine schedule, MAE loss with
(E, F, V) = (20, 20, 5), HybridMuon (slice, Gram, Magma-lite, wd 1e-3), ceil(atoms_per_batch / N) structures per step.
usage: python make_input_mini.py <grade> <epochs> <seed> <data_root> <out.json>   (env: MAX_LR, ATOMS_PER_BATCH, NGPU, VAL_ATOMS, STEPS, COMPILE, HM_MOMENTUM, DISP_FREQ, GRAD_CLIP, EMA, CKPT_KEEP)"""
import glob, json, os, sys
import numpy as np

grade, epochs, seed, data, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
PERIODIC_TABLE = """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr
Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg
Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og""".split()
PAPER = {  # grade -> (descriptor knobs from model_preset_data.py, max LR, atoms per batch from the Matbench table)
    "mini": ({"channels": 32, "lmax": 2, "n_blocks": 2, "mixing_layers": 3, "radial_so2_rank": 1, "n_focus": 1}, 2e-3, 7000),
    "neo": ({"channels": 32, "lmax": 3, "n_blocks": 2, "mixing_layers": 3, "radial_so2_rank": 1, "n_focus": 2}, 7e-4, 3000),
    "air": ({"channels": 64, "lmax": 3, "n_blocks": 3, "mixing_layers": 4, "radial_so2_rank": 1, "n_focus": 1}, 4.2e-4, 3150),
}
knobs, max_lr, apb = PAPER[grade]
max_lr = float(os.environ.get("MAX_LR", max_lr)); apb = int(os.environ.get("ATOMS_PER_BATCH", apb))
NGPU = int(os.environ.get("NGPU", 1))   # data parallel: each rank takes ceil((apb / NGPU) / N) structures, so one step still sees ~apb atoms
apb_rank = -(-apb // NGPU)
shared = {"type": "dpa4", "rcut": 6.0, "n_radial": 16, "use_env_seed": True, "mmax": 1, "radial_so2_mode": "degree_channel",
          "focus_dim": 0, "n_atten_head": 1, "message_node_so3": True, "ffn_neurons": 0, "ffn_so3_grid": True, "grid_mlp": False,
          "grid_branch": [0, 0, 1], "ffn_blocks": 1, "so3_readout": "mlp", "precision": "float32",
          "edge_norm": True, "sandwich_norm": [False, True, True, False], "seed": seed}   # last line: version v20260820

# steps per epoch under the batch rule: each system of N atoms contributes ceil(frames / ceil(apb / N)) batches
spe = 0
for d in glob.glob(os.path.join(data, "train", "*")):
    nat = len(open(os.path.join(d, "type.raw")).read().split())
    nf = np.load(os.path.join(d, "set.000", "energy.npy"), mmap_mode="r").shape[0]
    spe += -(-nf // -(-apb_rank // nat))
spe = spe / NGPU                          # optimizer steps per epoch (NGPU rank batches per step)
steps = int(os.environ.get("STEPS", 0)) or max(1, int(round(epochs * spe)))
cfg = {
  "model": {"type": "dpa4", "type_map": PERIODIC_TABLE, "descriptor": {**shared, **knobs},
            "fitting_net": {"type": "dpa4_ener", "neuron": [0], "precision": "float32", "seed": seed},
            "use_compile": os.environ.get("COMPILE", "1") == "1", "enable_tf32": True},
  "learning_rate": {"type": "wsd", "start_lr": max_lr, "stop_lr": 1e-6, "warmup_ratio": 0.003, "warmup_start_factor": 0.2,
                    "decay_phase_ratio": 0.65, "decay_type": "cosine"},
  "loss": {"type": "ener", "loss_func": "mae", "f_use_norm": True, "start_pref_e": 20, "limit_pref_e": 20,
           "start_pref_f": 20, "limit_pref_f": 20, "start_pref_v": 5, "limit_pref_v": 5},
  "optimizer": {"type": "HybridMuon", "muon_mode": "slice", "enable_gram": True, "magma_muon": True, "lr_adjust": 0.0, "weight_decay": 0.001,
                "momentum": float(os.environ.get("HM_MOMENTUM", 0.95))},
  "validating": {"compiled_infer": os.environ.get("COMPILE", "1") == "1"},   # eager validation at 7000 atoms OOMs on 32 GB
  "training": {
    "training_data": {"systems": f"{data}/train", "batch_size": f"auto:{apb_rank}"},
    "validation_data": {"systems": f"{data}/valid", "batch_size": f"auto:{os.environ.get('VAL_ATOMS', 2000)}", "numb_btch": 4},
    "numb_steps": steps, "seed": seed, "disp_file": "lcurve.out", "disp_freq": int(os.environ.get("DISP_FREQ", 1000)),
    "save_freq": max(1, min(steps, 50000)), "max_ckpt_keep": int(os.environ.get("CKPT_KEEP", 1)),
    # Match deepmd-kit's official DPA4 defaults (gradient_max_norm=5.0, EMA decay=0.999); set GRAD_CLIP=0 / EMA=0 to disable.
    "gradient_max_norm": float(os.environ.get("GRAD_CLIP", 5.0)),
    "enable_ema": os.environ.get("EMA", "1") == "1", "ema_decay": 0.999, "ema_ckpt_keep": int(os.environ.get("CKPT_KEEP", 1)),
  },
}
json.dump(cfg, open(out, "w"), indent=2)
print(f"wrote {out}: grade={grade} epochs={epochs} steps_per_epoch={spe} numb_steps={steps} max_lr={max_lr} atoms_per_batch={apb} ngpu={NGPU} atoms_per_rank={apb_rank} compile={cfg['model']['use_compile']} momentum={cfg['optimizer']['momentum']} grad_clip={cfg['training']['gradient_max_norm']} ema={cfg['training']['enable_ema']}")
