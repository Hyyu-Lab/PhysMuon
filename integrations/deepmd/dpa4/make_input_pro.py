"""input.json reproducing the Matbench Discovery entry DPA-4.0.1-Pro-MPtrj (v2026.06.05), optimizer swappable.
Model section: verbatim from the released checkpoint's model_params (22,808,191 parameters; bf16 AMP in the descriptor,
compiled, TF32). Training: the model card (16 GPUs, 2,000,000 steps, WSD max 5e-4 / min 1e-6, 5,000 warmup steps,
MAE loss 20/20/5, batch "filter:300" per GPU, gradient norm 5) and the checkpoint's optimizer state (HybridMuon slice,
Gram, flash, Magma-lite, lr_adjust 0 / coeff 0.18, wd 1e-3, momentum 0.95, Adam betas 0.9/0.95; EMA weights present).
usage: python make_input_pro.py <model_params.json> <data_root> <out.json>
env: STEPS (default 2000000), DECAY_RATIO (0.65), START_LR (5e-4), STOP_LR (1e-6), BATCH (default filter:300), SEED (42), HM_MOMENTUM, DISP_FREQ, SAVE_FREQ,
CKPT_KEEP, EMA_DECAY, FULL_VAL, FULL_VAL_FREQ (5000),
FULL_VAL_EMA (1)"""
import json, os, sys

mp_path, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
model = json.load(open(mp_path))
model.pop("info", None)                       # runtime bookkeeping written by the trainer, not an input key
model["fitting_net"]["type"] = "dpa4_ener"     # 3.2.0b0 stored "ener"; 3.2.0 names the SeZM fitting dpa4_ener / sezm_ener
seed = int(os.environ.get("SEED", 42))
model["descriptor"]["seed"] = seed
model["fitting_net"]["seed"] = seed
steps = int(os.environ.get("STEPS", 2000000))
cfg = {
    "model": model,
    # DECAY_RATIO with STEPS: a time-compressed tail of the official cosine schedule
    # START_LR=STOP_LR gives a flat learning rate (the WSD cosine interpolates between the two, so equal ends cancel)
    "learning_rate": {"type": "wsd", "start_lr": float(os.environ.get("START_LR", 5e-4)), "stop_lr": float(os.environ.get("STOP_LR", 1e-6)),
                      "warmup_steps": min(5000, max(1, steps // 400)),
                      "decay_phase_ratio": float(os.environ.get("DECAY_RATIO", 0.65)), "decay_type": "cosine"},
    "loss": {"type": "ener", "loss_func": "mae", "f_use_norm": True, "start_pref_e": 20, "limit_pref_e": 20,
             "start_pref_f": 20, "limit_pref_f": 20, "start_pref_v": 5, "limit_pref_v": 5},
    "optimizer": {"type": "HybridMuon", "muon_mode": "slice", "enable_gram": True, "flash_muon": True, "magma_muon": True,
                  "lr_adjust": 0.0, "lr_adjust_coeff": 0.18, "weight_decay": 0.001, "adam_beta1": 0.9, "adam_beta2": 0.95,
                  "momentum": float(os.environ.get("HM_MOMENTUM", 0.95))},
    # FULL_VAL=1: the official run's checkpoint selection (its train_infos: full_validation_metric e:mae, top-3 records at
    # 5,000-step spacing) -- full validation-set pass every 5,000 steps, best-3 checkpoints kept, also for the EMA weights
    "validating": {"compiled_infer": True, **({"full_validation": True,
                   "ema_full_validation": os.environ.get("FULL_VAL_EMA", "1") == "1",
                   "validation_freq": int(os.environ.get("FULL_VAL_FREQ", 5000)),
                   "save_best": True, "max_best_ckpt": 3, "validation_metric": "E:MAE", "full_val_start": 0}
                  if os.environ.get("FULL_VAL", "0") == "1" else {})},
    "training": {
        "training_data": {"systems": f"{data}/train", "batch_size": os.environ.get("BATCH", "filter:300")},
        "validation_data": {"systems": f"{data}/valid", "batch_size": "max:300", "numb_btch": 8},   # 2000-atom batches OOM on 32 GB (Pro)
        "numb_steps": steps, "seed": seed, "disp_file": "lcurve.out", "disp_freq": int(os.environ.get("DISP_FREQ", 1000)),
        "save_freq": int(os.environ.get("SAVE_FREQ", 50000)), "max_ckpt_keep": int(os.environ.get("CKPT_KEEP", 60)),
        "gradient_max_norm": 5.0,
        "enable_ema": True, "ema_decay": float(os.environ.get("EMA_DECAY", 0.999)), "ema_ckpt_keep": int(os.environ.get("CKPT_KEEP", 60)),
    },
}
json.dump(cfg, open(out, "w"), indent=2)
d = model["descriptor"]
print(f"wrote {out}: DPA-4.0.1-Pro channels={d['channels']} lmax={d['lmax']} n_blocks={d['n_blocks']} use_amp={d.get('use_amp')} "
      f"steps={steps} decay_ratio={cfg['learning_rate']['decay_phase_ratio']} lr={cfg['learning_rate']['start_lr']:g}->{cfg['learning_rate']['stop_lr']:g} "
      f"batch={cfg['training']['training_data']['batch_size']} "
      f"momentum={cfg['optimizer']['momentum']} seed={seed} full_val={int('full_validation' in cfg['validating'])}")
