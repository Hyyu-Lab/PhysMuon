"""input.json for DPA-3 on ANI-1x (wB97X/6-31G*): the repflow block used for all DPA-3 tasks here, dynamic neighbour selection,
non-periodic molecules, energy + force loss (the usual deepmd molecular prefactors).  Batch from env BATCH (default auto:256).
usage: python make_input_ani1x.py <steps> <seed> <data_root> <out.json> [lr_mult]"""
import json, os, sys
steps, seed, data, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
lr_mult = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
SCHED = os.environ.get("SCHED", "exp")
BATCH = os.environ.get("BATCH", "auto:256")
first = sorted(os.listdir(os.path.join(data, "train")))[0]
type_map = open(os.path.join(data, "train", first, "type_map.raw")).read().split()
cfg = {
  "model": {
    "type_map": type_map,
    "descriptor": {"type": "dpa3", "repflow": {"n_dim": 128, "e_dim": 64, "a_dim": 32, "nlayers": 6, "e_rcut": 6.0, "e_rcut_smth": 5.3, "e_sel": 1200, "a_rcut": 4.0, "a_rcut_smth": 3.5, "a_sel": 300, "axis_neuron": 4, "fix_stat_std": 0.3, "a_compress_rate": 1, "a_compress_e_rate": 2, "a_compress_use_split": True, "update_angle": True, "smooth_edge_update": True, "edge_init_use_dist": True, "use_exp_switch": True, "use_dynamic_sel": True, "sel_reduce_factor": 10.0, "update_style": "res_residual", "update_residual": 0.1, "update_residual_init": "const"},
                   "activation_function": "silut:10.0", "use_tebd_bias": False, "precision": "float32", "concat_output_tebd": False, "seed": seed},
    "fitting_net": {"neuron": [240, 240, 240], "resnet_dt": True, "precision": "float32", "activation_function": "silut:10.0", "seed": seed},
  },
  "learning_rate": ({"type": "exp", "decay_steps": max(steps // 50, 1), "start_lr": 1e-3 * lr_mult, "stop_lr": 3e-5 * lr_mult}
                    if SCHED == "exp" else
                    {"type": "wsd", "start_lr": 1e-3 * lr_mult, "stop_lr": 3e-5 * lr_mult, "decay_phase_ratio": 0.7, "decay_type": "linear"}),
  "loss": {"type": "ener", "start_pref_e": 0.02, "limit_pref_e": 1.0, "start_pref_f": 1000.0, "limit_pref_f": 1.0, "start_pref_v": 0.0, "limit_pref_v": 0.0},
  "training": {
    "training_data": {"systems": f"{data}/train", "batch_size": BATCH},
    "validation_data": {"systems": f"{data}/valid", "batch_size": BATCH, "numb_btch": 6},
    "numb_steps": steps, "seed": seed, "disp_file": "lcurve.out", "disp_freq": 100,
    "save_freq": steps, "max_ckpt_keep": 1, "gradient_max_norm": 5.0,
  },
  "optimizer": {"type": "Adam"},
}
json.dump(cfg, open(out, "w"), indent=2)
print(f"wrote {out}: steps={steps} seed={seed} lr_mult={lr_mult} sched={SCHED} batch={BATCH} types={type_map}")
