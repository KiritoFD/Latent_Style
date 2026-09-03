"""Generate OMF sweep experiments for style > 0.72 push"""
import json, os

ROOT = "G:/GitHub/Latent_Style/SchrodingerBridge"
with open(os.path.join(ROOT, "config.json")) as f:
    base = json.load(f)

experiments = [
    ("OMF-0_baseline", {"w_kinetic": 1.5, "terminal_swd_weight": 0.15, "w_low_freq": 1.0, "w_cycle": 1.0},
     "Reproduce ~0.716 style baseline"),
    ("OMF-1_style_push", {"w_kinetic": 1.0, "terminal_swd_weight": 0.30, "w_low_freq": 1.0, "w_cycle": 0.5},
     "Double SWD, reduce kinetic"),
    ("OMF-2_aggressive", {"w_kinetic": 0.5, "terminal_swd_weight": 0.50, "w_low_freq": 0.5, "w_cycle": 0.5},
     "High SWD, low kinetic"),
    ("OMF-3_style_focus", {"w_kinetic": 0.25, "terminal_swd_weight": 0.75, "w_low_freq": 0.5, "w_cycle": 0.25},
     "Minimal kinetic, max SWD"),
]

os.makedirs(os.path.join(ROOT, "omf_sweep"), exist_ok=True)

for name, overrides, note in experiments:
    exp_dir = os.path.join(ROOT, "omf_sweep", name)
    os.makedirs(exp_dir, exist_ok=True)
    cfg = json.loads(json.dumps(base))
    cfg["bridge"].update(overrides)
    cfg["bridge"]["note"] = note
    cfg["training"]["num_epochs"] = 12
    cfg["training"]["save_interval"] = 1
    cfg["checkpoint"]["save_dir"] = f"./omf_sweep/{name}/checkpoints"
    
    path = os.path.join(exp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    
    b = cfg["bridge"]
    print(f"{name:25s} w_k={b['w_kinetic']:<5} tswd={b['terminal_swd_weight']:<6} w_lf={b['w_low_freq']:<4} w_cyc={b['w_cycle']:<4}  | {note}")
