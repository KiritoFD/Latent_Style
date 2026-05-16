"""Generate corrected OMF sweep experiments"""
import json, os

ROOT = "G:/GitHub/Latent_Style/SchrodingerBridge"

# Use D0 as base config (known working: 0.7014 style)
d0_path = os.path.join(ROOT, "ablation_destructive_7epoch", "configs", "D0_full_correct_7ep.json")
with open(d0_path) as f:
    base = json.load(f)

experiments = [
    # (name, bridge_overrides, training_epochs, note)
    ("K1_W20_repro", {"w_kinetic": 1.0, "terminal_swd_weight": 20.0, "w_cycle": 0.0, "w_low_freq": 0.0, "w_color": 0.0},
     12, "Reproduce D0/K1: expected ~0.701-0.716 style"),
    ("K0p5_W20", {"w_kinetic": 0.5, "terminal_swd_weight": 20.0, "w_cycle": 0.0, "w_low_freq": 0.0, "w_color": 0.0},
     12, "Reduced kinetic: more style expected"),
    ("K0_W20", {"w_kinetic": 0.0, "terminal_swd_weight": 20.0, "w_cycle": 0.0, "w_low_freq": 0.0, "w_color": 0.0},
     12, "No kinetic: D2-style, high style but content risk"),
    ("K1_W30", {"w_kinetic": 1.0, "terminal_swd_weight": 30.0, "w_cycle": 0.0, "w_low_freq": 0.0, "w_color": 0.0},
     12, "Stronger SWD pull: style push"),
    ("K0p5_W30", {"w_kinetic": 0.5, "terminal_swd_weight": 30.0, "w_cycle": 0.0, "w_low_freq": 0.0, "w_color": 0.0},
     12, "Low kinetic + strong SWD: likely > 0.72"),
    ("K0p25_W40", {"w_kinetic": 0.25, "terminal_swd_weight": 40.0, "w_cycle": 0.0, "w_low_freq": 0.0, "w_color": 0.0},
     12, "Minimal kinetic + max SWD: aggressive style"),
]

os.makedirs(os.path.join(ROOT, "omf_sweep2"), exist_ok=True)

for name, overrides, num_epochs, note in experiments:
    exp_dir = os.path.join(ROOT, "omf_sweep2", name)
    os.makedirs(exp_dir, exist_ok=True)
    cfg = json.loads(json.dumps(base))
    cfg["bridge"].update(overrides)
    cfg["bridge"]["note"] = note
    cfg["training"]["num_epochs"] = num_epochs
    cfg["training"]["save_interval"] = 1
    cfg["training"]["batch_size"] = 32
    cfg["training"]["full_eval_batch_size"] = 6
    cfg["checkpoint"]["save_dir"] = f"./omf_sweep2/{name}/checkpoints"
    
    path = os.path.join(exp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    
    b = cfg["bridge"]
    print(f"{name:20s} w_k={b['w_kinetic']:<6} tswd={b['terminal_swd_weight']:<6}  | {note}")
