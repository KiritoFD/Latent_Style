import json, os
batch_dir = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20260617_1749_ot_vertical_sweep"
for name in sorted(os.listdir(batch_dir)):
    cfg_path = os.path.join(batch_dir, name, "config.json")
    if not os.path.exists(cfg_path): continue
    c = json.load(open(cfg_path))
    b = c["training"]["batch_size"]
    vl = c["training"]["virtual_length_multiplier"]
    bpm = c.get("bridge", {}).get("bridge_path_mode", "?")
    sig = c.get("bridge", {}).get("bridge_sigma", 0)
    csm = c.get("bridge", {}).get("coupling_structure_cost_mode", "?")
    solver = c.get("bridge", {}).get("coupling_solver", "?")
    print(f"{name:25s} b={b:2d} vl={vl:.2f} path={bpm:10s} sigma={sig:.2f} ot={solver:20s} {csm}")
