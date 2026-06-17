import json, os

batch_dir = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20260617_1749_ot_vertical_sweep"
for name in sorted(os.listdir(batch_dir)):
    cfg_path = os.path.join(batch_dir, name, "config.json")
    if not os.path.exists(cfg_path): continue
    c = json.load(open(cfg_path))
    c["training"]["batch_size"] = 16
    c["training"]["virtual_length_multiplier"] = 0.1
    json.dump(c, open(cfg_path, "w"), indent=2)
    print(f"Set {name}: b=16, vl=0.1")
