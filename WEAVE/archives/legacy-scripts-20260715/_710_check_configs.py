"""Check B1-B8 config key fields."""
import json
import glob
from pathlib import Path

cfg_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\configs")
for f in sorted(glob.glob(str(cfg_dir / "710_b*.json"))):
    cfg = json.load(open(f))
    name = Path(f).stem
    save_dir = cfg.get("checkpoint", {}).get("save_dir", "N/A")
    base = cfg.get("_base", "N/A")
    model = cfg.get("model", {})
    bridge = cfg.get("bridge", {})
    print(f"\n=== {name} ===")
    print(f"  _base: {base}")
    print(f"  save_dir: {save_dir}")
    for k in ["dwt_route_train_prob", "cross_attn_dwt_route", "endpoint_adain_mode",
              "num_res_blocks", "base_dim"]:
        if k in model:
            print(f"  model.{k}: {model[k]}")
    for k in ["spectral_w_ll"]:
        if k in bridge:
            print(f"  bridge.{k}: {bridge[k]}")
