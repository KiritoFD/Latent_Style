"""Generate same-LPIPS global-SWD control configs from the sem_r8 base.

The scientific control: take the GLOBAL SWD path (no semantic regions) and push
its distortion (via higher SWD weight + weaker content anchor) up to the LPIPS
level that semantic region SWD reaches (~0.38), to test whether the MUSIQ gain
is from semantic structure or merely from more distortion.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/semantic_swd_musiq/swd_cm_sem_r8.json"
OUT = ROOT / "configs/semantic_swd_musiq"

base = json.loads(BASE.read_text(encoding="utf-8"))

variants = {
    "swd_cm_ctrl_w24": {"single_step_swd_weight": 24.0, "w_endpoint_content": 0.5},
    "swd_cm_ctrl_w32": {"single_step_swd_weight": 32.0, "w_endpoint_content": 0.3},
}

for name, bridge_over in variants.items():
    cfg = json.loads(json.dumps(base))  # deep copy
    # Global SWD: turn OFF semantic region matching
    cfg["bridge"]["swd_semantic_mode"] = "off"
    for k, v in bridge_over.items():
        cfg["bridge"][k] = v
    cfg["checkpoint"]["save_dir"] = f"./exp/{name}"
    cfg["ablation"]["name"] = name
    cfg["ablation"]["notes"] = f"Global-SWD same-LPIPS control: {bridge_over}"
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"wrote {p} ({p.stat().st_size} bytes) swd_w={cfg['bridge']['single_step_swd_weight']} "
          f"w_ep={cfg['bridge']['w_endpoint_content']} sem={cfg['bridge']['swd_semantic_mode']}")
