import sys
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")
import torch
from model import build_model_from_config
from config_schema import load_experiment_config
from pathlib import Path

cfg = load_experiment_config(Path(r"I:\Github\Latent_Style\SchrodingerBridge\configs\refactor_clean_baseline.json").resolve())
m = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge)
ck = torch.load(r"I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\epoch_0005.pt", map_location="cpu", weights_only=False)
msg = m.load_state_dict(ck["model_state_dict"], strict=False)
print(f"Model class: {type(m).__name__}")
print(f"Missing: {len(msg.missing_keys)} Unexpected: {len(msg.unexpected_keys)}")
if msg.missing_keys:
    print(f"Missing keys (first 5): {msg.missing_keys[:5]}")
if msg.unexpected_keys:
    print(f"Unexpected keys (first 5): {msg.unexpected_keys[:5]}")
print(f"Total params: {sum(p.numel() for p in m.parameters()):,}")
print("CHECKPOINT_LOAD_OK")
