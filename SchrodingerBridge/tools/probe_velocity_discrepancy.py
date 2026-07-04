#!/usr/bin/env python3
"""Probe velocity discrepancy between training and eval.
Diagnose why training |v| (0.468) >> eval velocity_abs (0.189).
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any

import torch
import numpy as np

_SRC = "/mnt/i/Github/Latent_Style/SchrodingerBridge/src"
os.chdir(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config_schema import ExperimentConfig, load_experiment_config
from model620 import SpatialBridge620, build_spatial_bridge620_from_config
from utils.dataset import AdaCUTLatentDataset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke/epoch_0005.pt")
parser.add_argument("--config", type=str, default="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke/config.json")
parser.add_argument("--n_samples", type=int, default=4)
args = parser.parse_args()

CKPT = args.checkpoint
CONFIG = args.config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Loading config from {CONFIG}")
cfg = load_experiment_config(CONFIG)
print(f"  gate: {cfg.model.style_cross_attn_gate_init}")
print(f"  swd_noise_sigma: {cfg.bridge.swd_noise_sigma}")
print(f"  endpoint_head_mode: {cfg.model.endpoint_head_mode}")

print(f"\nLoading model from {CKPT}")
model = build_spatial_bridge620_from_config(cfg.model, bridge_cfg=cfg.bridge)
model.to(DEVICE)
model.eval()

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state, strict=True)
print("Model loaded")

# Load dataset
print("\nLoading dataset...")
dataset = AdaCUTLatentDataset(
    data_root=cfg.data.data_root,
    style_subdirs=cfg.data.style_subdirs,
    allow_hflip=False,
    identity_ratio=cfg.data.identity_ratio,
    balance_target_styles_per_batch=cfg.data.balance_target_styles_per_batch,
    pairing_cache_path=cfg.data.pairing_cache_path,
    pairing_cache_topk=cfg.data.pairing_cache_topk,
    pairing_cache_active_topk=cfg.data.pairing_cache_active_topk,
    pairing_cache_cross_only=cfg.data.pairing_cache_cross_only,
    dino_cache_path=cfg.data.dino_cache_path,
)
print(f"Dataset size: {len(dataset)}")

# Get a batch of samples
batch_size = 8
indices = list(range(min(batch_size, len(dataset))))
items = [dataset[i] for i in indices]
batch = {}
for key in items[0].keys():
    values = [item[key] for item in items]
    if isinstance(values[0], torch.Tensor):
        batch[key] = torch.stack(values).to(DEVICE)
    else:
        batch[key] = values

print(f"Dataset keys: {list(batch.keys())}")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

# Find the right keys
# The dataset likely returns latent data under specific keys
# Let's check what keys are available
x = None
y = None
style_latent = None

# Try different key names
for key in batch:
    if 'source' in key.lower() and 'latent' in key.lower():
        x = batch[key]
    elif 'target' in key.lower() and 'latent' in key.lower():
        y = batch[key]
    elif 'style' in key.lower() and 'latent' in key.lower():
        if style_latent is None:
            style_latent = batch[key]

if x is None:
    # Try alternative keys
    for key in batch:
        if isinstance(batch[key], torch.Tensor) and batch[key].ndim == 4:
            print(f"  Found 4D tensor: {key} shape={batch[key].shape}")
            if x is None:
                x = batch[key]
            elif y is None:
                y = batch[key]

if x is None or y is None:
    print("ERROR: Could not find source/target latents")
    print("Available keys:", list(batch.keys()))
    sys.exit(1)

print(f"\nx shape: {x.shape}, y shape: {y.shape}")
style_dino_cls = batch.get("target_style_dino_cls")
style_dino_patches = batch.get("target_style_dino_patches")
print(f"style_dino_cls: {style_dino_cls.shape if style_dino_cls is not None else 'N/A'}")
print(f"style_dino_patches: {style_dino_patches.shape if style_dino_patches is not None else 'N/A'}")

# Check gate value
gate_vals = []
for block in model.blocks:
    if hasattr(block, 'style_gate'):
        gate_vals.append(torch.tanh(block.style_gate).item())
print(f"Gate values per block: {[f'{g:.4f}' for g in gate_vals]}")
print(f"Gate mean: {np.mean(gate_vals):.4f}")

# Test velocity at different t values
t_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
print("\n" + "=" * 70)
print(f"{'t':>6} | {'|v|_mean':>10} | {'|v|_std':>10} | {'|v_target|':>12} | {'alpha':>8} | {'gate':>8}")
print("-" * 70)

for t_val in t_values:
    t = torch.full((x.shape[0],), t_val, device=DEVICE, dtype=torch.float32)
    
    with torch.no_grad():
        velocity = model(x, t=t, style_dino_cls=style_dino_cls, style_dino_patches=style_dino_patches)
    
    v_abs = velocity.abs().mean(dim=[1, 2, 3])  # [B]
    v_abs_mean = v_abs.mean().item()
    v_abs_std = v_abs.std().item()
    
    # Compute target velocity: v_target = (y - x_t) / (1-t)
    t_reshaped = t.view(-1, 1, 1, 1)
    x_t = (1 - t_reshaped) * x + t_reshaped * y
    if t_val < 1.0:
        v_target = (y - x_t) / (1 - t_val)
    else:
        v_target = torch.zeros_like(x)
    v_target_abs = v_target.abs().mean(dim=[1, 2, 3]).mean().item()
    
    alpha = v_abs_mean / (v_target_abs + 1e-8) if v_target_abs > 0 else 0
    gate = model.last_debug.get("style_gate_value", torch.tensor([0.0])).item()
    
    print(f"{t_val:>6.2f} | {v_abs_mean:>10.4f} | {v_abs_std:>10.4f} | {v_target_abs:>12.4f} | {alpha:>8.3f} | {gate:>8.4f}")

# Also check training-level velocity_abs
print(f"\nTraining-level velocity_abs (from model.last_debug): {model.last_debug.get('velocity_abs', 'N/A')}")

# Style sensitivity test
print("\n\n=== Style sensitivity test ===")
t_fixed = 0.5
t = torch.full((x.shape[0],), t_fixed, device=DEVICE, dtype=torch.float32)

with torch.no_grad():
    velocity = model(x, t=t, style_dino_cls=style_dino_cls, style_dino_patches=style_dino_patches)
v1 = velocity

# Shuffle style DINO features to test style sensitivity
perm = torch.randperm(style_dino_cls.shape[0])
style_dino_cls_shuffled = style_dino_cls[perm]
style_dino_patches_shuffled = style_dino_patches[perm]
with torch.no_grad():
    velocity2 = model(x, t=t, style_dino_cls=style_dino_cls_shuffled, style_dino_patches=style_dino_patches_shuffled)
v2 = velocity2

v_diff = (v1 - v2).abs().mean(dim=[1, 2, 3])
v_norm = v1.abs().mean(dim=[1, 2, 3])
cos_sim = torch.nn.functional.cosine_similarity(
    v1.flatten(1), v2.flatten(1), dim=1
)

print(f"v_diff (shuffled style): {v_diff.mean().item():.4f} +/- {v_diff.std().item():.4f}")
print(f"v_norm: {v_norm.mean().item():.4f}")
print(f"v_diff / v_norm: {(v_diff / (v_norm + 1e-8)).mean().item():.4f}")
print(f"cos_sim(v1, v2): {cos_sim.mean().item():.4f}")
print(f"  -> If cos_sim ~ 1, model ignores style (collapse)")
print(f"  -> If cos_sim < 0.9, model uses style")

print("\nDone!")