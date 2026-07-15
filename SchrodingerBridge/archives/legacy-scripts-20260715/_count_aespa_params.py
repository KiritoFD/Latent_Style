"""Inspect AesPA-Net checkpoint structure and count params."""
import torch
from pathlib import Path

paths = [
    r"I:\AesPA-Net\train_results\aespa\log\dec_model_.pth",
    r"I:\AesPA-Net\train_results\aespa\log\transformer_model_.pth",
    r"I:\AesPA-Net\baseline_checkpoints\vgg_normalised_conv5_1.pth",
]

def count_params(obj, prefix=""):
    """Recursively count parameters in nested structures."""
    if hasattr(obj, "numel"):
        return obj.numel()
    if isinstance(obj, dict):
        return sum(count_params(v, f"{prefix}.{k}") for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return sum(count_params(v, f"{prefix}[{i}]") for i, v in enumerate(obj))
    return 0

total = 0
for p in paths:
    if not Path(p).exists():
        print(f"MISSING: {p}")
        continue
    sd = torch.load(p, map_location="cpu")
    print(f"\n=== {Path(p).name} ===")
    print(f"Type: {type(sd).__name__}")
    if isinstance(sd, dict):
        print(f"Top-level keys: {list(sd.keys())[:10]}")
        for k, v in sd.items():
            if isinstance(v, dict):
                n = count_params(v)
                print(f"  '{k}': dict with {len(v)} entries, {n:,} params")
            elif hasattr(v, "numel"):
                print(f"  '{k}': tensor {v.shape}, {v.numel():,} params")
            else:
                print(f"  '{k}': {type(v).__name__}")
        n = count_params(sd)
        print(f"  TOTAL params in this file: {n:,} = {n/1e6:.2f}M")
        total += n
    else:
        print(f"Not a dict: {sd}")

print(f"\n=== GRAND TOTAL: {total:,} = {total/1e6:.2f}M ===")

# VGG (frozen)
vgg_path = paths[2]
if Path(vgg_path).exists():
    vgg_sd = torch.load(vgg_path, map_location="cpu")
    vgg_n = count_params(vgg_sd)
    print(f"VGG (frozen): {vgg_n:,} = {vgg_n/1e6:.2f}M")
    print(f"TRAINABLE (excl VGG): {(total-vgg_n):,} = {(total-vgg_n)/1e6:.2f}M")
