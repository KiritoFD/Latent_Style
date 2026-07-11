"""Inspect summary.json structure to find correct metric keys."""
import json
import os

summ = r"I:\Github\Latent_Style\SchrodingerBridge\exp\abl_swd_to_mse\full_eval\epoch_0005\summary.json"
with open(summ, "r") as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))
for k, v in data.items():
    if isinstance(v, dict):
        print(f"\n[{k}] keys: {list(v.keys())}")
        for sk, sv in v.items():
            if isinstance(sv, (int, float)):
                print(f"  {sk} = {sv}")
            elif isinstance(sv, dict):
                print(f"  {sk} (dict): {list(sv.keys())}")
            elif isinstance(sv, list) and len(sv) <= 5:
                print(f"  {sk} (list len={len(sv)}): {sv}")
    elif isinstance(v, (int, float)):
        print(f"{k} = {v}")
    elif isinstance(v, str) and len(v) < 100:
        print(f"{k} = {v}")
