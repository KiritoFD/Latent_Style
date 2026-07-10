"""Inspect summary.json structure to find correct field names."""
import json
import os

base = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
path = os.path.join(base, "d1_gram_hf1_15ep", "full_eval", "epoch_0015", "summary.json")
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)

print("TOP-LEVEL KEYS:", list(d.keys()))
for k in d.keys():
    v = d[k]
    if isinstance(v, dict):
        print(f"\n[{k}] sub-keys:", list(v.keys()))
        for sk, sv in v.items():
            if isinstance(sv, (int, float)):
                print(f"  {k}.{sk} = {sv}")
            elif isinstance(sv, str) and len(sv) < 80:
                print(f"  {k}.{sk} = {sv}")
    elif isinstance(v, (int, float)):
        print(f"{k} = {v}")
    elif isinstance(v, str) and len(v) < 80:
        print(f"{k} = {v}")
    elif isinstance(v, list):
        print(f"{k} = list[{len(v)}]")
