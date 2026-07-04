"""Recursively explore summary.json structure for metric values."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/i/exp_our_models_eval/latent512_e7/summary.json"
with open(path) as f:
    d = json.load(f)


def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                walk(v, key)
            else:
                # Print leaf scalar values
                if any(t in k.lower() for t in ["clip", "lpips", "fid", "musiq", "art", "content", "n_pair", "n_img"]):
                    print(f"{key} = {v}")
    elif isinstance(obj, list):
        # Just print length
        if obj and isinstance(obj[0], (dict, list)):
            print(f"{prefix} (list[{len(obj)}])")
        else:
            print(f"{prefix} (list[{len(obj)}]) = {obj[:5]}")


walk(d)
