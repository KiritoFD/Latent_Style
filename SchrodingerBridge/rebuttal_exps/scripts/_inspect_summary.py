"""Quickly inspect an eval summary.json."""
import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else r"g:\GitHub\Latent_Style\SchrodingerBridge\weave_gen\exp\rebuttal\task1_sdxl_retrain\epoch_0015\summary.json"
d = json.load(open(p, encoding="utf-8"))
print("top keys:", list(d.keys()))
a = d.get("analysis", {})
print("analysis keys:", list(a.keys()))
for k, v in a.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for kk, vv in v.items():
            print(f"    {kk}: {vv}")
    else:
        print(f"  {k}: {v}")
