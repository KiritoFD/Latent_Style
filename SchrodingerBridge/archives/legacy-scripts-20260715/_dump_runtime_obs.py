"""Dump runtime_observability from a summary.json."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep/full_eval/epoch_0011/summary.json"
)
with open(path) as f:
    d = json.load(f)
ro = d.get("runtime_observability", {})
print("=== runtime_observability keys ===")
for k in sorted(ro.keys()):
    v = ro[k]
    if isinstance(v, float):
        print(f"  {k} = {v:.6f}")
    else:
        print(f"  {k} = {v}")
