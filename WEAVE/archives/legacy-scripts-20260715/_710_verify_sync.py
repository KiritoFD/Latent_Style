import sys
path = r"I:\Github\Latent_Style\SchrodingerBridge\src\utils\compute_dino_metrics.py"
s = open(path, encoding="utf-8").read()
print("HAS_PREFIXED:", "prefixed" in s)
print("HAS_OFF_DS_FIX:", "sum(ds for _, ds, _" in s)
print("HAS_OFF_DC_FIX:", "sum(dc for _, _, dc" in s)
