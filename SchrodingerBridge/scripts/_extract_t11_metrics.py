"""Extract key metrics from t11_repro summary.json."""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

p = r"I:\Github\Latent_Style\SchrodingerBridge\exp\t11_repro_15ep\full_eval\epoch_0005\summary.json"
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)

print("=== TOP-LEVEL KEYS ===")
for k in d.keys():
    print(f"  {k}: {type(d[k]).__name__}")

# Print all scalar metrics at top level
print("\n=== SCALAR METRICS ===")
for k, v in d.items():
    if isinstance(v, (int, float, str, bool)) and not isinstance(v, bool):
        print(f"  {k} = {v}")
    elif isinstance(v, dict):
        # One level deep
        for k2, v2 in v.items():
            if isinstance(v2, (int, float)):
                print(f"  {k}.{k2} = {v2}")

# Look for clip, lpips, dino
print("\n=== KEY METRICS (filtered) ===")
def search_keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)):
                kl = full.lower()
                if any(x in kl for x in ["clip", "lpips", "dino", "musiq", "style", "content"]):
                    print(f"  {full} = {v}")
            elif isinstance(v, dict):
                search_keys(v, full)
    elif isinstance(obj, list) and len(obj) > 0:
        # check first item
        if isinstance(obj[0], dict):
            search_keys(obj[0], f"{prefix}[0]")
search_keys(d)
