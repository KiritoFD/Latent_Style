"""Get all available metrics + check seedream eval."""
import json

# Get all fields from samam and ours eval JSONs
for fname, label in [("_ev_samam.json", "SaMam"), ("_ev_ours.json", "WD-VF")]:
    with open(fname) as f:
        data = json.load(f)
    top_key = list(data.keys())[0]
    val = data[top_key]
    print(f"\n=== {label} ({top_key}) ===")
    for k, v in val.items():
        if isinstance(v, float):
            print(f"  {k} : {v:.4f}")
        else:
            print(f"  {k} : {v}")

# Check baseline_pipeline metrics.csv for per-method
print("\n=== baseline_pipeline/results/metrics.csv ===")
import csv
with open(r"G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\metrics.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row}")

# Check if there's a seedream eval
import os
sd_dir = r"G:\GitHub\Latent_Style\seedream45_api"
for root, dirs, files in os.walk(sd_dir):
    for fn in files:
        fp = os.path.join(root, fn)
        if fn.endswith(('.json', '.csv')) and 'metric' in fn.lower() or 'eval' in fn.lower():
            print(f"\nFound: {fp}")
            if fn.endswith('.json'):
                with open(fp) as f2:
                    d = json.load(f2)
                print(f"  keys: {list(d.keys())[:10]}")
                for k, v in list(d.items())[:8]:
                    print(f"  {k}: {v}")
