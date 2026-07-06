"""Extract aggregate metrics for photo2art 256 across all methods."""
import csv

# Ours per-image (already known)
ours_clip_s = 0.736
ours_lpips = 0.444

print("=== Per-image (this image: photo_2013-11-17 -> vangogh) ===")
print(f"  WD-VF (ours) : CLIP-S={ours_clip_s:.3f}  LPIPS={ours_lpips:.3f}")

# Aggregate from unified CSV
uni = "../baseline_metrics_unified.csv"
rows = []
with open(uni, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("protocol", "") == "protocol_a_800" and "photo2art" in row.get("source","").lower() or \
           row.get("baseline", "") == "s2wat_strict":
            rows.append(row)

# Also get all rows that look like photo2art
print(f"\n=== All protocol_a_800 / photo2art rows in unified CSV ===")
with open(uni, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = row.get("source", "")
        proto = row.get("protocol", "")
        method = row.get("method", "")
        if "photo2art" in src.lower() or "photo2art" in method.lower() or "256" in str(row.values()):
            cs = float(row["clip_style"]) if row.get("clip_style") else 0
            lp = float(row["content_lpips"]) if row.get("content_lpips") else 0
            print(f"  {method:<20s} | baseline={row['baseline']:<15s} | src={src:<20s} | CLIP-S={cs:.4f} LPIPS={lp:.4f}")

# Also check the exp JSON for samam/ours aggregates
import json
for fname, label in [("_ev_samam.json", "SaMam"), ("_ev_ours.json", "WD-VF(agg)")]:
    with open(fname) as f:
        data = json.load(f)
    top_key = list(data.keys())[0]
    val = data[top_key]
    if isinstance(val, dict):
        # Look for clip_style / lpips inside
        for k, v in val.items():
            if any(x in k.lower() for x in ["clip_s", "lpips", "style", "photo"]):
                print(f"\n  {label}/{top_key}/{k} = {v}")
        # Print first few keys
        subkeys = list(val.keys())[:10]
        print(f"\n{label} [{top_key}] subkeys: {subkeys}")
