"""Extract metrics for photo_2013-11-17 16_40_10 -> vangogh across methods."""
import csv, json

TARGET_SRC = "2013-11-17 16_40_10"

# === 1) Ours: from metrics.csv (per-image) ===
ours_row = None
with open("_metrics_e10.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if TARGET_SRC in row["src_image"] and row["tgt_style"] == "vangogh":
            ours_row = row
            break

if ours_row:
    print("=== Ours (per-image) ===")
    print(f"  CLIP-S : {ours_row['clip_style']}")
    print(f"  LPIPS  : {ours_row['content_lpips']}")
    print(f"  CLIP-C : {ours_row['clip_content']}")
else:
    print("Ours not found in CSV")

# === 2) SaMam / Ours aggregate: from eval JSONs ===
for fname in ["_ev_samam.json", "_ev_ours.json"]:
    label = "SaMam" if "samam" in fname else "Ours(agg)"
    try:
        with open(fname, "r") as f:
            data = json.load(f)
        # Try to find photo->vangogh aggregate or per-image
        if isinstance(data, dict):
            # Check top-level keys
            keys = list(data.keys())[:10]
            print(f"\n=== {label} JSON keys sample: {keys} ===")
            # Look for photo_vangogh or similar
            for k in data:
                if "photo" in k.lower() and "vangogh" in k.lower() or \
                   "photo2art" in k.lower():
                    val = data[k]
                    if isinstance(val, dict):
                        print(f"  [{k}] : {list(val.items())[:8]}")
                    else:
                        print(f"  [{k}] = {val}")
    except Exception as e:
        print(f"{fname}: error - {e}")

# === 3) Check unified baseline CSV ===
import os
uni = "../baseline_metrics_unified.csv"
if os.path.exists(uni):
    print("\n=== baseline_metrics_unified.csv header ===")
    with open(uni, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(f"Columns: {reader.fieldnames[:15]}")
        for row in reader:
            if "photo2art" in str(row.values()).lower() or "256" in str(row.values()).lower():
                print(f"  Sample row: {dict(list(row.items())[:10])}")
                break

# Check local seedream/samam eval results too
sd_csv = r"G:\GitHub\Latent_Style\seedream45_api\protocol_a_800\metrics.csv"
samam_csv = r"G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\metrics.csv"
for p, label in [(sd_csv, "Seedream"), (samam_csv, "baseline_pipeline")]:
    if os.path.exists(p):
        print(f"\n=== {label}: {p} ===")
        with open(p, "r", encoding="utf-8") as f:
            hdr = f.readline().strip()
            print(f"  Header: {hdr[:200]}")
