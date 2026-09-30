"""Inspect 710 B0 metrics.csv first 5 rows (first 5 cols only)."""
import csv
from pathlib import Path

csv_path = Path(
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/710_b0_t11/full_eval/epoch_0005/metrics.csv"
)
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print("Fields:", reader.fieldnames[:6])
    for i, row in enumerate(reader):
        if i >= 5:
            break
        print(f"Row {i}: src_style={row['src_style']!r}, tgt_style={row['tgt_style']!r}, "
              f"src_image={row['src_image']!r}, gen_image={row['gen_image']!r}")

# Check test_dir file naming
import os
test_dir = Path("/mnt/i/datasets/wikiart_distinct5_samam_512_classview/test/Early_Renaissance")
files = sorted(os.listdir(test_dir))[:3]
print(f"\nTest dir files (first 3): {files}")

# Check if CSV src_image matches any file in test_dir
csv_src = "fra-angelico_saint-cosmas-and-saint-damian-before-lisius-1440.jpg"
direct = test_dir / csv_src
prefixed = test_dir / f"Early_Renaissance__{csv_src}"
print(f"\nDirect path exists? {direct.exists()}: {direct}")
print(f"Prefixed path exists? {prefixed.exists()}: {prefixed}")
