"""Check new metrics.csv after re-eval."""
import csv
import os
from pathlib import Path

csv_path = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_t11\full_eval\epoch_0005\metrics.csv")
print(f"CSV exists: {csv_path.exists()}, mtime: {os.path.getmtime(csv_path) if csv_path.exists() else 'N/A'}")

with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 3:
            break
        src_img = row['src_image']
        gen_img = row['gen_image']
        print(f"Row {i}: src_image={src_img!r}, gen_image={gen_img!r}")
        # Check actual file existence
        test_dir = Path(r"I:\datasets\wikiart_distinct5_samam_512_classview\test")
        direct = test_dir / row['src_style'] / src_img
        prefixed = test_dir / row['src_style'] / f"{row['src_style']}__{src_img}"
        print(f"  Direct: {direct.exists()} -> {direct}")
        print(f"  Prefixed: {prefixed.exists()} -> {prefixed}")
        # Check gen image
        eval_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_t11\full_eval\epoch_0005")
        gen_path = eval_dir / gen_img
        print(f"  Gen: {gen_path.exists()} -> {gen_path}")
