"""Fix metrics.csv gen_image column to add 'images/' prefix for compute_dino_metrics.py compatibility."""
import csv
import sys
from pathlib import Path

for csv_path in sys.argv[1:]:
    p = Path(csv_path)
    rows = list(csv.reader(p.open(encoding="utf-8-sig")))
    if not rows:
        continue
    header = rows[0]
    if "gen_image" not in header:
        continue
    gen_idx = header.index("gen_image")
    fixed = 0
    for row in rows[1:]:
        if not row[gen_idx].startswith("images/"):
            row[gen_idx] = "images/" + row[gen_idx]
            fixed += 1
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Fixed {fixed} rows in {csv_path}")
