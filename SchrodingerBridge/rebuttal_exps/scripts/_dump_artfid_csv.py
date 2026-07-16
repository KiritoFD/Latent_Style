"""Dump artfid_comparison_points.csv fully."""
from pathlib import Path

p = Path("I:/Github/Latent_Style/WEAVE/docs/experiments/comparison_20260602/artfid_comparison_points.csv")
text = p.read_text(encoding="utf-8", errors="ignore")
lines = text.strip().split("\n")
print(f"Total lines: {len(lines)}")
print(f"\nHeader:\n{lines[0]}\n")
print("=== All rows (formatted) ===")
import csv
reader = csv.DictReader(lines)
rows = list(reader)
print(f"Total rows: {len(rows)}\n")

# Group by dataset
by_ds = {}
for r in rows:
    ds = r["dataset"]
    by_ds.setdefault(ds, []).append(r)

for ds, ds_rows in by_ds.items():
    print(f"\n--- Dataset: {ds} ({len(ds_rows)} rows) ---")
    # Group by scope
    by_scope = {}
    for r in ds_rows:
        by_scope.setdefault(r["scope"], []).append(r)
    for scope, sc_rows in by_scope.items():
        print(f"\n  Scope: {scope} ({len(sc_rows)} rows)")
        for r in sc_rows:
            print(f"    method={r['method']}, label={r['label']}")
            print(f"      clip_style={r['clip_style']}, content_lpips={r['content_lpips']}")
            print(f"      aggregate_art_fid={r['aggregate_art_fid']}")
            print(f"      aggregate_art_fid_fid={r['aggregate_art_fid_fid']}")
            print(f"      aggregate_art_fid_content_lpips={r['aggregate_art_fid_content_lpips']}")
            print(f"      summary_path={r['summary_path']}")
            print(f"      artfid_path={r['artfid_path']}")
            print(f"      source_type={r['source_type']}, train_time={r['train_time_label']}")
            print()

# Also check existence of each summary_path
print("\n=== Existence check of summary_path and artfid_path ===")
import os
for r in rows:
    sp = r["summary_path"]
    ap = r["artfid_path"]
    sp_exists = os.path.exists(sp) if sp else False
    ap_exists = os.path.exists(ap) if ap else False
    if not sp_exists or not ap_exists:
        print(f"  MISSING: method={r['method']}, label={r['label']}, dataset={r['dataset']}, scope={r['scope']}")
        print(f"    summary_path={sp} [{sp_exists}]")
        print(f"    artfid_path={ap} [{ap_exists}]")
    else:
        print(f"  OK: {r['method']}/{r['label']} on {r['dataset']}/{r['scope']}")

# Also dump artfid_comparison_points.json structure
print("\n\n=== artfid_comparison_points.json structure ===")
import json
jp = Path("I:/Github/Latent_Style/WEAVE/docs/experiments/comparison_20260602/artfid_comparison_points.json")
d = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
if isinstance(d, list):
    print(f"list with {len(d)} items")
    if d:
        print(f"first item keys: {list(d[0].keys())}")
        print(f"first item: {d[0]}")
elif isinstance(d, dict):
    print(f"dict keys: {list(d.keys())}")
