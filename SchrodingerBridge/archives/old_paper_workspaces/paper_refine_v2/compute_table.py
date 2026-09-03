"""Compute S2WAT artifact metrics + fairness split table"""
import csv, os, json

# 1. Extract all core metrics from the summary CSV
csv_path = 'G:/GitHub/Latent_Style/Related_Works/run_511/complete_750/summary_all_tested_metrics_with_ablations.csv'
rows = []
with open(csv_path) as f:
    r = csv.DictReader(f)
    for row in r:
        if row['group'] == 'baseline':
            rows.append(row)

print("=== Core Metrics (from summary CSV) ===")
print(f"{'Method':<18} {'CLIP-S':>8} {'LPIPS':>8} {'EC':>8} {'Params':>10} {'Train(s)':>10} {'MUSIQ':>8} {'MANIQA':>8} {'DISTS':>8} {'HF-KID':>8}")
print("-"*105)
for row in rows:
    m = row['run']
    cs = float(row['clip_style'])
    lp = float(row['content_lpips']) if row.get('content_lpips') else float(row['lpips'])
    ec = cs * (1-lp)
    train = row.get('train_sec', '')
    mu = row.get('musiq', '')
    ma = row.get('maniqa', '')
    di = row.get('dists_content', '')
    hk = row.get('hf_patch_kid', '')
    print(f"{m:<18} {cs:8.4f} {lp:8.4f} {ec:8.4f} {'':>10} {train:>10} {mu:>8} {ma:>8} {di:>8} {hk:>8}")
