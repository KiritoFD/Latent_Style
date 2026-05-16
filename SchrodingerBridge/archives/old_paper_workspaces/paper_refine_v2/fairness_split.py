"""Fairness split + bootstrap significance from per-image metrics.csv"""
import csv, numpy as np
from collections import defaultdict

path = 'G:/GitHub/Latent_Style/SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0/full_eval/epoch_0007/metrics.csv'
rows = []
with open(path) as f:
    for row in csv.DictReader(f):
        rows.append(row)

by_src = defaultdict(list)
for r in rows:
    by_src[r['src_style']].append(r)

print("=== Fairness Split (Ours epoch 7) ===")
print(f"{'Direction':<25} {'n':>5} {'CLIP-S':>8} {'LPIPS':>8} {'EC':>8}")
print("-" * 55)

for src in sorted(by_src):
    items = by_src[src]
    cs = np.mean([float(r['clip_style']) for r in items])
    lp = np.mean([float(r['content_lpips']) for r in items])
    print(f"{src+'->all':<25} {len(items):5d} {cs:8.4f} {lp:8.4f} {cs*(1-lp):8.4f}")

# photo->art
pa = [r for r in rows if r['src_style']=='photo' and r['tgt_style']!='photo']
cs_pa = np.mean([float(r['clip_style']) for r in pa])
lp_pa = np.mean([float(r['content_lpips']) for r in pa])
print(f"{'photo->art':<25} {len(pa):5d} {cs_pa:8.4f} {lp_pa:8.4f} {cs_pa*(1-lp_pa):8.4f}")

# non-identity
ni = [r for r in rows if r['src_style']!=r['tgt_style']]
cs_ni = np.mean([float(r['clip_style']) for r in ni])
lp_ni = np.mean([float(r['content_lpips']) for r in ni])
print(f"{'non-identity':<25} {len(ni):5d} {cs_ni:8.4f} {lp_ni:8.4f} {cs_ni*(1-lp_ni):8.4f}")

# Full bootstrap
n = len(rows)
lp_all = np.array([float(r['content_lpips']) for r in rows])
cs_all = np.array([float(r['clip_style']) for r in rows])
rng = np.random.RandomState(42)
lp_boot = [np.mean(rng.choice(lp_all, n, replace=True)) for _ in range(5000)]
cs_boot = [np.mean(rng.choice(cs_all, n, replace=True)) for _ in range(5000)]

print(f"\n=== Ours overall (n={n}, bootstrap 5000) ===")
print(f"LPIPS:  {np.mean(lp_all):.4f}  95%CI=[{np.percentile(lp_boot,2.5):.4f},{np.percentile(lp_boot,97.5):.4f}]")
print(f"CLIP-S: {np.mean(cs_all):.4f}  95%CI=[{np.percentile(cs_boot,2.5):.4f},{np.percentile(cs_boot,97.5):.4f}]")
print(f"EC:     {np.mean(cs_all)*(1-np.mean(lp_all)):.4f}")

# Per-target bootstrap for comparison
print("\n=== Per-target breakdown ===")
by_tgt = defaultdict(list)
for r in rows:
    by_tgt[r['tgt_style']].append(r)
print(f"{'Target':<10} {'n':>5} {'CLIP-S':>8} {'LPIPS':>8} {'EC':>8}")
for tgt in sorted(by_tgt):
    items = by_tgt[tgt]
    cs = np.mean([float(r['clip_style']) for r in items])
    lp = np.mean([float(r['content_lpips']) for r in items])
    print(f"{tgt:<10} {len(items):5d} {cs:8.4f} {lp:8.4f} {cs*(1-lp):8.4f}")
