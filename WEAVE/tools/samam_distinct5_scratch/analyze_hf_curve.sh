#!/usr/bin/env bash
CSV=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_hf_750_batched/curve_metrics.csv

echo "=== Header ==="
head -1 "$CSV"

echo ""
echo "=== Key checkpoints (every 2500 steps) ==="
python3 -c "
import csv
rows = list(csv.DictReader(open('$CSV')))
print(f'Total rows: {len(rows)}')
# Fix step parsing (step=-1 is a bug, parse from image_dir)
for r in rows:
    if r['step'] == '-1':
        import re
        m = re.search(r'step_(\d+)', r.get('image_dir', ''))
        if m:
            r['step'] = m.group(1)
    elif r['step'] == '1000000000000':
        r['step'] = 'last'
# Sort by step
def sort_key(r):
    s = r['step']
    return int(s) if s != 'last' else 10**12
rows.sort(key=sort_key)

print('')
print('=== Full curve (every 2500 steps) ===')
print(f'{\"step\":>8} {\"clip_style\":>11} {\"lpips\":>8} {\"clip_content\":>12}')
for r in rows:
    step = r['step']
    if step == 'last' or int(step) % 2500 == 0 or int(step) == 250:
        print(f'{step:>8} {float(r[\"clip_style\"]):>11.4f} {float(r[\"content_lpips\"]):>8.4f} {float(r[\"clip_content\"]):>12.4f}')

print('')
print('=== Convergence analysis ===')
# Find where clip_style stabilizes
clip_values = [(int(r['step']) if r['step'] != 'last' else 20000, float(r['clip_style'])) for r in rows if r['step'] != 'last']
clip_values.sort()
for i in range(1, len(clip_values)):
    delta = abs(clip_values[i][1] - clip_values[i-1][1])
    if delta < 0.01 and i > 10:
        print(f'CLIP-S first stabilizes (Δ<0.01) at step {clip_values[i][0]}: {clip_values[i][1]:.4f}')
        break

# Last 5 checkpoints
print('')
print('=== Last 5 checkpoints ===')
for r in rows[-6:]:
    print(f'{r[\"step\"]:>8} clip_s={float(r[\"clip_style\"]):.4f} lpips={float(r[\"content_lpips\"]):.4f}')

# Best
print('')
best_clip = max(rows, key=lambda r: float(r['clip_style']))
best_lpips = min(rows, key=lambda r: float(r['content_lpips']))
print(f'Best CLIP-S: step={best_clip[\"step\"]} = {float(best_clip[\"clip_style\"]):.4f}')
print(f'Best LPIPS:  step={best_lpips[\"step\"]} = {float(best_lpips[\"content_lpips\"]):.4f}')
"
