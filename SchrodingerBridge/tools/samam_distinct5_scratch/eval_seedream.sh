#!/usr/bin/env bash
SEEDREAM=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750

echo "===== 1. summary.json settings ====="
python3 -c "
import json
d = json.load(open('$SEEDREAM/summary.json'))
print('settings:', json.dumps(d.get('settings', {}), indent=2)[:2000])
print()
print('timings:', json.dumps(d.get('timings_sec', {}), indent=2)[:500])
"

echo ""
echo "===== 2. Compute overall mean from metrics.csv ====="
python3 -c "
import csv
rows = list(csv.DictReader(open('$SEEDREAM/metrics.csv')))
print(f'Rows: {len(rows)}')
n = len(rows)
clip_style = sum(float(r['clip_style']) for r in rows) / n
content_lpips = sum(float(r['content_lpips']) for r in rows) / n
clip_content = sum(float(r['clip_content']) for r in rows) / n
print(f'clip_style: {clip_style}')
print(f'content_lpips: {content_lpips}')
print(f'clip_content: {clip_content}')
# delta_idt vs 0.6933
print(f'delta_idt (vs 0.6933): {clip_style - 0.6933}')
"

echo ""
echo "===== 3. Compare with existing baselines ====="
echo "StyleID: clip_style=0.8223, lpips=0.5523"
echo "SDEdit0.40: clip_style=0.7934, lpips=0.4826"
echo "SeeDream: (from above)"

echo ""
echo "===== 4. Copy to baseline_v2/eval/seedream/ ====="
DEST=/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/seedream
mkdir -p "$DEST"
# Copy images, metrics.csv, summary.json
cp -r "$SEEDREAM/images" "$DEST/images" 2>/dev/null
cp "$SEEDREAM/metrics.csv" "$DEST/metrics.csv" 2>/dev/null
cp "$SEEDREAM/summary.json" "$DEST/summary.json" 2>/dev/null
echo "Copied to $DEST"
echo "Image count: $(ls $DEST/images/*.png 2>/dev/null | wc -l)"
echo "Files:"
ls -la "$DEST/" 2>/dev/null
