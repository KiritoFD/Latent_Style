import csv

rows = []
with open('_metrics_e10.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['src_style'] == 'photo' and row['tgt_style'] == 'vangogh':
            cs = float(row['clip_style'])
            lpips = float(row['content_lpips'])
            cc = float(row['clip_content'])
            rows.append({
                'src': row['src_image'],
                'gen': row['gen_image'],
                'clip_s': cs,
                'lpips': lpips,
                'clip_c': cc,
                'score': cs - abs(lpips - 0.25) * 0.5 + cc * 0.2,
            })

rows.sort(key=lambda x: x['score'], reverse=True)

print(f'Total photo->vangogh: {len(rows)}')
print()
print('=== TOP 15 by CLIP-S + balanced LPIPS ===')
hdr = '{:<5} {:<35} {:>8} {:>8} {:>8} {:>8}'.format('Rank', 'Source', 'CLIP-S', 'LPIPS', 'CLIP-C', 'Score')
print(hdr)
print('-' * len(hdr))
for i, r in enumerate(rows[:15]):
    print('{:<5} {:<35} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}'.format(
        i+1, r['src'], r['clip_s'], r['lpips'], r['clip_c'], r['score']))

print()
print('=== Top 10 pure CLIP-S (style strength) ===')
rows_cs = sorted(rows, key=lambda x: x['clip_s'], reverse=True)
for i, r in enumerate(rows_cs[:10]):
    print('  {}. {}  CS={:.4f}  LP={:.4f}  CC={:.4f}'.format(i+1, r['src'][:40], r['clip_s'], r['lpips'], r['clip_c']))
