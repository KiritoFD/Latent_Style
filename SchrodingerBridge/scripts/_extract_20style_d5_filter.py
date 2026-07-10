import csv, os

# D5 style names (target styles we care about)
d5_styles = {'Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e'}

p = 'exp/630_random20_heun_5ep/full_eval/d5_test/metrics.csv'
if not os.path.exists(p):
    print('CSV_NOT_FOUND')
    exit()

rows = list(csv.DictReader(open(p, encoding='utf-8')))
print(f'Total rows: {len(rows)}')

# Filter to only D5 target styles
d5_rows = [x for x in rows if x['tgt_style'] in d5_styles]
print(f'D5 target rows: {len(d5_rows)}')

# Also filter to D5 source styles (should already be D5 since test_dir is D5)
d5_src_rows = [x for x in d5_rows if x['src_style'] in d5_styles]
print(f'D5 src+tgt rows: {len(d5_src_rows)}')

n_all = len(d5_src_rows)
off = [x for x in d5_src_rows if x['src_style'] != x['tgt_style']]
n_off = len(off)

all_clip = sum(float(x['clip_style']) for x in d5_src_rows) / n_all
all_lpips = sum(float(x['content_lpips']) for x in d5_src_rows) / n_all
off_clip = sum(float(x['clip_style']) for x in off) / n_off
off_lpips = sum(float(x['content_lpips']) for x in off) / n_off

print(f'20style_d5_filtered: n_all={n_all} n_off={n_off} all_clip={all_clip:.4f} all_lpips={all_lpips:.4f} off_clip={off_clip:.4f} off_lpips={off_lpips:.4f}')

# Also show per-target-style breakdown
for tgt in sorted(d5_styles):
    tgt_rows = [x for x in d5_src_rows if x['tgt_style'] == tgt]
    if tgt_rows:
        tgt_clip = sum(float(x['clip_style']) for x in tgt_rows) / len(tgt_rows)
        tgt_lpips = sum(float(x['content_lpips']) for x in tgt_rows) / len(tgt_rows)
        print(f'  tgt={tgt}: n={len(tgt_rows)} clip={tgt_clip:.4f} lpips={tgt_lpips:.4f}')
