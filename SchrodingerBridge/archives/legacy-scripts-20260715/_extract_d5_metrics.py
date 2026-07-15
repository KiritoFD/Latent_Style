import csv, os
dirs = {
    'baseline': 'exp/evo_d5_baseline/full_eval/epoch_0005/metrics.csv',
    'adain10': 'exp/evo_d5_adain10/full_eval/epoch_0005/metrics.csv',
    'extrap02': 'exp/evo_d5_extrap02/full_eval/epoch_0005/metrics.csv',
    'long10_ep10': 'exp/evo_d5_long10/full_eval/epoch_0010/metrics.csv',
    'combo_ep5': 'exp/evo_d5_combo/full_eval/epoch_0005/metrics.csv',
    '20style_d5eval': 'exp/630_random20_heun_5ep/full_eval/d5_test/metrics.csv',
}
for k, p in dirs.items():
    if not os.path.exists(p):
        print(f'{k}: CSV_NOT_FOUND')
        continue
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    n_all = len(rows)
    off = [x for x in rows if x['src_style'] != x['tgt_style']]
    n_off = len(off)
    all_clip = sum(float(x['clip_style']) for x in rows) / n_all
    all_lpips = sum(float(x['content_lpips']) for x in rows) / n_all
    off_clip = sum(float(x['clip_style']) for x in off) / n_off
    off_lpips = sum(float(x['content_lpips']) for x in off) / n_off
    print(f'{k}: n_all={n_all} n_off={n_off} all_clip={all_clip:.4f} all_lpips={all_lpips:.4f} off_clip={off_clip:.4f} off_lpips={off_lpips:.4f}')
