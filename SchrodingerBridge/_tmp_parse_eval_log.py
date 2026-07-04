import csv, glob
for name in ['620_ablation_dino_baseline_smoke','620_ablation_dino_adapter_smoke','620_ablation_intrinsic_latent_smoke']:
    files = glob.glob(f'exp/620_spatial_bridge/{name}/logs/full_eval_runtime.csv')
    if not files:
        print(name, 'no eval log')
        continue
    with open(files[0], newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        continue
    row = rows[-1]
    print(name, 'wall_sec', row.get('wall_sec'), 'summary_wall_total_sec', row.get('summary_wall_total_sec'))
