import csv, glob, os
for name in ['620_ablation_dino_baseline_smoke','620_ablation_dino_adapter_smoke','620_ablation_intrinsic_latent_smoke']:
    files = glob.glob(f'exp/620_spatial_bridge/{name}/logs/training_*.csv')
    if not files:
        print(name, 'no log')
        continue
    files.sort(key=os.path.getmtime)
    latest = files[-1]
    with open(latest, newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        continue
    row = rows[-1]
    print(name, 'epoch_time_sec', row.get('epoch_time_sec'), 'compute_time_sec', row.get('compute_time_sec'), 'log', os.path.basename(latest))
