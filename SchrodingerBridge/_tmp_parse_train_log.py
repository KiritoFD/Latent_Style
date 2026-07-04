import csv, pathlib, glob
for name in ['620_ablation_dino_baseline_smoke','620_ablation_dino_adapter_smoke','620_ablation_intrinsic_latent_smoke']:
    files = glob.glob(f'exp/620_spatial_bridge/{name}/logs/training_*.csv')
    if not files:
        print(name, 'no log')
        continue
    with open(files[0], newline='') as f:
        r = csv.reader(f)
        header = next(r)
        row = next(r)
    print(name)
    for i, h in enumerate(header):
        if any(k in h.lower() for k in ['time', 'wall', 'sec', 'epoch', 'step', 'total']):
            print(i, h, row[i])
    print()
