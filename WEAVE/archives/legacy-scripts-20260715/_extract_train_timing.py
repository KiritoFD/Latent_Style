import csv, glob, os

files = sorted(glob.glob('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/infra_train_opt/logs/training_*.csv'))
if not files:
    print("No CSV found")
    exit(1)
f = files[-1]
print(f"=== {os.path.basename(f)} ===")
with open(f) as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        epoch = row.get('epoch', '?')
        epoch_time = float(row.get('epoch_time_sec', 0))
        samples_per_sec = float(row.get('samples_per_sec', 0))
        vram_alloc = float(row.get('cuda_peak_allocated_gb', 0))
        vram_reserved = float(row.get('cuda_peak_reserved_gb', 0))
        gpu_util = float(row.get('gpu_util_mean', 0)) if row.get('gpu_util_mean') else 0
        avg_batch = float(row.get('avg_batch_time_sec', 0))
        avg_fwd = float(row.get('avg_forward_time_sec', 0))
        avg_bwd = float(row.get('avg_backward_time_sec', 0))
        avg_opt = float(row.get('avg_optimizer_time_sec', 0))
        print(f"epoch={epoch}  epoch_time={epoch_time:.1f}s  samples/s={samples_per_sec:.1f}  batch={avg_batch:.3f}s  fwd={avg_fwd:.4f}s  bwd={avg_bwd:.4f}s  opt={avg_opt:.4f}s  VRAM_alloc={vram_alloc:.2f}GB  VRAM_reserved={vram_reserved:.2f}GB  gpu_util={gpu_util:.1f}%")

# Compare with baseline (t1_asg_5ep)
print("\n=== BASELINE (t1_asg_5ep) ===")
base_files = sorted(glob.glob('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/t1_asg_5ep/logs/training_*.csv'))
if base_files:
    with open(base_files[-1]) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            epoch = row.get('epoch', '?')
            epoch_time = float(row.get('epoch_time_sec', 0))
            samples_per_sec = float(row.get('samples_per_sec', 0))
            vram_alloc = float(row.get('cuda_peak_allocated_gb', 0))
            avg_batch = float(row.get('avg_batch_time_sec', 0))
            print(f"epoch={epoch}  epoch_time={epoch_time:.1f}s  samples/s={samples_per_sec:.1f}  batch={avg_batch:.3f}s  VRAM_alloc={vram_alloc:.2f}GB")
