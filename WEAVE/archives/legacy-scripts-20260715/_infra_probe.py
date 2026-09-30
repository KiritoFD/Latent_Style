import json, sys

# Training CSV
try:
    with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/t1_asg_5ep/logs/training_20260711_015903.csv') as f:
        lines = f.readlines()
    print("=== TRAINING CSV (first 3 + last 3 lines) ===")
    for line in lines[:3]:
        print(line.rstrip())
    print("...")
    for line in lines[-3:]:
        print(line.rstrip())
except Exception as e:
    print("train csv err:", e)

# Summary timings
try:
    d = json.load(open('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/t1_asg_5ep/full_eval/epoch_0005/summary.json'))
    print("\n=== SUMMARY KEYS ===")
    print(list(d.keys()))
    t = d.get('timings')
    if t:
        print("\n=== TIMINGS ===")
        print(json.dumps(t, indent=2))
    else:
        print("\nNo 'timings' key. Searching...")
        for k in d.keys():
            v = d[k]
            if isinstance(v, dict):
                for sk in v.keys():
                    if 'time' in sk.lower() or 'dur' in sk.lower():
                        print(f"  {k}.{sk} = {v[sk]}")
            if 'time' in k.lower() or 'dur' in k.lower():
                print(f"  {k} = {v}")
except Exception as e:
    print("summary err:", e)

# Config optimization flags
try:
    cfg = json.load(open('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/t1_asg_5ep/config.json'))
    tr = cfg.get('training', {})
    md = cfg.get('model', {})
    dt = cfg.get('data', {})
    print("\n=== TRAINING OPT FLAGS ===")
    for k in ['torch_compile','torch_compile_backend','torch_compile_mode','use_amp','amp_dtype','num_workers','persistent_workers','prefetch_factor','pin_memory','fused_adamw','allow_tf32','cudnn_benchmark','channels_last','use_gradient_checkpointing','gpu_monitor_enabled','cpu_threads']:
        print(f"  {k} = {tr.get(k)}")
    print("\n=== DATA OPT FLAGS ===")
    for k in ['preload_to_gpu','preload_max_vram_gb','latent_cache_mode','dino_cache_path','dino_cache_required']:
        print(f"  {k} = {dt.get(k)}")
    print("\n=== MODEL OPT FLAGS ===")
    for k in ['spectral_ode_enabled','use_checkpointing']:
        print(f"  {k} = {md.get(k)}")
    print("\n=== FULL_EVAL ===")
    fe = cfg.get('full_eval', {})
    for k in ['num_steps','batch_size','vae_compile_decoder','vae_onnx_decoder','vae_onnx_tensorrt']:
        print(f"  {k} = {fe.get(k)}")
except Exception as e:
    print("config err:", e)
