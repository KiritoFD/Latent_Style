"""Check pairing cache availability and T2a config."""
import os, json

cache_path = 'G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt'
print(f'4J.1 cache exists: {os.path.exists(cache_path)}')
if os.path.exists(cache_path):
    print(f'  size: {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB')

t2a_cfg = 'exp/630_local_t2_soft_ll_t2a/config.json'
if os.path.exists(t2a_cfg):
    with open(t2a_cfg) as f:
        t2a = json.load(f)
    data = t2a.get('data', {})
    train = t2a.get('training', {})
    print(f'T2a pairing_cache_topk: {data.get("pairing_cache_topk", "<MISSING>")}')
    print(f'T2a pairing_cache_active_topk: {data.get("pairing_cache_active_topk", "<MISSING>")}')
    print(f'T2a pairing_cache_path: {data.get("pairing_cache_path", "<MISSING>")}')
    print(f'T2a batch_size: {train.get("batch_size", "<MISSING>")}')

cache_dir = 'G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema/train/.latent_cache'
if os.path.exists(cache_dir):
    print(f'Cache dir contents:')
    for f in sorted(os.listdir(cache_dir)):
        size = os.path.getsize(os.path.join(cache_dir, f))
        print(f'  {f}: {size / 1024 / 1024:.1f} MB')
else:
    print(f'Cache dir does NOT exist: {cache_dir}')
    base = 'G:/GitHub/Latent_Style/Dataset'
    if os.path.exists(base):
        print(f'Searching under {base} for .latent_cache dirs...')
        for root, dirs, files in os.walk(base):
            if '.latent_cache' in dirs:
                full = os.path.join(root, '.latent_cache')
                print(f'  Found: {full}')
                for f in sorted(os.listdir(full)):
                    size = os.path.getsize(os.path.join(full, f))
                    print(f'    {f}: {size / 1024 / 1024:.1f} MB')
            # Don't recurse too deep
            depth = root[len(base):].count(os.sep)
            if depth >= 3:
                dirs[:] = []
