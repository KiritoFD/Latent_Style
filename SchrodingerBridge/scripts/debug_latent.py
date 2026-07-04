import torch
import os

# Compare original latent vs new latent256
orig_dir = r"I:/wikiart_distinct5_samam_512_latents_ema/train/Early_Renaissance"
new_dir = r"I:/wikiart_distinct5_samam_512_latent256/train/Early_Renaissance"

# Find first file in each
orig_files = sorted([f for f in os.listdir(orig_dir) if f.endswith('.pt')])
new_files = sorted([f for f in os.listdir(new_dir) if f.endswith('.pt')])

print(f"Original dir: {len(orig_files)} files")
print(f"Latent256 dir: {len(new_files)} files")

if orig_files:
    a = torch.load(os.path.join(orig_dir, orig_files[0]), map_location='cpu', weights_only=False)
    print(f"\nOriginal latent ({orig_files[0]}):")
    if torch.is_tensor(a):
        print(f"  shape={a.shape}, dtype={a.dtype}")
        print(f"  min={a.min().item():.4f}, max={a.max().item():.4f}, mean={a.mean().item():.4f}, std={a.std().item():.4f}")
    else:
        print(f"  type={type(a)}, keys={list(a.keys()) if isinstance(a, dict) else 'N/A'}")

if new_files:
    b = torch.load(os.path.join(new_dir, new_files[0]), map_location='cpu', weights_only=False)
    print(f"\nLatent256 ({new_files[0]}):")
    if torch.is_tensor(b):
        print(f"  shape={b.shape}, dtype={b.dtype}")
        print(f"  min={b.min().item():.4f}, max={b.max().item():.4f}, mean={b.mean().item():.4f}, std={b.std().item():.4f}")
    else:
        print(f"  type={type(b)}, keys={list(b.keys()) if isinstance(b, dict) else 'N/A'}")
