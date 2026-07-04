"""Check actual image sizes in test dataset, SaMam output, and T11 output."""
from pathlib import Path
from PIL import Image
import os

# Test dataset
test_dir = Path(r"I:\wikiart_distinct5_samam_512_classview\test")
# SaMam output
samam_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam\images")
# T11 output
t11_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\full_eval\epoch_0001\images")
# Latents dataset
latents_dir = Path(r"I:\wikiart_distinct5_latents_512_ema")

def check_sizes(label, directory, pattern="*.png", limit=20):
    print(f"\n=== {label}: {directory} ===")
    if not directory.exists():
        print(f"  NOT FOUND")
        return
    files = sorted(directory.rglob(pattern)) if directory.is_dir() else []
    if not files:
        # Try other patterns
        files = sorted(directory.rglob("*"))
        files = [f for f in files if f.is_file()][:limit]
    print(f"  Total files found: {len(files)}")
    sizes = {}
    for f in files[:limit]:
        try:
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
                img = Image.open(f)
                sizes[img.size] = sizes.get(img.size, 0) + 1
            elif f.suffix.lower() in ('.pt', '.pth'):
                import torch
                t = torch.load(f, map_location='cpu', weights_only=False)
                if isinstance(t, dict):
                    for k, v in list(t.items())[:2]:
                        if hasattr(v, 'shape'):
                            print(f"    {f.name}[{k}]: shape={v.shape}")
                elif hasattr(t, 'shape'):
                    print(f"    {f.name}: shape={t.shape}")
        except Exception as e:
            pass
    if sizes:
        print(f"  Image sizes (first {limit}):")
        for sz, cnt in sorted(sizes.items()):
            print(f"    {sz}: {cnt}")

# Check test dataset
check_sizes("Test dataset", test_dir, "*.png")

# Check test dataset subdirectories
if test_dir.exists():
    print(f"\n  Test dir structure:")
    for item in sorted(test_dir.iterdir())[:5]:
        if item.is_dir():
            sub_files = list(item.glob("*"))
            print(f"    {item.name}/: {len(sub_files)} files")
            if sub_files:
                try:
                    img = Image.open(sub_files[0])
                    print(f"      Sample: {sub_files[0].name} size={img.size}")
                except:
                    pass

# Check SaMam
check_sizes("SaMam output", samam_dir, "*.png")

# Check T11
check_sizes("T11 output", t11_dir, "*.png")

# Check latents
if latents_dir.exists():
    print(f"\n=== Latents dataset: {latents_dir} ===")
    for item in sorted(latents_dir.iterdir())[:3]:
        print(f"  {item.name}/")
        if item.is_dir():
            sub = list(item.iterdir())[:3]
            for s in sub:
                print(f"    {s.name}")
                if s.suffix == '.pt':
                    import torch
                    t = torch.load(s, map_location='cpu', weights_only=False)
                    if isinstance(t, dict):
                        for k, v in list(t.items())[:3]:
                            if hasattr(v, 'shape'):
                                print(f"      [{k}]: shape={v.shape}")
                    elif hasattr(t, 'shape'):
                        print(f"      shape={t.shape}")
