import torch, transformers
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("vram:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
print("transformers:", transformers.__version__)
from PIL import Image
print("PIL ok")
import pathlib

# Check DINOv2 cache
cache = pathlib.Path(r"I:\Github\Latent_Style\WEAVE\eval_cache\hub\models--facebook--dinov2-small\snapshots")
if cache.exists():
    revs = [p for p in cache.iterdir() if p.is_dir()]
    print(f"DINOv2 cache: {len(revs)} snapshots")
    if revs:
        files = list(revs[0].glob("*"))
        print(f"  files: {[f.name for f in files]}")
else:
    print("DINOv2 cache: NOT FOUND")
    # Try other locations
    alt = pathlib.Path(r"I:\Github\Latent_Style\WEAVE\eval_cache")
    if alt.exists():
        print(f"  eval_cache contents: {[f.name for f in alt.iterdir()]}")

# Check SaMam curve - use curve_eval_30src (real files, not symlinks)
curve = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_30src")
steps = sorted(curve.glob("step_*"))
print(f"\nSaMam curve_eval_30src: {len(steps)} steps")
if steps:
    s0 = steps[0]
    imgs_dir = s0 / "images"
    if imgs_dir.exists():
        imgs = list(imgs_dir.glob("*.png"))
        print(f"  {s0.name}: {len(imgs)} PNGs")
        if imgs:
            p = imgs[0]
            print(f"  first: {p.name}")
            print(f"  path len: {len(str(p))}")
            try:
                img = Image.open(p).convert("RGB")
                print(f"  PIL OK: {img.size}")
            except Exception as e:
                print(f"  PIL error: {e}")

# Also check 'last'
last = curve / "last" / "images"
if last.exists():
    imgs = list(last.glob("*.png"))
    print(f"  last: {len(imgs)} PNGs")