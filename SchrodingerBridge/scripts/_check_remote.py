"""Check remote GPU environment."""
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")

try:
    import transformers
    print("transformers ok, ver:", transformers.__version__)
except:
    print("NO transformers")

try:
    import PIL
    print("PIL ok")
except:
    print("NO PIL")

# Check DINO cache
import pathlib
cache = pathlib.Path(r"I:\Github\Latent_Style\WEAVE\eval_cache\offline_pairing")
for f in cache.glob("*dino*"):
    print(f"DINO cache: {f.name} ({f.stat().st_size//1024**2}MB)")

# Check SaMam curve dir
curve = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_hf_750_batched")
steps = sorted(curve.glob("step_*"))
print(f"SaMam curve steps: {len(steps)}, range: {steps[0].name} - {steps[-1].name}")

# Check classview manifest
cv = pathlib.Path(r"I:\datasets\wikiart_distinct5_samam_512_classview")
if cv.exists():
    mf = cv / "manifest.json"
    if mf.exists():
        import json
        m = json.load(open(mf))
        print(f"classview manifest: {len(m)} classes: {list(m.keys())[:8]}")
        for k, v in list(m.items())[:2]:
            print(f"  {k}: {len(v)} images, e.g. {v[0]}")
    else:
        print("NO manifest.json")
else:
    print(f"classview dir not found: {cv}")

# Check SaMST images
samst = pathlib.Path(r"I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samst_wikiarts5_wsl_20260610_172206\eval_bundle")
if samst.exists():
    print(f"SaMST eval_bundle exists")
    for sub in samst.glob("*"):
        if sub.is_dir():
            imgs = list(sub.glob("*.png"))
            print(f"  {sub.name}: {len(imgs)} PNGs")