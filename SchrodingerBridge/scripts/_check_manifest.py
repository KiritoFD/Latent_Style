"""Check manifest and SaMam curve structure."""
import json, pathlib

# Check manifest
cv = pathlib.Path(r"I:\datasets\wikiart_distinct5_samam_512_classview")
mf = cv / "manifest.json"
m = json.load(open(mf))
print("manifest keys:", list(m.keys()))
print()

# styles - it's a list
if "styles" in m:
    styles = m["styles"]
    print(f"styles list ({len(styles)}): {styles}")

# classes
if "classes" in m:
    classes = m["classes"]
    print(f"\nclasses type: {type(classes).__name__}")
    if isinstance(classes, dict):
        for k, v in list(classes.items())[:5]:
            print(f"  {k}: {v}")
    elif isinstance(classes, list):
        print(f"  list ({len(classes)}): {classes[:5]}")

# Check train dir structure
train_dir = cv / "train"
if train_dir.exists():
    subdirs = sorted(train_dir.glob("*"))
    print(f"\ntrain subdirs ({len(subdirs)}):")
    for sd in subdirs[:5]:
        imgs = list(sd.glob("*.jpg")) + list(sd.glob("*.png"))
        print(f"  {sd.name}: {len(imgs)} images")

# Check SaMam curve dir for one step
curve = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_hf_750_batched")
step = curve / "step_020000"
imgs = sorted(step.glob("*.png"))[:5]
print(f"\nSaMam step_020000: {len(list(step.glob('*.png')))} PNGs")
print("first 5 filenames:")
for img in imgs:
    print(f"  {img.name}")

# Check SaMST
samst = pathlib.Path(r"I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samst_wikiarts5_wsl_20260610_172206\eval_bundle")
for sub in sorted(samst.glob("*")):
    if sub.is_dir():
        imgs = list(sub.glob("*.png")) + list(sub.glob("*.jpg"))
        print(f"\nSaMST {sub.name}: {len(imgs)} images")
        if imgs:
            print(f"  first: {imgs[0].name}")