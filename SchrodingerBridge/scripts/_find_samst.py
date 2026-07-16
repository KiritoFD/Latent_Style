"""Find SaMST curve image directories on remote."""
import pathlib

# Check local SaMST results
samst_local = pathlib.Path(r"I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samst_wikiarts5_wsl_20260610_172206")
print(f"SaMST base exists: {samst_local.exists()}")
if samst_local.exists():
    for sub in sorted(samst_local.rglob("*")):
        if sub.is_dir():
            imgs = list(sub.glob("*.png")) + list(sub.glob("*.jpg"))
            if imgs:
                print(f"  {sub.relative_to(samst_local)}: {len(imgs)} images")
                print(f"    first: {imgs[0].name}")

# Also check other SaMST locations
samst_dirs = [
    pathlib.Path(r"I:\Github\Latent_Style\exp_samst"),
    pathlib.Path(r"I:\Github\Latent_Style\Related_Works\baseline_pipeline\results"),
]
for d in samst_dirs:
    if d.exists():
        for sub in sorted(d.iterdir()):
            if "samst" in sub.name.lower():
                print(f"\nFound: {sub}")
                if sub.is_dir():
                    for sub2 in sorted(sub.rglob("*.png"))[:3]:
                        print(f"  PNG: {sub2}")