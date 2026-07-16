"""Find SaMST curve image directories - broader search."""
import pathlib

# Search for samst directories
roots = [
    pathlib.Path(r"I:\Github\Latent_Style"),
    pathlib.Path(r"I:\datasets"),
]

for root in roots:
    if not root.exists():
        continue
    for item in root.rglob("*"):
        if item.is_dir() and "samst" in item.name.lower():
            imgs = list(item.glob("*.png")) + list(item.glob("*.jpg"))
            if imgs:
                print(f"{item}: {len(imgs)} images")
                print(f"  first: {imgs[0].name[:80]}")

# Also check the local G drive SaMST results
local_samst = pathlib.Path(r"G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_wikiarts5_wsl_20260610_172206")
print(f"\nLocal SaMST exists: {local_samst.exists()}")
if local_samst.exists():
    for sub in sorted(local_samst.rglob("*")):
        if sub.is_dir():
            imgs = list(sub.glob("*.png")) + list(sub.glob("*.jpg"))
            if imgs:
                print(f"  {sub.relative_to(local_samst)}: {len(imgs)} images")
                print(f"    first: {imgs[0].name[:80]}")