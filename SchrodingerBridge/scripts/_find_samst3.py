"""Find SaMST curve images - safe search avoiding symlink dirs."""
import pathlib, os

# Check local G drive SaMST results
local_samst = pathlib.Path(r"G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_wikiarts5_wsl_20260610_172206")
print(f"Local SaMST exists: {local_samst.exists()}")
if local_samst.exists():
    for sub in sorted(local_samst.rglob("*")):
        try:
            if sub.is_dir():
                imgs = list(sub.glob("*.png")) + list(sub.glob("*.jpg"))
                if imgs:
                    print(f"  {sub.relative_to(local_samst)}: {len(imgs)} images")
                    print(f"    first: {imgs[0].name[:80]}")
        except:
            pass

# Check specific known SaMST dirs on remote
samst_dirs = [
    pathlib.Path(r"I:\Github\Latent_Style\exp_baselines\samst"),
    pathlib.Path(r"I:\Github\Latent_Style\final_works\SaMST-epoch_0100"),
]
for d in samst_dirs:
    print(f"\n{d}: exists={d.exists()}")
    if d.exists():
        for sub in sorted(d.rglob("*")):
            try:
                if sub.is_file() and sub.suffix.lower() in {".png", ".jpg"}:
                    print(f"  {sub.name}")
                elif sub.is_dir():
                    imgs = list(sub.glob("*.png")) + list(sub.glob("*.jpg"))
                    if imgs:
                        print(f"  DIR {sub.name}: {len(imgs)} images")
            except:
                pass

# Check for samst eval_bundle
eval_bundle = pathlib.Path(r"I:\Github\Latent_Style\Related_Works\baseline_pipeline\results")
if eval_bundle.exists():
    for sub in sorted(eval_bundle.iterdir()):
        try:
            if "samst" in sub.name.lower():
                print(f"\nFound: {sub}")
        except:
            pass