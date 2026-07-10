"""Check test_image_dir and key metrics in random20_heun summary.json."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def check(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    settings = data.get("settings", {})
    test_dir = settings.get("test_image_dir", settings.get("test_dir", "N/A"))
    apo = data.get("analysis", {}).get("all_pairs_overview", {})
    clip_s = apo.get("clip_style", "N/A")
    lpips = apo.get("content_lpips", apo.get("lpips", "N/A"))
    ckpt = data.get("checkpoint", {})
    print(f"=== {path} ===")
    print(f"  test_image_dir: {test_dir}")
    print(f"  clip_style: {clip_s}")
    print(f"  lpips: {lpips}")
    print(f"  checkpoint: {ckpt}")
    # also check data_root in settings
    for k, v in settings.items():
        kl = k.lower()
        if "test" in kl and "dir" in kl:
            print(f"  setting {k}: {v}")
        if "data_root" in kl:
            print(f"  setting {k}: {v}")


base = Path("g:/GitHub/Latent_Style/SchrodingerBridge/exp/630_random20_heun_5ep")
# Check all summary.json in this exp dir
for s in base.rglob("summary.json"):
    check(str(s))
    print()
