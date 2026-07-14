"""Inspect latent_wct p2a_256 and r5_wikiart summary settings."""
import json

for sub in ["p2a_256", "r5_wikiart", "d5_512"]:
    sp = rf"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\{sub}\summary.json"
    with open(sp) as f:
        s = json.load(f)
    settings = s.get("settings", {})
    print(f"\n=== {sub} ===")
    print(json.dumps(settings, indent=2))
    # Also check test_dir
    print(f"  test_dir: {settings.get('test_dir', 'N/A')}")
    print(f"  style_subdirs: {settings.get('style_subdirs', 'N/A')}")
