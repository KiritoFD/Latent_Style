"""Debug reuse_generated: check filename parsing and src_lookup."""
import sys
from pathlib import Path
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge")

from utils.run_evaluation import _parse_generated_name, _list_reuse_generated_files

out_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\full_eval\epoch_0000")
test_dir = Path(r"I:\datasets\wikiart_distinct5_samam_512_classview\test")

# Get style_subdirs
style_subdirs = [d.name for d in test_dir.iterdir() if d.is_dir()]
print(f"style_subdirs: {style_subdirs}")

# List reuse files
files = _list_reuse_generated_files(out_dir)
print(f"Found {len(files)} reuse files")
if files:
    print(f"First file: {files[0].name}")

    # Parse first file
    parsed = _parse_generated_name(files[0].name, style_subdirs)
    print(f"Parsed: {parsed}")

    # Build src_lookup
    all_src_info = []
    for s_id, s_name in enumerate(style_subdirs):
        s_dir = test_dir / s_name
        if not s_dir.exists():
            continue
        imgs = sorted([p for p in s_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']])
        for p in imgs:
            all_src_info.append({'path': p, 'style_id': s_id, 'style_name': s_name})

    src_lookup = {(x["style_name"], x["path"].stem): x["path"] for x in all_src_info}
    print(f"src_lookup has {len(src_lookup)} entries")

    if parsed:
        src_style, src_stem, tgt_style = parsed
        src_path = src_lookup.get((src_style, src_stem))
        print(f"Lookup ({src_style}, {src_stem}): {src_path}")

        # Try to find the actual key
        matching_keys = [k for k in src_lookup.keys() if k[0] == src_style]
        print(f"Keys for {src_style}: {len(matching_keys)}")
        if matching_keys:
            print(f"  First key: {matching_keys[0]}")
            print(f"  src_stem matches: {src_stem in [k[1] for k in matching_keys]}")
