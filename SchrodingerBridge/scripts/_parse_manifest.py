"""Parse manifest.json to understand train/test split structure."""
import json
import pathlib

manifest_path = pathlib.Path(r"I:\datasets\wikiart_distinct5_samam_512_classview\manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print("=== manifest top-level keys ===")
print(list(manifest.keys()) if isinstance(manifest, dict) else f"type={type(manifest)}, len={len(manifest)}")

if isinstance(manifest, dict):
    for k, v in manifest.items():
        if isinstance(v, list):
            print(f"\n{k}: list of {len(v)}")
            if v:
                print(f"  first item type: {type(v[0])}")
                if isinstance(v[0], dict):
                    print(f"  first item keys: {list(v[0].keys())}")
                    # Print first item without the long lists
                    item = dict(v[0])
                    for kk in list(item.keys()):
                        if isinstance(item[kk], list):
                            item[kk] = f"[list of {len(item[kk])}]"
                    print(f"  first item: {json.dumps(item, indent=2)[:500]}")
        elif isinstance(v, dict):
            print(f"\n{k}: dict with keys {list(v.keys())[:10]}")
        else:
            print(f"\n{k}: {v}")

# Check if there's a styles list with train/test sources
if isinstance(manifest, dict) and "styles" in manifest:
    styles = manifest["styles"]
    print("\n\n=== styles detail ===")
    for s in styles:
        name = s.get("name", s.get("style", "?"))
        train_src = s.get("train_sources", s.get("train", []))
        test_src = s.get("test_sources", s.get("test", []))
        print(f"\n{name}: train={len(train_src)}, test={len(test_src)}")
        if test_src:
            print(f"  first test: {pathlib.Path(test_src[0]).name}")
            print(f"  last test: {pathlib.Path(test_src[-1]).name}")

# Also check train dir file counts
import os
train_dir = pathlib.Path(r"I:\datasets\wikiart_distinct5_samam_512_classview\train")
print("\n\n=== train dir counts ===")
for style_dir in sorted(train_dir.iterdir()):
    if style_dir.is_dir():
        cnt = len([f for f in style_dir.iterdir() if f.is_file()])
        print(f"  {style_dir.name}: {cnt} files")
