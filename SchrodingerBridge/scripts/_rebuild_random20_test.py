"""Rebuild wikiart_random20_512 test directory from manifest (copy from F: source)."""
import json
import shutil
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

manifest_path = Path("g:/GitHub/Latent_Style/Dataset/wikiart_random20_512/wikiart_random20_512/manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

test_split = manifest["splits"]["test"]
total = 0
copied = 0
missing = 0

def copy_one(src, dst):
    global copied, missing
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        return ("exists", src, dst)
    if not src.exists():
        return ("missing_src", src, dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return ("copied", src, dst)

tasks = []
for style, info in test_split.items():
    for rec in info["records"]:
        tasks.append((rec["source"], rec["target"]))
        total += 1

print(f"Total test images to rebuild: {total}")

results = {"copied": 0, "exists": 0, "missing_src": 0}
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(copy_one, s, d): s for s, d in tasks}
    for i, future in enumerate(as_completed(futures)):
        status, src, dst = future.result()
        results[status] += 1
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total} (copied={results['copied']}, exists={results['exists']}, missing={results['missing_src']})")

print(f"\nDone: copied={results['copied']}, exists={results['exists']}, missing_src={results['missing_src']}")
print(f"Test dir: g:/GitHub/Latent_Style/Dataset/wikiart_random20_512/wikiart_random20_512/images/test")
