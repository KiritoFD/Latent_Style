"""Create a manifest.json for a packed latent cache directory.

In packed mode, AdaCUTLatentDataset still needs a manifest to register
style item stems (used for pairing). The actual latents are loaded from
packed cache, so the manifest file paths don't need to exist on disk —
only the stems are used.

This script reads the `files` field from each packed .pt file and builds
a manifest.json consistent with what _load_or_build_manifest expects.
"""
import json
import sys
import torch
from pathlib import Path

WEAVE_ROOT = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\weave_gen")

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_FILES = [
    "00_Early_Renaissance.pt",
    "01_Impressionism.pt",
    "02_Minimalism.pt",
    "03_Rococo.pt",
    "04_Ukiyo_e.pt",
]


def create_manifest(data_root: str):
    data_root_path = Path(data_root)
    cache_dir = data_root_path / ".latent_cache"
    packed_dir = cache_dir / "packed"

    if not packed_dir.exists():
        print(f"ERROR: packed dir not found: {packed_dir}")
        sys.exit(1)

    styles_payload = {}
    for style, packed_file in zip(STYLES, STYLE_FILES):
        packed_path = packed_dir / packed_file
        if not packed_path.exists():
            print(f"ERROR: packed cache not found: {packed_path}")
            sys.exit(1)
        payload = torch.load(packed_path, map_location="cpu", weights_only=False)
        files = payload["files"]
        count = payload.get("count", len(files))
        styles_payload[style] = {
            "count": count,
            "files": [str(f) for f in files],
        }
        print(f"  {style}: {count} files (from {packed_file})")

    manifest = {
        "schema": 1,
        "data_root": str(data_root_path),
        "style_subdirs": STYLES,
        "styles": styles_payload,
    }

    manifest_path = cache_dir / "manifest.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest written to: {manifest_path}")
    print(f"Total styles: {len(STYLES)}")


if __name__ == "__main__":
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/train_sdxl"
    if not Path(data_root).is_absolute():
        data_root = str(WEAVE_ROOT / data_root)
    print(f"Creating manifest for data_root: {data_root}")
    create_manifest(data_root)
