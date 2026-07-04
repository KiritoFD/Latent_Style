"""Fix the fewshot6 manifest by reading packed .pt files to get correct counts and file lists.

The packed .pt files already exist at:
  fewshot6_512_latents_ema/train/.latent_cache/packed/packed/{0X}_{style}.pt

The manifest needs:
1. data_root matching str(Path(config_data_root)) on Windows
2. style_subdirs = 6 styles
3. Each style's count and files matching the packed .pt payload
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> int:
    # Paths
    cache_dir = Path(
        "G:/GitHub/Latent_Style/Dataset/fewshot6_512_latents_ema/train/.latent_cache/packed"
    )
    packed_subdir = cache_dir / "packed"
    manifest_path = cache_dir / "manifest.json"

    # Config values
    data_root_str = "G:/GitHub/Latent_Style/Dataset/fewshot6_512_latents_ema/train"
    # On Windows, str(Path(...)) normalizes forward slashes to backslashes
    data_root_normalized = str(Path(data_root_str))
    style_subdirs = [
        "Early_Renaissance",
        "Impressionism",
        "Minimalism",
        "Rococo",
        "Ukiyo_e",
        "Pop_Art",
    ]

    print(f"[fix] data_root config string : {data_root_str}")
    print(f"[fix] data_root normalized    : {data_root_normalized}")
    print(f"[fix] manifest_path           : {manifest_path}")
    print(f"[fix] packed_subdir           : {packed_subdir}")
    print(f"[fix] style_subdirs           : {style_subdirs}")

    if not packed_subdir.exists():
        print(f"[ERROR] packed subdir not found: {packed_subdir}", file=sys.stderr)
        return 1

    # Read each packed .pt to get count and files
    styles_payload: dict[str, dict] = {}
    for style_id, subdir in enumerate(style_subdirs):
        cache_name = f"{style_id:02d}_{subdir}.pt"
        packed_path = packed_subdir / cache_name
        if not packed_path.exists():
            print(f"[ERROR] packed file not found: {packed_path}", file=sys.stderr)
            return 1
        print(f"[fix] loading {packed_path} ...")
        payload = torch.load(packed_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            print(f"[ERROR] unexpected payload type: {type(payload)}", file=sys.stderr)
            return 1
        count = int(payload.get("count", 0))
        files = list(payload.get("files", []))
        latents = payload.get("latents")
        lat_shape = tuple(latents.shape) if latents is not None else None
        print(f"  subdir={payload.get('subdir')}, count={count}, files={len(files)}, latents={lat_shape}")
        styles_payload[subdir] = {"count": count, "files": files}

    # Build manifest
    manifest = {
        "schema": 1,
        "data_root": data_root_normalized,
        "style_subdirs": style_subdirs,
        "styles": styles_payload,
    }

    # Write manifest
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_path)
    print(f"[fix] wrote manifest: {manifest_path}")
    print(f"[fix] data_root in manifest: {data_root_normalized}")

    # Verify
    verify = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert verify["data_root"] == data_root_normalized
    assert verify["style_subdirs"] == style_subdirs
    for subdir in style_subdirs:
        c = verify["styles"][subdir]["count"]
        f = len(verify["styles"][subdir]["files"])
        print(f"  {subdir}: count={c}, files={f}")
    print("[fix] verification PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
