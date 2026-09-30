"""Phase 4J.6: Prepare few-shot Pop_Art dataset (8 images) as packed latent.

Creates:
  {output_root}/.latent_cache/packed/05_Pop_Art.pt   (new, 8 images)
  {output_root}/.latent_cache/packed/0X_{Base}.pt     (hardlinks to base cache)
  {output_root}/.latent_cache/packed/manifest.json    (6-style manifest)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def _load_vae(device: str, model_id: str = "ema"):
    src_path = str(_repo_src_path())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from utils.inference import load_vae
    return load_vae(device=device, model_id=model_id)


def _load_image_tensor(path: Path, image_size: int = 512) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


@torch.no_grad()
def encode_popart_images(
    image_paths: list[Path],
    vae,
    device: torch.device,
    latent_mode: str = "mode",
) -> torch.Tensor:
    """Encode images to latents [N, 4, 64, 64]."""
    latents_list = []
    batch = []
    for path in image_paths:
        batch.append(_load_image_tensor(path, 512))
        if len(batch) >= 4:
            tensor_batch = torch.stack(batch, dim=0).to(device=device, dtype=torch.float16)
            latent_dist = vae.encode(tensor_batch).latent_dist
            if latent_mode == "sample":
                lat = latent_dist.sample()
            elif latent_mode == "mean":
                lat = getattr(latent_dist, "mean", None) or latent_dist.mode()
            else:
                lat = latent_dist.mode()
            lat = lat * float(vae.config.scaling_factor)
            latents_list.append(lat.detach().float().cpu())
            batch = []
    if batch:
        tensor_batch = torch.stack(batch, dim=0).to(device=device, dtype=torch.float16)
        latent_dist = vae.encode(tensor_batch).latent_dist
        if latent_mode == "sample":
            lat = latent_dist.sample()
        elif latent_mode == "mean":
            lat = getattr(latent_dist, "mean", None) or latent_dist.mode()
        else:
            lat = latent_dist.mode()
        lat = lat * float(vae.config.scaling_factor)
        latents_list.append(lat.detach().float().cpu())
    return torch.cat(latents_list, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--popart-source-dir",
        required=True,
        help="Pop_Art source image directory",
    )
    parser.add_argument(
        "--base-cache-dir",
        required=True,
        help="Base latent_cache_dir (config value; contains manifest.json + packed/ subdir with .pt files)",
    )
    parser.add_argument(
        "--output-cache-dir",
        required=True,
        help="Output latent_cache_dir for 6-style few-shot dataset",
    )
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    popart_dir = Path(args.popart_source_dir)
    base_cache_dir = Path(args.base_cache_dir)  # latent_cache_dir from config
    out_cache_dir = Path(args.output_cache_dir)  # new latent_cache_dir
    # .pt files live in {cache_dir}/packed/, manifest in {cache_dir}/manifest.json
    base_packed_subdir = base_cache_dir / "packed"
    out_packed_subdir = out_cache_dir / "packed"
    out_packed_subdir.mkdir(parents=True, exist_ok=True)

    # 1. Select 8 Pop_Art images (random, seeded)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_paths = sorted(
        [p for p in popart_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    if len(all_paths) < args.num_images:
        raise ValueError(f"Only {len(all_paths)} Pop_Art images, need {args.num_images}")
    random.Random(args.seed).shuffle(all_paths)
    selected = sorted(all_paths[: args.num_images], key=lambda p: p.name)
    print(f"Selected {len(selected)} Pop_Art images:")
    for p in selected:
        print(f"  {p.name}")

    # 2. Load VAE and encode
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE (model={args.vae_model}) on {device}...")
    vae = _load_vae(str(device), args.vae_model)
    vae.eval()
    print("Encoding Pop_Art images to latents...")
    latents = encode_popart_images(selected, vae, device, latent_mode="mode")
    print(f"Encoded latents shape: {latents.shape}")  # [8, 4, 64, 64]

    # 3. Save packed 05_Pop_Art.pt
    popart_packed = out_packed_subdir / "05_Pop_Art.pt"
    # files list: relative to a hypothetical data_root, style subdir "Pop_Art"
    files_list = [f"Pop_Art\\{p.stem}.pt" for p in selected]
    payload = {
        "schema": 1,
        "subdir": "Pop_Art",
        "count": int(latents.shape[0]),
        "files": files_list,
        "latents": latents.contiguous(),
    }
    torch.save(payload, popart_packed)
    print(f"Saved packed Pop_Art latents: {popart_packed} ({popart_packed.stat().st_size / 1e6:.1f} MB)")

    # 4. Hardlink the 5 base style .pt files
    base_styles = [
        ("Early_Renaissance", "00_Early_Renaissance.pt"),
        ("Impressionism", "01_Impressionism.pt"),
        ("Minimalism", "02_Minimalism.pt"),
        ("Rococo", "03_Rococo.pt"),
        ("Ukiyo_e", "04_Ukiyo_e.pt"),
    ]
    for style_name, fname in base_styles:
        src = base_packed_subdir / fname
        dst = out_packed_subdir / fname
        if dst.exists():
            dst.unlink()
        if not src.exists():
            raise FileNotFoundError(f"Base packed file missing: {src}")
        os.link(src, dst)  # hardlink (same drive)
        print(f"Hardlinked: {dst.name} -> {src}")

    # 5. Build 6-style manifest
    # Read base manifest for file lists
    base_manifest_path = base_cache_dir / "manifest.json"
    if not base_manifest_path.exists():
        raise FileNotFoundError(f"Base manifest missing: {base_manifest_path}")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))

    # The dataset code checks: payload.get("data_root") == str(self.data_root)
    # data_root is the parent of .latent_cache (i.e., .../train)
    new_data_root = str(out_cache_dir.parent.parent)  # .../fewshot6_512_latents_ema/train

    styles_manifest = {}
    for style_name, fname in base_styles:
        base_style_data = base_manifest.get("styles", {}).get(style_name, {})
        styles_manifest[style_name] = {
            "count": base_style_data.get("count", 1000),
            "files": base_style_data.get("files", []),
        }
    styles_manifest["Pop_Art"] = {
        "count": int(latents.shape[0]),
        "files": files_list,
    }

    new_manifest = {
        "schema": 1,
        "data_root": new_data_root,
        "style_subdirs": [s[0] for s in base_styles] + ["Pop_Art"],
        "styles": styles_manifest,
    }
    manifest_path = out_cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(new_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved 6-style manifest: {manifest_path}")
    print(f"  data_root: {new_data_root}")
    print(f"  styles: {new_manifest['style_subdirs']}")
    print("\nDone. Few-shot Pop_Art dataset ready.")


if __name__ == "__main__":
    main()
