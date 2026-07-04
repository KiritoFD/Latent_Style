"""Generate SaMam photo2art baseline images on legacy256_overfit50.

SaMam's final_model.ckpt is a LatentLightningModel: a 4-channel latent-space
SaMam (patch_size=1) with an embedded SDXL VAE. The inference flow is:
  1. encode content/style PIL images -> latents (vae.encode * 0.18215)
  2. model.forward(content_latent, style_latent) -> stylized latent
  3. decode latent -> RGB image (vae.decode((z/0.18215)).sample)

For each target style we use the first image of that target style's test dir
as the style reference, then stylize all 150 source images (5 styles x 30) to
all 5 targets, giving 5*5*30 = 750 images.

Output naming: {src_style}_{src_id}_to_{tgt_style}.jpg
Output dir:    /mnt/i/exp_256_photo2art/samam_256/images/

Reference: scripts/samam_latent/gen_samam_latent.py (latent inference flow)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Hard-coded absolute WSL paths (remote repo layout)
SAMAM_REPO = Path("/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam")
SCHRODINGER_SRC = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/src")
sys.path.insert(0, str(SAMAM_REPO))
if str(SCHRODINGER_SRC) not in sys.path:
    sys.path.insert(0, str(SCHRODINGER_SRC))

from TRAIN.lightning_module.latent_lightningmodel import LatentLightningModel  # noqa: E402
from utils.inference import load_vae  # noqa: E402

STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
DEFAULT_CKPT = Path("/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/final_model.ckpt")
DEFAULT_TEST_ROOT = Path("/mnt/i/legacy256_overfit50/test")
DEFAULT_OUT_ROOT = Path("/mnt/i/exp_256_photo2art/samam_256")
LATENT_SCALE = 0.18215
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def image_files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def collect_sources(test_root: Path) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for s in STYLES:
        d = test_root / s
        if not d.exists():
            raise FileNotFoundError(f"Missing source style dir: {d}")
        for p in image_files(d):
            sources.append((s, p))
    return sources


def encode_image_to_latent(vae, img_pil: Image.Image, device: torch.device, size: int = 256) -> torch.Tensor:
    """Resize -> ToTensor -> [0,1] -> [-1,1] fp16 -> vae.encode -> *scale."""
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])(img_pil).unsqueeze(0).to(device)
    t = (t * 2.0 - 1.0).half()  # SDXL VAE is fp16; [0,1] -> [-1,1]
    with torch.no_grad():
        latent = vae.encode(t).latent_dist.sample() * LATENT_SCALE
    return latent.float()


def decode_latent_to_image(vae, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    """z = latent/scale (fp16) -> vae.decode -> (x+1)/2 -> clamp(0,1) -> float."""
    with torch.no_grad():
        z = (latent / LATENT_SCALE).half()
        img = vae.decode(z).sample
        img = (img + 1.0) / 2.0
        img = img.clamp(0, 1)
    return img.float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--vae-model",
        type=str,
        default="/mnt/i/Github/Latent_Style/SchrodingerBridge/eval_cache/hf/modelscope/stabilityai_sd-vae-ft-ema/stabilityai/sd-vae-ft-ema",
        help="Local path to VAE checkpoint (offline-friendly).",
    )
    parser.add_argument(
        "--vae-cache-dir",
        type=str,
        default="/mnt/i/Github/Latent_Style/SchrodingerBridge/eval_cache/hf",
        help="HF cache dir for VAE fallback search.",
    )
    # LatentLightningModel hyperparams (must match training of final_model.ckpt)
    parser.add_argument("--n-vssms", type=int, default=2)
    parser.add_argument("--n-savssms", type=int, default=2)
    parser.add_argument("--n-savssgs", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=1)  # latent: patch_size=1
    parser.add_argument("--representation-dim", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--expand", type=float, default=2.0)
    parser.add_argument("--compress-ratio", type=int, default=8)
    parser.add_argument("--squeeze-factor", type=int, default=8)
    parser.add_argument("--mamba-from-trion", type=int, default=1)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-scaling-factor", type=float, default=0.18215)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs (default: skip).")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"SaMam checkpoint not found: {args.ckpt}")

    out_dir = args.out_root / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"[samam] ckpt={args.ckpt}", flush=True)
    print(f"[samam] device={device}", flush=True)
    print(f"[samam] START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)

    # Load VAE (fp16 SDXL VAE; xformers disabled to keep deps minimal)
    cache_dir = args.vae_cache_dir.strip() or None
    vae = load_vae(device=str(device), model_id=args.vae_model, cache_dir=cache_dir, enable_xformers=False)
    vae.eval()
    for p_ in vae.parameters():
        p_.requires_grad_(False)
    print(f"[samam] VAE loaded (model_id={args.vae_model})", flush=True)

    # Load LatentLightningModel with explicit hyperparams (bypasses saved hparams resolution)
    model = LatentLightningModel.load_from_checkpoint(
        checkpoint_path=str(args.ckpt),
        map_location=device,
        nVSSMs=args.n_vssms,
        nSAVSSMs=args.n_savssms,
        nSAVSSGs=args.n_savssgs,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        representation_dim=args.representation_dim,
        d_state=args.d_state,
        expand=args.expand,
        compress_ratio=args.compress_ratio,
        squeeze_factor=args.squeeze_factor,
        mamba_from_trion=args.mamba_from_trion,
        latent_channels=args.latent_channels,
        latent_scaling_factor=args.latent_scaling_factor,
        vae_model=args.vae_model,
        vae_cache_dir=args.vae_cache_dir,
    )
    model = model.to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"[samam] LatentLightningModel loaded (patch_size={args.patch_size}, "
          f"latent_channels={args.latent_channels})", flush=True)

    sources = collect_sources(args.test_root)
    total = len(sources) * len(STYLES)

    # Build style reference latents (first image per target style's test dir)
    style_ref_latents: dict[str, torch.Tensor] = {}
    for s in STYLES:
        refs = image_files(args.test_root / s)
        if not refs:
            raise FileNotFoundError(f"No style reference images for {s}: {args.test_root / s}")
        simg = Image.open(refs[0]).convert("RGB")
        style_ref_latents[s] = encode_image_to_latent(vae, simg, device, args.image_size)
    print(f"[samam] {len(sources)} sources x {len(STYLES)} targets = {total} images",
          flush=True)

    t0 = time.time()
    generated = 0
    skipped = 0
    errors = 0

    with torch.inference_mode():
        for tgt in STYLES:
            style_latent = style_ref_latents[tgt]
            for src_style, src_path in sources:
                out_path = out_dir / f"{src_style}_{src_path.stem}_to_{tgt}.jpg"
                if not args.overwrite and out_path.exists():
                    skipped += 1
                    continue
                try:
                    cimg = Image.open(src_path).convert("RGB")
                    content_latent = encode_image_to_latent(vae, cimg, device, args.image_size)
                    output_latent = model.forward(content_latent, style_latent)
                    out_img = decode_latent_to_image(vae, output_latent, device)
                    save_image(out_img[0], str(out_path))
                    generated += 1
                except Exception as e:
                    errors += 1
                    print(f"[samam] ERROR {out_path.name} -> {e}", flush=True)
            print(f"[samam] target={tgt} done (generated={generated}, skipped={skipped}, "
                  f"errors={errors})", flush=True)
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"[samam] generated={generated} skipped={skipped} errors={errors} "
          f"expected={total} in {elapsed:.1f}s ({elapsed / max(generated, 1):.2f}s/img)",
          flush=True)
    print(f"[samam] out_dir={out_dir}", flush=True)
    print(f"[samam] END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
