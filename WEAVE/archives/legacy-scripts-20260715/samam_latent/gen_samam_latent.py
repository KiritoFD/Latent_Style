"""SaMam-latent inference: encode test images -> stylize latents -> decode -> save PNG.

Loads a LatentLightningModel checkpoint and runs inference in latent space.
Output naming matches eval_samam_metrics_phase2.py convention:
    {src_style}__{src_stem}__to__{tgt_style}.png
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# SaMam repo root
SAMAM_ROOT = Path("/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam")
sys.path.insert(0, str(SAMAM_ROOT))

# SchrodingerBridge src (for VAE loader)
SCHRODINGER_SRC = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/src")
if str(SCHRODINGER_SRC) not in sys.path:
    sys.path.insert(0, str(SCHRODINGER_SRC))

from TRAIN.lightning_module.latent_lightningmodel import LatentLightningModel
from utils.inference import load_vae


STYLE_NAMES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
LATENT_SCALE = 0.18215


def encode_image_to_latent(vae, img_pil, device):
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])(img_pil).unsqueeze(0).to(device)
    # SDXL VAE loaded as fp16
    t = (t * 2.0 - 1.0).half()  # [0,1] -> [-1,1] fp16
    with torch.no_grad():
        latent = vae.encode(t).latent_dist.sample() * LATENT_SCALE
    return latent.float()


def decode_latent_to_image(vae, latent, device):
    with torch.no_grad():
        z = (latent / LATENT_SCALE).half()
        img = vae.decode(z).sample
        img = (img + 1.0) / 2.0
        img = img.clamp(0, 1)
    return img.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--vae-cache-dir", default="")
    p.add_argument("--vae-model", default="ema")
    p.add_argument("--style-names", default=",".join(STYLE_NAMES))
    p.add_argument("--num-src", type=int, default=30)
    # Model params (must match training)
    p.add_argument("--nVSSMs", type=int, default=2)
    p.add_argument("--nSAVSSMs", type=int, default=2)
    p.add_argument("--nSAVSSGs", type=int, default=2)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=1)  # latent: patch_size=1
    p.add_argument("--representation-dim", type=int, default=64)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--expand", type=float, default=2.0)
    p.add_argument("--compress-ratio", type=int, default=8)
    p.add_argument("--squeeze-factor", type=int, default=8)
    p.add_argument("--mamba-from-trion", type=int, default=1)
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--latent-scaling-factor", type=float, default=0.18215)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    device = torch.device(args.device)

    # Load VAE
    cache_dir = args.vae_cache_dir.strip() or None
    vae = load_vae(device=device, model_id=args.vae_model, cache_dir=cache_dir, enable_xformers=False)
    vae.eval()
    for p_ in vae.parameters():
        p_.requires_grad_(False)

    # Load model
    model = LatentLightningModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        map_location=device,
        nVSSMs=args.nVSSMs,
        nSAVSSMs=args.nSAVSSMs,
        nSAVSSGs=args.nSAVSSGs,
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
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")
    print(f"[INFO] patch_size={args.patch_size}, latent_channels={args.latent_channels}")

    # Build style reference latents (first image per style)
    style_ref_latents = {}
    for sname in args.style_names:
        style_dir = Path(args.test_root) / sname
        files = sorted([f for f in style_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        if not files:
            print(f"[WARN] No style ref for {sname}")
            continue
        img = Image.open(files[0]).convert("RGB")
        style_ref_latents[sname] = encode_image_to_latent(vae, img, device)

    out_dir = Path(args.output_root) / "step_000001" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    t0 = time.time()

    with torch.no_grad():
        for src_style in args.style_names:
            src_dir = Path(args.test_root) / src_style
            if not src_dir.exists():
                continue
            src_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}])
            src_files = src_files[:args.num_src]

            for tgt_style in args.style_names:
                if tgt_style not in style_ref_latents:
                    continue

                for src_path in src_files:
                    src_stem = src_path.stem  # keep full stem
                    out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                    out_path = out_dir / out_name
                    if out_path.exists():
                        continue

                    try:
                        img = Image.open(src_path).convert("RGB")
                        content_latent = encode_image_to_latent(vae, img, device)
                        style_latent = style_ref_latents[tgt_style]

                        output = model.forward(content_latent, style_latent)
                        out_img = decode_latent_to_image(vae, output, device)
                        out_pil = transforms.ToPILImage()(out_img.squeeze(0).cpu())
                        out_pil.save(str(out_path))
                        total += 1
                    except Exception as e:
                        print(f"  ERROR: {out_name} -> {e}")

    elapsed = time.time() - t0
    print(f"[INFO] Generated {total} images in {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
