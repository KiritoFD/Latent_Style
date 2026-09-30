"""SAMST-latent inference: encode test images -> stylize latents -> decode -> save PNG.

For each (src_style, tgt_style) pair, generates stylized images.
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from networks_latent import TransformerNetLatent

# Reuse SchrodingerBridge VAE loader
SCHRODINGER_SRC = PROJECT_ROOT / "src"
if str(SCHRODINGER_SRC) not in sys.path:
    sys.path.insert(0, str(SCHRODINGER_SRC))
from utils.inference import load_vae


STYLE_NAMES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
LATENT_SCALE = 0.18215


def encode_image_to_latent(vae, img_pil, device):
    """Encode PIL image to VAE latent (4x32x32), pre-scaled by 0.18215."""
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])(img_pil).unsqueeze(0).to(device)
    # SDXL VAE expects [-1, 1] in fp16 (VAE loaded as fp16)
    t = (t * 2.0 - 1.0).half()
    with torch.no_grad():
        latent = vae.encode(t).latent_dist.sample() * LATENT_SCALE
    return latent.float()


def decode_latent_to_image(vae, latent, device):
    """Decode latent (pre-scaled) back to PIL image."""
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
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    style_num = len(args.style_names)
    device = torch.device(args.device)

    # Load VAE
    cache_dir = args.vae_cache_dir.strip() or None
    vae = load_vae(device=device, model_id=args.vae_model, cache_dir=cache_dir, enable_xformers=False)
    vae.eval()
    vae = vae.to(device)
    for p_ in vae.parameters():
        p_.requires_grad_(False)

    # Load model
    model = TransformerNetLatent(style_num=style_num, in_channels=4, latent_channels=4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

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

    for src_style in args.style_names:
        src_dir = Path(args.test_root) / src_style
        if not src_dir.exists():
            continue
        src_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        src_files = src_files[:args.num_src]

        for tgt_style in args.style_names:
            if tgt_style not in style_ref_latents:
                continue
            tgt_idx = args.style_names.index(tgt_style) + 1  # 1-indexed style_id

            for src_path in src_files:
                src_stem = src_path.stem  # keep full stem with style prefix
                out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                out_path = out_dir / out_name
                if out_path.exists():
                    continue

                try:
                    img = Image.open(src_path).convert("RGB")
                    content_latent = encode_image_to_latent(vae, img, device)

                    with torch.no_grad():
                        y, _ = model(content_latent, style_id=[tgt_idx])

                    out_img = decode_latent_to_image(vae, y, device)
                    out_pil = transforms.ToPILImage()(out_img.squeeze(0).cpu())
                    out_pil.save(str(out_path))
                    total += 1
                except Exception as e:
                    print(f"  ERROR: {out_name} -> {e}")

    elapsed = time.time() - t0
    print(f"[INFO] Generated {total} images in {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
