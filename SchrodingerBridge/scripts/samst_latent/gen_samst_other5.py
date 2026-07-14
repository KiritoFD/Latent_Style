"""SAMST-latent inference for other5 dataset.

SaMST uses integer style_id indexing into a StyleBank of 5 learned D5 style
representations. For other5 (unseen styles), we map each other5 style to a
D5 style_id via a fixed seed=42 permutation. This tests SaMST's generalization
ability given its architecture constraint (no style reference embedding).

Output naming matches ours convention:
    {src_style}__{src_stem}__to__{tgt_style}.png
"""
import argparse
import random
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

# D5 training styles (order matters for style_id 1..5)
D5_STYLE_NAMES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
# other5 test styles
OTHER5_STYLE_NAMES = [
    "Abstract_Expressionism",
    "Art_Nouveau_Modern",
    "Cubism",
    "Expressionism",
    "Symbolism",
]
LATENT_SCALE = 0.18215


def encode_image_to_latent(vae, img_pil, device):
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])(img_pil).unsqueeze(0).to(device)
    t = (t * 2.0 - 1.0).half()
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
    p.add_argument("--style-names", default=",".join(OTHER5_STYLE_NAMES))
    p.add_argument("--num-src", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    style_num = len(args.style_names)
    device = torch.device(args.device)

    # Build fixed other5 -> D5 style_id mapping (seed=42)
    rng = random.Random(args.seed)
    d5_indices = list(range(1, len(D5_STYLE_NAMES) + 1))  # [1,2,3,4,5]
    rng.shuffle(d5_indices)
    other5_to_d5 = {sname: d5_indices[i] for i, sname in enumerate(args.style_names)}
    print("=== other5 -> D5 style_id mapping (seed=42) ===")
    for sname, sid in other5_to_d5.items():
        d5_name = D5_STYLE_NAMES[sid - 1]
        print(f"  {sname} -> style_id={sid} ({d5_name})")

    # Load VAE
    cache_dir = args.vae_cache_dir.strip() or None
    vae = load_vae(device=device, model_id=args.vae_model, cache_dir=cache_dir, enable_xformers=False)
    vae.eval()
    vae = vae.to(device)
    for p_ in vae.parameters():
        p_.requires_grad_(False)

    # Load model (5 D5 styles)
    model = TransformerNetLatent(style_num=len(D5_STYLE_NAMES), in_channels=4, latent_channels=4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    out_dir = Path(args.output_root) / "step_000001" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    t0 = time.time()
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}")

    for src_style in args.style_names:
        src_dir = Path(args.test_root) / src_style
        if not src_dir.exists():
            print(f"[WARN] No dir for {src_style}")
            continue
        src_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        src_files = src_files[: args.num_src]

        for tgt_style in args.style_names:
            tgt_idx = other5_to_d5[tgt_style]  # mapped D5 style_id (1-indexed)

            for src_path in src_files:
                src_stem = src_path.stem
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
    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}")


if __name__ == "__main__":
    main()
