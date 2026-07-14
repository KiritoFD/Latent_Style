"""Latent-WCT Baseline (0-second training): VAE encode -> Haar decompose -> high-freq WCT -> decode.

This baseline isolates the contribution of Rectified Flow's 8-step ODE integration.
It reuses WEAVE's exact WCT implementation and Haar wavelet decomposition, but
removes the trained velocity field backbone and the ODE integration loop entirely.

Pipeline:
  1. VAE encode source image -> src_latent (1, 4, 32, 32)
  2. VAE encode target style's first reference image -> style_latent
  3. Haar multi-level decompose both (levels=1, matching WEAVE default)
  4. Apply WCT to LH/HL/HH subbands (alpha=1.0 full transfer), LL locked (preserve content)
  5. iDWT reconstruct -> modified latent
  6. VAE decode -> output image (256x256)

Output naming matches SaMam/IDT convention:
  {output_root}/step_000001/images/{src_style}__{src_stem}__to__{tgt_style}.png
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

# Add src to path for imports
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from utils.inference import load_vae, encode_image, decode_latent  # noqa: E402
from spectral620 import dwt2_haar_multi_decompose, idwt2_haar_multi_reconstruct  # noqa: E402
from spectral_bridge620 import _wct_match_fiber  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STYLE_NAMES = [
    "Abstract_Expressionism",
    "Art_Nouveau_Modern",
    "Cubism",
    "Expressionism",
    "Symbolism",
]


def load_image_tensor(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    return T.ToTensor()(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True, help="other5 test root with style subdirs")
    parser.add_argument("--output-dir", required=True, help="Output dir")
    parser.add_argument("--style-names", default=",".join(STYLE_NAMES))
    parser.add_argument("--num-src", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--vae-cache-dir", default="")
    parser.add_argument("--levels", type=int, default=1, help="Haar decomposition levels (1=WEAVE default)")
    parser.add_argument("--alpha", type=float, default=1.0, help="WCT blend strength (1.0=full transfer)")
    parser.add_argument("--ll-alpha", type=float, default=0.0, help="LL subband transfer strength (0.0=locked)")
    args = parser.parse_args()

    style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    device = torch.device(args.device)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "step_000001" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE
    cache_dir = args.vae_cache_dir.strip() or None
    print(f"[INFO] Loading VAE (model_id={args.vae_model})...")
    vae = load_vae(device=device, model_id=args.vae_model, cache_dir=cache_dir)
    print("[INFO] VAE loaded")

    # Precompute style latent (first reference image per style, matching WEAVE convention)
    print("[INFO] Precomputing style latents (first ref per style)...")
    style_latents = {}
    for sname in style_names:
        sdir = test_dir / sname
        ref_files = sorted([p for p in sdir.iterdir() if p.suffix.lower() in IMAGE_EXTS])[:args.num_src]
        if not ref_files:
            print(f"  [WARN] No reference images for {sname}")
            continue
        # Use first reference image (matches run_evaluation.py _tgt_paths[0])
        ref_img = load_image_tensor(ref_files[0], size=256).unsqueeze(0).to(device)
        ref_latent = encode_image(vae, ref_img, device)
        style_latents[sname] = ref_latent
        print(f"  {sname}: latent shape={tuple(ref_latent.shape)}")

    # Precompute style DWT decomposition
    style_dwts = {}
    for sname, s_lat in style_latents.items():
        s_decomp = dwt2_haar_multi_decompose(s_lat.float(), levels=args.levels)
        style_dwts[sname] = s_decomp

    # Process each (src, tgt) pair
    t0 = time.time()
    n_done = 0
    for src_style in style_names:
        src_dir = test_dir / src_style
        src_files = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])[:args.num_src]
        for src_path in src_files:
            # Encode source
            src_img = load_image_tensor(src_path, size=256).unsqueeze(0).to(device)
            with torch.no_grad():
                src_latent = encode_image(vae, src_img, device)

            for tgt_style in style_names:
                if tgt_style not in style_dwts:
                    continue
                out_name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
                out_path = images_dir / out_name

                s_decomp = style_dwts[tgt_style]

                with torch.no_grad():
                    # Decompose source latent
                    h_decomp = dwt2_haar_multi_decompose(src_latent.float(), levels=args.levels)

                    # Apply WCT to high-freq subbands, LL locked
                    ll_K = h_decomp["ll_K"]
                    if args.ll_alpha > 0.0:
                        s_ll = s_decomp["ll_K"]
                        ll_K = (1.0 - args.ll_alpha) * ll_K + args.ll_alpha * _wct_match_fiber(ll_K, s_ll)

                    new_subs = []
                    for k, (lh, hl, hh) in enumerate(h_decomp["h"]):
                        s_lh, s_hl, s_hh = s_decomp["h"][k]
                        lh_new = (1.0 - args.alpha) * lh + args.alpha * _wct_match_fiber(lh, s_lh)
                        hl_new = (1.0 - args.alpha) * hl + args.alpha * _wct_match_fiber(hl, s_hl)
                        hh_new = (1.0 - args.alpha) * hh + args.alpha * _wct_match_fiber(hh, s_hh)
                        new_subs.append((lh_new, hl_new, hh_new))

                    # Reconstruct
                    modified_latent = idwt2_haar_multi_reconstruct(
                        {"ll_K": ll_K, "h": new_subs}, levels=args.levels
                    )

                    # Decode
                    modified_latent = modified_latent.to(dtype=torch.float16, device=device)
                    output_img = decode_latent(vae, modified_latent, device)

                # Save
                from torchvision.utils import save_image
                save_image(output_img[0].cpu(), str(out_path))
                n_done += 1

                if n_done % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{n_done}/750] {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    print(f"[INFO] Done: {n_done} images in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
