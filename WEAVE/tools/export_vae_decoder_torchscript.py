from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.inference import VAEDecodeWrapper, load_vae


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a fixed-shape TorchScript VAE decoder.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--vae-cache-dir", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-height", type=int, default=32)
    parser.add_argument("--latent-width", type=int, default=32)
    args = parser.parse_args()

    device = "cuda"
    vae = load_vae(
        device=device,
        model_id=args.vae_model,
        cache_dir=args.vae_cache_dir or None,
        enable_xformers=False,
    )
    wrapper = VAEDecodeWrapper(vae).to(
        device=device,
        dtype=torch.float16,
        memory_format=torch.channels_last,
    ).eval()
    example = torch.randn(
        args.batch_size,
        4,
        args.latent_height,
        args.latent_width,
        device=device,
        dtype=torch.float16,
    ).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced = torch.jit.freeze(traced.eval())
        traced(example)
        torch.cuda.synchronize()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output))
    print(f"Saved fixed VAE decoder: {output}")


if __name__ == "__main__":
    main()
