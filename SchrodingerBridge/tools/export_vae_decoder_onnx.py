from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_PATH = str(_repo_src_path())
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.inference import VAEDecodeWrapper, load_vae  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-shape SD VAE decoder to ONNX.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--hf-cache-dir", default="I:/Github/Latent_Style/eval_cache/hf")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-h", type=int, default=64)
    parser.add_argument("--latent-w", type=int, default=64)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        print(f"exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = args.device
    t0 = time.perf_counter()
    vae = load_vae(device=device, model_id=args.vae_model, cache_dir=args.hf_cache_dir, enable_xformers=False)
    wrapper = VAEDecodeWrapper(vae).to(device=device, dtype=torch.float16).eval()
    dummy = torch.randn(
        int(args.batch_size),
        4,
        int(args.latent_h),
        int(args.latent_w),
        device=device,
        dtype=torch.float16,
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["latent"],
            output_names=["image"],
            opset_version=int(args.opset),
            do_constant_folding=True,
            dynamic_axes=None,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    manifest = {
        "output": str(out_path),
        "vae_model": args.vae_model,
        "batch_size": int(args.batch_size),
        "latent_shape": [4, int(args.latent_h), int(args.latent_w)],
        "opset": int(args.opset),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seconds": time.perf_counter() - t0,
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
