from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMST_ROOT = REPO_ROOT / "Related_Works" / "repos" / "SaMST-main"
import sys

sys.path.insert(0, str(SAMST_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SchrodingerBridge" / "src"))

from networks.latent_transfer_net import LatentTransformerNet  # noqa: E402
from utils.inference import encode_image, load_vae  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def tensor_for_vae(path: Path, size: int, device: torch.device) -> torch.Tensor:
    tr = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    return tr(load_rgb(path)).unsqueeze(0).to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)


def decode_latent_train(vae, latent: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    dtype = torch.float16 if latent.device.type == "cuda" else torch.float32
    z = latent.to(dtype=dtype) / max(float(scaling_factor), 1e-8)
    decoded = vae.decode(z).sample
    decoded = (decoded + 1.0) / 2.0
    return torch.clamp(decoded, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--style-names", type=str, required=True)
    parser.add_argument("--vae-model", type=str, default="ema")
    parser.add_argument("--vae-cache-dir", type=str, default="")
    parser.add_argument("--latent-scaling-factor", type=float, default=0.18215)
    args = parser.parse_args()

    style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    output_root = args.output_root.resolve()
    image_root = args.image_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LatentTransformerNet(style_num=len(style_names)).to(device).eval()
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    vae = load_vae(device=str(device), model_id=str(args.vae_model), cache_dir=str(args.vae_cache_dir).strip() or None, enable_xformers=False)
    vae.requires_grad_(False)
    vae.eval()

    style_refs = {}
    for style in style_names:
        paths = image_paths(image_root / style)
        if not paths:
            raise FileNotFoundError(image_root / style)
        style_refs[style] = tensor_for_vae(paths[0], args.image_size, device)

    with torch.no_grad():
        for src_style in style_names:
            src_paths = image_paths(image_root / src_style)[: args.max_src_per_style]
            for src_path in src_paths:
                content = tensor_for_vae(src_path, args.image_size, device)
                content_latent = encode_image(vae, content, device=str(device)).float()
                for tgt_idx, (tgt_style, style_tensor) in enumerate(style_refs.items(), start=1):
                    style_latent = encode_image(vae, style_tensor, device=str(device)).float()
                    output_latent, _ = model(content_latent, style_id=[tgt_idx])
                    output = decode_latent_train(vae, output_latent, args.latent_scaling_factor)[0].detach().cpu()
                    name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
                    save_image(output, output_root / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
