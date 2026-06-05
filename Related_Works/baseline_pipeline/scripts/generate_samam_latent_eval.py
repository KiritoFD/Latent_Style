from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMAM_ROOT = REPO_ROOT / "Related_Works" / "repos" / "SaMam"
import sys

sys.path.insert(0, str(SAMAM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SchrodingerBridge" / "src"))

from TRAIN.lightning_module.latent_lightningmodel import LatentLightningModel  # noqa: E402
from utils.inference import encode_image  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def tensor_for_vae(path: Path, size: int, device: torch.device) -> torch.Tensor:
    tr = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    return tr(load_rgb(path)).unsqueeze(0).to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--style-names", type=str, required=True)
    args = parser.parse_args()

    style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    output_root = args.output_root.resolve()
    image_root = args.image_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LatentLightningModel.load_from_checkpoint(checkpoint_path=str(args.checkpoint), map_location=device)
    model = model.to(device).eval()
    vae = model.vae

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
                for tgt_style, style_tensor in style_refs.items():
                    style_latent = encode_image(vae, style_tensor, device=str(device)).float()
                    output_latent = model.forward(content_latent, style_latent)
                    output = model._decode_latent_train(output_latent)[0].detach().cpu()
                    name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
                    save_image(output, output_root / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
