"""
Custom StyTR-2 inference script.
Runs style transfer following our evaluation protocol:
  - Each content image -> each target style
  - Uses first image from each target style dir as style reference
  - Output naming: {src_style}__{src_stem}__to__{tgt_style}.png

Usage:
  python run_stytr2_inference.py \
      --test_dir <test root with style subdirs> \
      --output_dir <output dir> \
      --style_names <comma-separated style names> \
      --vgg <path to vgg_normalised.pth> \
      --decoder_path <path to decoder_iter_160000.pth> \
      --trans_path <path to transformer_iter_160000.pth> \
      --embedding_path <path to embedding_iter_160000.pth> \
      --stytr2_root <path to StyTR-2 repo root> \
      --num_src 30 \
      --content_size 512 \
      --style_size 512
"""
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def add_stytr2_to_path(stytr2_root: str):
    """Add StyTR-2 repo to sys.path so its modules can be imported."""
    root = Path(stytr2_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Also add parent for 'models' and 'util' imports
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))


def monkey_patch_forward(stytr2_root: str):
    """Monkey-patch StyTrans.forward to only return Ics (test mode)."""
    import importlib
    # Force reimport of models.StyTR
    if 'models.StyTR' in sys.modules:
        del sys.modules['models.StyTR']
    if 'models' in sys.modules:
        del sys.modules['models']

    import models.StyTR as StyTR_module
    from util.misc import nested_tensor_from_tensor_list
    from function import normal

    def forward_test(self, samples_c, samples_s):
        """Inference-only forward: returns only Ics."""
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)

        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)

        pos_s = None
        pos_c = None
        mask = None
        hs = self.transformer(style, mask, content, pos_c, pos_s)
        Ics = self.decode(hs)
        return Ics

    StyTR_module.StyTrans.forward = forward_test
    print("[INFO] Monkey-patched StyTrans.forward for test mode")


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def style_transform(h, w):
    transform_list = []
    transform_list.append(transforms.CenterCrop((h, w)))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def content_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True, help="Test dataset root with style subdirs")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--style_names", required=True, help="Comma-separated style names")
    parser.add_argument("--vgg", default="./experiments/vgg_normalised.pth")
    parser.add_argument("--decoder_path", default="./experiments/decoder_iter_160000.pth")
    parser.add_argument("--trans_path", default="./experiments/transformer_iter_160000.pth")
    parser.add_argument("--embedding_path", default="./experiments/embedding_iter_160000.pth")
    parser.add_argument("--stytr2_root", required=True, help="Path to StyTR-2 repo root")
    parser.add_argument("--num_src", type=int, default=30)
    parser.add_argument("--content_size", type=int, default=512)
    parser.add_argument("--style_size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--position_embedding", default="sine", choices=("sine", "learned"))
    parser.add_argument("--hidden_dim", default=512, type=int)
    args = parser.parse_args()

    style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup StyTR-2 imports
    add_stytr2_to_path(args.stytr2_root)
    monkey_patch_forward(args.stytr2_root)

    import models.StyTR as StyTR
    import models.transformer as transformer_module

    # Load models
    print("[INFO] Loading VGG...")
    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg, map_location="cpu"))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    Trans = transformer_module.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()

    from collections import OrderedDict

    for path, target_name in [
        (args.decoder_path, "decoder"),
        (args.trans_path, "Trans"),
        (args.embedding_path, "embedding"),
    ]:
        state_dict = torch.load(path, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        if target_name == "decoder":
            decoder.load_state_dict(new_state_dict)
        elif target_name == "Trans":
            Trans.load_state_dict(new_state_dict)
        elif target_name == "embedding":
            embedding.load_state_dict(new_state_dict)

    # Build args namespace for StyTrans
    class ArgsNamespace:
        def __init__(self, args):
            self.position_embedding = args.position_embedding
            self.hidden_dim = args.hidden_dim

    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, ArgsNamespace(args))
    network.eval()
    network.to(device)

    content_tf = test_transform(args.content_size, True)
    style_tf = test_transform(args.style_size, True)
    content_transform_fn = content_transform()

    # Collect content images: all images from all style subdirs
    # For each content image, transfer to each target style
    total = 0
    for src_style in style_names:
        src_dir = test_dir / src_style
        if not src_dir.exists():
            print(f"[WARN] Source style dir not found: {src_dir}")
            continue
        content_files = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])[:args.num_src]
        for content_path in content_files:
            for tgt_style in style_names:
                # Skip identity transfer (src_style == tgt_style) -- actually keep it for complete eval
                # Get style reference image (first from target style dir)
                tgt_dir = test_dir / tgt_style
                style_files = sorted([p for p in tgt_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
                if not style_files:
                    print(f"[WARN] No style images in {tgt_dir}")
                    continue
                style_path = style_files[0]

                # Output naming: {src_style}__{src_stem}__to__{tgt_style}.png
                src_stem = content_path.stem
                out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                out_path = output_dir / out_name
                if out_path.exists():
                    continue

                try:
                    content = content_transform_fn(Image.open(content_path).convert("RGB"))
                    h, w, c = np.shape(content)
                    stf = style_transform(h, w)
                    style = style_tf(Image.open(style_path).convert("RGB"))

                    style = style.to(device).unsqueeze(0)
                    content = content.to(device).unsqueeze(0)

                    with torch.no_grad():
                        output = network(content, style)
                    output = output.cpu()

                    save_image(output, str(out_path))
                    total += 1
                    if total % 50 == 0:
                        print(f"  Generated {total} images...")
                except Exception as e:
                    print(f"[WARN] Failed on {content_path.name} -> {tgt_style}: {e}")

                torch.cuda.empty_cache()

    print(f"[INFO] Total generated: {total} images in {output_dir}")


if __name__ == "__main__":
    main()
