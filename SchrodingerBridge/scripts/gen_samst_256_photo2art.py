"""Generate SAMST photo2art baseline images on legacy256_overfit50.

SAMST is trained per target style (style_num=1). For each art target style we
load that style's checkpoint and stylize ALL 150 source images (5 styles x 30).
For the photo target there is no checkpoint, so we copy the source image as-is
(identity). This yields 4*150 + 150 = 750 images.

Output naming: {src_style}_{src_id}_to_{tgt_style}.jpg
Output dir:    /mnt/i/exp_256_photo2art/samst_256/images/

Reference inference: SaMST-main/test_model/test/test.py
  - TransformerNet(style_num=N); state_dict loaded directly from .model file
  - content: ToTensor then x.mul(255); model(content, style_id=[i])
  - style_id 0 = identity/AE branch, style_id 1..N = trained style bank entries
  - output clamped to [0, 255] then scaled to [0, 1] for save_image
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Hard-coded absolute WSL paths (remote repo layout)
SAMST_REPO = Path("/mnt/i/Github/Latent_Style/Related_Works/repos/SaMST-main")
sys.path.insert(0, str(SAMST_REPO))
from networks.transfer_net import TransformerNet  # noqa: E402

STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
# Art target styles that have a per-style checkpoint (no photo checkpoint).
ART_STYLES = ["cezanne", "Hayao", "monet", "vangogh"]
DEFAULT_CKPT_ROOT = Path(
    "/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/checkpoints/samst"
)
DEFAULT_TEST_ROOT = Path("/mnt/i/legacy256_overfit50/test")
DEFAULT_OUT_ROOT = Path("/mnt/i/exp_256_photo2art/samst_256")
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


def detect_style_num(state: dict) -> int:
    """Count Style_bank entries; total_style = entries - 1 (entry 0 is the AE/identity)."""
    n = 0
    for k in state:
        if k.startswith("style_bank.style_para_list.") and k.endswith(".params"):
            n += 1
    return max(n - 1, 1)


def load_content(path: Path, size: int, device: torch.device) -> torch.Tensor:
    tr = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ])
    return tr(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--ckpt-name", type=str, default="epoch_100.model",
                        help="Checkpoint filename inside each style subdir.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs (default: skip).")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_root / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")

    sources = collect_sources(args.test_root)
    total = len(sources) * len(STYLES)
    print(f"[samst] {len(sources)} sources x {len(STYLES)} targets = {total} images", flush=True)
    print(f"[samst] device={device} ckpt_root={args.ckpt_root}", flush=True)
    print(f"[samst] START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)

    t0 = time.time()
    generated = 0
    skipped = 0

    # --- Art target styles: per-style checkpoint, stylize all sources ---
    for tgt in ART_STYLES:
        ckpt = args.ckpt_root / tgt / args.ckpt_name
        if not ckpt.exists():
            print(f"[samst] WARN missing checkpoint {ckpt}, skipping target={tgt}", flush=True)
            continue
        print(f"[samst] target={tgt} loading {ckpt}", flush=True)

        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        style_num = detect_style_num(state)
        model = TransformerNet(style_num=style_num)
        model.load_state_dict(state)
        model.to(device).eval()
        # For a per-style checkpoint style_id=[style_num] is the trained style.
        stylized_id = [style_num]

        with torch.inference_mode():
            for src_style, src_path in sources:
                out_path = out_dir / f"{src_style}_{src_path.stem}_to_{tgt}.jpg"
                if not args.overwrite and out_path.exists():
                    skipped += 1
                    continue
                content = load_content(src_path, args.image_size, device)
                output, _ = model(content, style_id=stylized_id)
                output = output.cpu().clamp(0, 255) / 255.0
                save_image(output[0], str(out_path))
                generated += 1

        del model
        torch.cuda.empty_cache()
        print(f"[samst] target={tgt} done (generated={generated}, skipped={skipped})",
              flush=True)

    # --- Photo target: identity copy (no photo checkpoint) ---
    for src_style, src_path in sources:
        out_path = out_dir / f"{src_style}_{src_path.stem}_to_photo.jpg"
        if not args.overwrite and out_path.exists():
            skipped += 1
            continue
        shutil.copy2(src_path, out_path)
        generated += 1
    print(f"[samst] target=photo (identity) done", flush=True)

    elapsed = time.time() - t0
    print(f"[samst] generated={generated} skipped={skipped} expected={total} "
          f"in {elapsed:.1f}s", flush=True)
    print(f"[samst] out_dir={out_dir}", flush=True)
    print(f"[samst] END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
