import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _collect_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def get_image_files(input_root: Path, style_subdirs: list[str]) -> list[Path]:
    if style_subdirs:
        all_files: list[Path] = []
        for name in style_subdirs:
            style_dir = input_root / name
            if not style_dir.exists() or not style_dir.is_dir():
                print(f"WARNING: style subdir not found, skip: {style_dir}")
                continue
            all_files.extend(_collect_images(style_dir))
        return sorted(set(all_files))
    return _collect_images(input_root)


def parse_style_subdirs(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode images into SDXL VAE latent .pt files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input root directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory for latents.")
    parser.add_argument(
        "--style_subdirs",
        type=str,
        default="",
        help="Comma-separated subdirs under input_dir to process. "
        "Example: photo,Hayao,monet,cezanne,vangogh",
    )
    parser.add_argument(
        "--vae_model",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="Hugging Face VAE model id or local path.",
    )
    parser.add_argument("--resolution", type=int, default=256, help="Resize/Crop resolution, multiple of 8.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    args = parser.parse_args()

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    style_subdirs = parse_style_subdirs(args.style_subdirs)

    if not input_root.exists():
        raise FileNotFoundError(f"Input dir not found: {input_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from diffusers import AutoencoderKL

    print(f"Loading VAE: {args.vae_model}")
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=torch.float16).to(device)
    except Exception as e:
        print(f"FP16 load failed, fallback to FP32: {e}")
        vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    vae.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, paths: list[Path]) -> None:
            self.paths = paths

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int):
            p = self.paths[idx]
            try:
                img = Image.open(p).convert("RGB")
                return transform(img), str(p)
            except Exception as e:
                print(f"Read failed: {p} ({e})")
                return torch.zeros(3, args.resolution, args.resolution), "ERROR"

    image_files = get_image_files(input_root, style_subdirs)
    if not image_files:
        print(f"No images found under: {input_root}")
        return
    print(f"Found {len(image_files)} images.")

    dataset = ImageDataset(image_files)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    latents_sq_sum = 0.0
    total_elems = 0

    print("Start encoding...")
    with torch.no_grad():
        for imgs, paths in tqdm(dataloader):
            valid = [p != "ERROR" for p in paths]
            if not any(valid):
                continue

            valid_indices = [i for i, ok in enumerate(valid) if ok]
            valid_paths = [Path(paths[i]) for i in valid_indices]
            imgs = imgs[valid].to(device=device, dtype=vae.dtype)

            latents = vae.encode(imgs).latent_dist.mode()
            latents_f32 = latents.float()
            latents_sq_sum += (latents_f32 ** 2).sum().item()
            total_elems += latents_f32.numel()

            latents_cpu = latents_f32.cpu()
            for i, src_path in enumerate(valid_paths):
                rel = src_path.relative_to(input_root)
                dst = output_root / rel.with_suffix(".pt")
                dst.parent.mkdir(parents=True, exist_ok=True)
                torch.save(latents_cpu[i], dst)

    if total_elems > 0:
        estimated_std = float(np.sqrt(latents_sq_sum / total_elems))
        recommended_scale_factor = float(1.0 / max(estimated_std, 1e-8))
        print("Done.")
        print(f"Estimated latent std: {estimated_std:.6f}")
        print(f"Recommended latent_scale_factor: {recommended_scale_factor:.6f}")
    else:
        print("Done, but no valid samples were encoded.")


if __name__ == "__main__":
    main()
