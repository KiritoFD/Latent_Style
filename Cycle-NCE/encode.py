import argparse
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LATENT_SCALE = 0.18215


class ImageToLatentDataset(Dataset):
    def __init__(self, jobs: Sequence[Tuple[Path, Path]], image_size: int) -> None:
        self.jobs = list(jobs)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, idx: int):
        image_path, latent_path = self.jobs[idx]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        tensor = tensor.mul(2.0).sub(1.0)
        return tensor, str(latent_path)


def collect_jobs(root_dir: Path, latent_dir: Path, overwrite: bool) -> List[Tuple[Path, Path]]:
    jobs: List[Tuple[Path, Path]] = []
    for image_path in root_dir.rglob("*"):
        if (not image_path.is_file()) or (image_path.suffix.lower() not in EXTS):
            continue
        rel_dir = image_path.parent.relative_to(root_dir)
        out_dir = latent_dir / rel_dir
        out_path = out_dir / f"{image_path.stem}.pt"
        if overwrite or (not out_path.exists()):
            jobs.append((image_path, out_path))
    return jobs


def save_latent(latent_cpu: torch.Tensor, latent_path: str) -> None:
    out_path = Path(latent_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latent_cpu, out_path)


def resolve_device(requested_device: str) -> torch.device:
    req = str(requested_device).lower()
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda" and (not torch.cuda.is_available()):
        print("CUDA not available, fallback to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_autocast_dtype(dtype_name: str, device: torch.device):
    name = str(dtype_name).lower()
    if device.type != "cuda" or name == "fp32":
        return None
    if name == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("bf16 not supported on this GPU, fallback to fp16.")
        return torch.float16
    return torch.float16


def run(args) -> None:
    root_dir = Path(args.root_dir).expanduser().resolve()
    latent_dir = Path(args.latent_dir).expanduser().resolve()
    latent_dir.mkdir(parents=True, exist_ok=True)

    if not root_dir.exists():
        raise FileNotFoundError(f"Root image directory not found: {root_dir}")

    jobs = collect_jobs(root_dir=root_dir, latent_dir=latent_dir, overwrite=bool(args.overwrite))
    print(f"Found {len(jobs)} images to encode.")
    if not jobs:
        return

    device = resolve_device(args.device)
    amp_dtype = resolve_autocast_dtype(args.amp_dtype, device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    from diffusers import AutoencoderKL

    print(f"Loading VAE: {args.vae_model}")
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    vae.eval()
    if device.type == "cuda":
        vae = vae.to(memory_format=torch.channels_last)

    dataset = ImageToLatentDataset(jobs=jobs, image_size=int(args.image_size))
    pin_memory = bool((device.type == "cuda") and (not args.no_pin_memory))
    num_workers = max(0, int(args.num_workers))
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": max(1, int(args.batch_size)),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    dataloader = DataLoader(**loader_kwargs)

    max_pending = max(8, int(args.save_workers) * 4)
    pending = set()
    save_workers = max(1, int(args.save_workers))

    print(
        f"Start encoding | device={device.type} batch={loader_kwargs['batch_size']} "
        f"workers={num_workers} save_workers={save_workers} pin_memory={pin_memory}"
    )

    with ThreadPoolExecutor(max_workers=save_workers) as pool:
        progress = tqdm(total=len(dataset), desc="Encoding", unit="img")
        for images, latent_paths in dataloader:
            if device.type == "cuda":
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                images = images.to(device)

            with torch.inference_mode():
                if amp_dtype is None:
                    latents = vae.encode(images).latent_dist.sample()
                else:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        latents = vae.encode(images).latent_dist.sample()
                latents = (latents * LATENT_SCALE).to(device="cpu", dtype=torch.float32)

            for latent, latent_path in zip(latents, latent_paths):
                # Keep legacy saved shape [1,C,H,W].
                latent_cpu = latent.unsqueeze(0).contiguous()
                fut = pool.submit(save_latent, latent_cpu, latent_path)
                pending.add(fut)
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for d in done:
                        d.result()

            progress.update(len(latent_paths))

        if pending:
            done, _ = wait(pending)
            for d in done:
                d.result()
        progress.close()

    print(f"Done. Saved latents to: {latent_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch encode images to VAE latents.")
    parser.add_argument("--root_dir", type=str, default="/mnt/g/GitHub/Latent_Style/style_data/train")
    parser.add_argument("--latent_dir", type=str, default="../latent-256")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--save_workers", type=int, default=8)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing latent files.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
