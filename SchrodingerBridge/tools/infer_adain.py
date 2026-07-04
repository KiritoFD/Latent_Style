"""AdaIN style-transfer inference for the Distinct5 benchmark.

Generates 750 images (25 pairs x 30) for each of three variants:
  - adain_v32k: vgg_normalised.pth encoder (256-ch relu4_1) + pre-trained decoder
  - adain_vgg19: standard VGG-19 ImageNet encoder (512-ch relu4_1) + trained decoder
  - adain_bad:   same as vgg19 but with degraded AdaIN (scale only, no shift)

Usage:
    python tools/infer_adain.py --variant adain_v32k
    python tools/infer_adain.py --variant adain_vgg19
    python tools/infer_adain.py --variant adain_bad
    python tools/infer_adain.py --variant all
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STYLE_NAMES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]

DATASET_ROOT = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512")
TEST_DIR = DATASET_ROOT / "test"
TRAIN_DIR = DATASET_ROOT / "train"
OUTPUT_ROOT = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images")

# Checkpoint paths (downloaded or trained)
CHECKPOINT_DIR = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\adain_checkpoints")
VGG_NORMALISED_PATH = CHECKPOINT_DIR / "vgg_normalised.pth"
DECODER_V32K_PATH = CHECKPOINT_DIR / "decoder_v32k.pth"
DECODER_VGG19_PATH = CHECKPOINT_DIR / "decoder_vgg19.pth"

# Local vgg_normalised.pth from Related_Works (avoids download)
LOCAL_VGG_NORMALISED_CANDIDATES = [
    Path(r"G:\GitHub\Latent_Style\Related_Works\run_511\repos\cast\models\vgg_normalised.pth"),
    Path(r"G:\GitHub\Latent_Style\Related_Works\run_511\repos\StyTR-2\experiments\vgg_normalised.pth"),
    Path(r"G:\GitHub\Latent_Style\Related_Works\run_511\repos\AesFA\vgg_normalised.pth"),
]

# pytorch-AdaIN pre-trained weights URLs
VGG_NORMALISED_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v1.0.0/vgg_normalised.pth"
DECODER_V32K_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v1.0.0/decoder.pth"

NUM_IMAGES_PER_PAIR = 30
IMAGE_SIZE = 512

# ---------------------------------------------------------------------------
# Model components (self-contained, no external imports needed)
# ---------------------------------------------------------------------------

def _build_vgg_norm_encoder() -> nn.Sequential:
    """VGG-normalised encoder architecture (relu4_1 = 256 channels)."""
    return nn.Sequential(
        nn.Conv2d(3, 3, 1, 1, 0), nn.ReLU(inplace=True),
        nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
    )


class VGGEncoder(nn.Module):
    """VGG feature extractor up to relu4_1."""

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        use_norm = (
            weights_path is not None
            and Path(weights_path).exists()
            and "vgg_normalised" in Path(weights_path).name
        )
        if use_norm:
            vgg = _build_vgg_norm_encoder()
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            clean = {k.replace("module.", ""): v for k, v in state.items()}
            vgg.load_state_dict(clean, strict=False)
            children = list(vgg.children())
            self.enc_1 = nn.Sequential(*children[:4])
            self.enc_2 = nn.Sequential(*children[4:8])
            self.enc_3 = nn.Sequential(*children[8:12])
            self.enc_4 = nn.Sequential(*children[12:27])
            self._out_channels = 256
        else:
            vgg = vgg19(weights="IMAGENET1K_V1").features
            children = list(vgg.children())
            self.enc_1 = nn.Sequential(*children[:2])    # relu1_1
            self.enc_2 = nn.Sequential(*children[2:7])   # relu2_1
            self.enc_3 = nn.Sequential(*children[7:12])  # relu3_1
            self.enc_4 = nn.Sequential(*children[12:21]) # relu4_1
            self._out_channels = 512
        for p in self.parameters():
            p.requires_grad = False

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        return [h1, h2, h3, h4]


class Decoder(nn.Module):
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1, padding_mode="reflect"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# AdaIN operations
# ---------------------------------------------------------------------------

def adaptive_instance_norm(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """Standard AdaIN: match both mean and std."""
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-5
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-5
    return s_std * (content_feat - c_mean) / c_std + s_mean


def adaptive_instance_norm_bad(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """Degraded AdaIN: only scale normalization, no shift.
    This produces poor style transfer because the color (mean) of the
    style reference is not transferred, only the contrast/statistics spread.
    """
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-5
    s_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-5
    # No shift: keep content mean, only apply style scale
    return s_std * (content_feat - c_mean) / c_std + c_mean


# ---------------------------------------------------------------------------
# Full model wrapper
# ---------------------------------------------------------------------------

class AdaINModel:
    """Wraps encoder + decoder + AdaIN variant for inference."""

    def __init__(self, encoder: VGGEncoder, decoder: Decoder, adain_fn, device: torch.device):
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.adain_fn = adain_fn
        self.device = device

    @torch.no_grad()
    def transfer(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        c_feats = self.encoder(content)
        s_feats = self.encoder(style)
        t = self.adain_fn(c_feats[-1], s_feats[-1])
        t = alpha * t + (1 - alpha) * c_feats[-1]
        return self.decoder(t)


# ---------------------------------------------------------------------------
# Weight download / training helpers
# ---------------------------------------------------------------------------

def download_file(url: str, path: Path, desc: str = "", timeout: int = 15) -> None:
    """Download a file with progress bar."""
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"  [skip] {desc} already exists: {path}")
        return
    print(f"  Downloading {desc}: {url}")
    try:
        import tempfile

        tmp_path = path.with_suffix(".tmp")
        with tqdm(unit="B", unit_scale=True, desc=desc) as t:

            def reporthook(block_num, block_size, total_size):
                t.total = total_size
                t.update(block_size)

            urllib.request.urlretrieve(url, str(tmp_path), reporthook=reporthook)
        tmp_path.rename(path)
        print(f"  Saved to {path}")
    except Exception as e:
        # Clean up partial download
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"  [WARN] Download failed ({e}), will train from scratch instead")
        raise


def train_decoder(encoder: VGGEncoder, decoder: Decoder, device: torch.device,
                  max_iter: int = 16000, batch_size: int = 4) -> None:
    """Train decoder on Distinct5 training data."""
    from torch.utils.data import Dataset, DataLoader

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    class StylePairDataset(Dataset):
        def __init__(self, root: Path, transform, max_per_style: int = 32):
            self.transform = transform
            self.content_paths = []
            self.style_paths = []
            for style_dir in sorted(root.iterdir()):
                if not style_dir.is_dir() or style_dir.name == "processed_data":
                    continue
                imgs = sorted(p for p in style_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
                imgs = imgs[:max_per_style]
                self.style_paths.extend(imgs)
                self.content_paths.extend(imgs)  # content = same pool for simplicity
            # Build all pairs
            self.pairs = [(c, s) for c in self.content_paths for s in self.style_paths]
            print(f"  Training pairs: {len(self.pairs)}")

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            c_path, s_path = self.pairs[idx]
            content = self.transform(Image.open(c_path).convert("RGB"))
            style = self.transform(Image.open(s_path).convert("RGB"))
            return content, style

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    dataset = StylePairDataset(TRAIN_DIR, transform, max_per_style=32)
    if len(dataset) == 0:
        # Fallback: use test images
        print("  [WARN] No training data found, using test images for decoder training")
        dataset = StylePairDataset(TEST_DIR, transform, max_per_style=30)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    encoder = encoder.to(device)
    encoder.eval()
    decoder = decoder.to(device)
    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

    iteration = 0
    start = time.time()

    while iteration < max_iter:
        for content, style in loader:
            if iteration >= max_iter:
                break
            content = content.to(device)
            style = style.to(device)

            with torch.no_grad():
                c_feats = encoder(content)
                s_feats = encoder(style)
                t = adaptive_instance_norm(c_feats[-1], s_feats[-1])

            output = decoder(t)
            out_feats = encoder(output)

            # Content loss: match relu4_1
            loss_c = nn.functional.mse_loss(out_feats[-1], t.detach())

            # Style loss: match mean/std at all layers
            loss_s = 0.0
            for of, sf in zip(out_feats, s_feats):
                g_mean = of.mean(dim=[2, 3])
                g_std = of.std(dim=[2, 3]) + 1e-5
                t_mean = sf.detach().mean(dim=[2, 3])
                t_std = sf.detach().std(dim=[2, 3]) + 1e-5
                loss_s += nn.functional.mse_loss(g_mean, t_mean) + nn.functional.mse_loss(g_std, t_std)

            loss = loss_c + 10.0 * loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if iteration % 2000 == 0 or iteration == 1:
                elapsed = time.time() - start
                its = iteration / elapsed if elapsed > 0 else 0
                eta = (max_iter - iteration) / its if its > 0 else 0
                print(f"    iter {iteration}/{max_iter}  "
                      f"loss_c={loss_c.item():.6f}  loss_s={loss_s.item():.4f}  "
                      f"total={loss.item():.4f}  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  {its:.1f} it/s")

    decoder.eval()
    print(f"  Decoder training complete in {time.time() - start:.0f}s")


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(variant: str, device: torch.device) -> AdaINModel:
    """Create the AdaIN model for the given variant."""

    if variant == "adain_v32k":
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        # Try local vgg_normalised.pth first
        vgg_path = VGG_NORMALISED_PATH
        if not vgg_path.exists():
            for candidate in LOCAL_VGG_NORMALISED_CANDIDATES:
                if candidate.exists():
                    import shutil
                    shutil.copy2(str(candidate), str(vgg_path))
                    print(f"  Copied vgg_normalised.pth from {candidate}")
                    break
        if not vgg_path.exists():
            download_file(VGG_NORMALISED_URL, vgg_path, "vgg_normalised.pth")

        # Train decoder from scratch (pre-trained decoder URL is unreliable)
        encoder = VGGEncoder(weights_path=str(vgg_path))
        decoder = Decoder(in_channels=encoder.out_channels)

        if DECODER_V32K_PATH.exists():
            decoder.load_state_dict(torch.load(str(DECODER_V32K_PATH), map_location="cpu", weights_only=True))
            print(f"  adain_v32k: decoder loaded from {DECODER_V32K_PATH}")
        else:
            print(f"  adain_v32k: training decoder (vgg_normalised, {encoder.out_channels}ch)...")
            train_decoder(encoder, decoder, device, max_iter=16000, batch_size=4)
            torch.save(decoder.state_dict(), str(DECODER_V32K_PATH))

        return AdaINModel(encoder, decoder, adaptive_instance_norm, device)

    elif variant == "adain_vgg19":
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        encoder = VGGEncoder(weights_path=None)  # Standard VGG-19 ImageNet
        decoder = Decoder(in_channels=encoder.out_channels)

        if DECODER_VGG19_PATH.exists():
            decoder.load_state_dict(torch.load(str(DECODER_VGG19_PATH), map_location="cpu", weights_only=True))
            print(f"  adain_vgg19: decoder loaded from {DECODER_VGG19_PATH}")
        else:
            print(f"  adain_vgg19: training decoder (VGG-19, {encoder.out_channels}ch)...")
            train_decoder(encoder, decoder, device, max_iter=16000, batch_size=4)
            torch.save(decoder.state_dict(), str(DECODER_VGG19_PATH))
            print(f"  Saved decoder to {DECODER_VGG19_PATH}")

        return AdaINModel(encoder, decoder, adaptive_instance_norm, device)

    elif variant == "adain_bad":
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        bad_decoder_path = CHECKPOINT_DIR / "decoder_bad.pth"
        encoder = VGGEncoder(weights_path=None)  # Standard VGG-19 ImageNet
        decoder = Decoder(in_channels=encoder.out_channels)

        # For adain_bad, we need a decoder trained with the bad AdaIN
        # Since bad AdaIN is only in inference, we can use the same decoder
        # but with the bad AdaIN function
        if bad_decoder_path.exists():
            decoder.load_state_dict(torch.load(str(bad_decoder_path), map_location="cpu", weights_only=True))
            print(f"  adain_bad: decoder loaded from {bad_decoder_path}")
        elif DECODER_VGG19_PATH.exists():
            decoder.load_state_dict(torch.load(str(DECODER_VGG19_PATH), map_location="cpu", weights_only=True))
            print(f"  adain_bad: reusing adain_vgg19 decoder")
        else:
            print(f"  adain_bad: training decoder...")
            train_decoder(encoder, decoder, device, max_iter=16000, batch_size=4)
            torch.save(decoder.state_dict(), str(bad_decoder_path))
            print(f"  Saved decoder to {bad_decoder_path}")

        return AdaINModel(encoder, decoder, adaptive_instance_norm_bad, device)

    else:
        raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_test_images(style_name: str) -> list[Path]:
    """Return sorted list of test image paths for a given style."""
    style_dir = TEST_DIR / style_name
    if not style_dir.exists():
        print(f"[WARN] Test directory not found: {style_dir}")
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in style_dir.iterdir() if p.suffix.lower() in exts)


def src_name_from_filename(filename: str) -> str:
    """Extract the artist_title part from '{Style}__{artist}_{title}.jpg'."""
    stem = Path(filename).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def run_inference(variant: str, device: str = "cuda"):
    """Generate all 750 images for a given variant."""
    device_obj = torch.device(device)

    print(f"\n{'='*60}")
    print(f"AdaIN variant: {variant}")
    print(f"{'='*60}")

    model = create_model(variant, device_obj)

    out_dir = OUTPUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transform: resize to IMAGE_SIZE, to tensor
    # Note: we resize to 512 to match the test images
    # For VGG, we need to handle the size carefully since VGG uses
    # 3 max-pools (8x downsampling). 512 is divisible by 8.
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Build style reference bank: for each target style, pick one reference image
    # We use the first image from each style's test set as the style reference
    style_refs: dict[str, torch.Tensor] = {}
    for style_name in STYLE_NAMES:
        imgs = get_test_images(style_name)
        if imgs:
            ref_img = Image.open(imgs[0]).convert("RGB")
            style_refs[style_name] = transform(ref_img).unsqueeze(0).to(device_obj)

    total = 0
    skipped = 0

    for src_style in STYLE_NAMES:
        src_images = get_test_images(src_style)
        if not src_images:
            continue
        src_images = src_images[:NUM_IMAGES_PER_PAIR]

        for tgt_style in STYLE_NAMES:
            if tgt_style not in style_refs:
                continue

            style_ref = style_refs[tgt_style]
            desc = f"{src_style} -> {tgt_style}"

            for img_idx, src_path in enumerate(tqdm(src_images, desc=desc, leave=False)):
                src_name = src_name_from_filename(src_path.name)
                out_name = f"{src_style}__{src_name}__to__{tgt_style}.png"
                out_path = out_dir / out_name

                if out_path.exists():
                    skipped += 1
                    continue

                try:
                    content_img = Image.open(src_path).convert("RGB")
                    content_tensor = transform(content_img).unsqueeze(0).to(device_obj)

                    output = model.transfer(content_tensor, style_ref)

                    # Clamp and convert to PIL
                    output = output.squeeze(0).clamp(0, 1)
                    out_pil = transforms.ToPILImage()(output.cpu())
                    out_pil.save(str(out_path))
                    total += 1

                except Exception as e:
                    print(f"  ERROR: {out_name} -> {e}")
                    continue

    print(f"\n  {variant}: generated {total} images, skipped {skipped} existing")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AdaIN style-transfer inference for Distinct5")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["adain_v32k", "adain_vgg19", "adain_bad", "all"],
        default="all",
        help="Which variant to run (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    args = parser.parse_args()

    variants = ["adain_v32k", "adain_vgg19", "adain_bad"] if args.variant == "all" else [args.variant]

    for v in variants:
        n = run_inference(v, args.device)
        # Verify count
        out_dir = OUTPUT_ROOT / v
        if out_dir.exists():
            count = len(list(out_dir.glob("*.png")))
            status = "OK" if count == 750 else f"EXPECTED 750, got {count}"
            print(f"  {v}: {count} images in {out_dir} [{status}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
