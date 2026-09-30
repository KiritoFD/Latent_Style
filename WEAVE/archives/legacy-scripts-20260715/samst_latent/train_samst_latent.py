"""SAMST-latent training script.

Trains TransformerNetLatent on VAE latents (4x32x32) for 5 WikiArt styles.
Uses direct MSE content loss + Gram-matrix style loss on latent features
(VGG doesn't apply to latents; we use a small conv feature extractor instead).

Usage:
    python scripts/samst_latent/train_samst_latent.py \
        --latent-root I:/wikiart_distinct5_samam_512_latent256/train \
        --style-names Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e \
        --save-dir I:/exp_samst_latent \
        --epochs 15 --batch-size 8 --lr 1e-4
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from networks_latent import TransformerNetLatent


STYLE_NAMES_DEFAULT = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


class LatentStyleDataset(Dataset):
    """Loads pre-encoded VAE latents from .pt files.

    Directory layout:
        <root>/<style_name>/*.pt   (each .pt is a (4, 32, 32) tensor or dict with "latent")
    Or packed cache:
        <root>/.latent_cache/packed/00_<StyleName>.pt  (dict with "latents" key, shape (N, 4, 32, 32))
    """

    def __init__(self, root, style_names, max_per_style=0):
        self.root = Path(root)
        self.style_names = list(style_names)
        self.samples = []  # list of (style_id, tensor_idx, style_name)
        self.style_tensors = {}  # style_id -> (N, C, H, W) tensor

        for sid, sname in enumerate(self.style_names):
            # Try multiple packed cache layouts (some use 1-level, some 2-level)
            packed_candidates = [
                self.root / ".latent_cache" / "packed" / f"00_{sname}.pt",
                self.root / ".latent_cache" / "packed" / "packed" / f"00_{sname}.pt",
                self.root / ".latent_cache" / "packed" / f"{sid:02d}_{sname}.pt",
                self.root / ".latent_cache" / "packed" / "packed" / f"{sid:02d}_{sname}.pt",
            ]
            packed = None
            for c in packed_candidates:
                if c.exists():
                    packed = c
                    break
            if packed is not None:
                obj = torch.load(packed, map_location="cpu", weights_only=False)
                if isinstance(obj, dict):
                    tensor = obj.get("latents", obj.get("tensor", None))
                    if tensor is None:
                        # Try first tensor value
                        for v in obj.values():
                            if torch.is_tensor(v):
                                tensor = v
                                break
                    if tensor is None:
                        raise KeyError(f"Packed cache {packed} has no tensor")
                else:
                    tensor = obj
                tensor = tensor.float()
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                print(f"[INFO] Loaded {sname}: {tensor.shape} from {packed}")
            else:
                # fallback: load individual .pt files
                style_dir = self.root / sname
                files = sorted(style_dir.glob("*.pt"))
                if not files:
                    raise FileNotFoundError(f"No latents for style {sname} at {style_dir}")
                tensors = []
                for f in files:
                    obj = torch.load(f, map_location="cpu", weights_only=False)
                    if isinstance(obj, dict):
                        obj = obj.get("latent", obj)
                    tensors.append(torch.as_tensor(obj).float().squeeze(0))
                tensor = torch.stack(tensors, dim=0)
                print(f"[INFO] Loaded {sname}: {tensor.shape} from {len(files)} individual files")

            if max_per_style > 0 and tensor.size(0) > max_per_style:
                tensor = tensor[:max_per_style]

            self.style_tensors[sid] = tensor
            for i in range(tensor.size(0)):
                self.samples.append((sid, i, sname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, tidx, sname = self.samples[idx]
        return self.style_tensors[sid][tidx], sid


class LatentFeatureExtractor(nn.Module):
    """Small conv feature extractor for Gram-matrix style loss on latents.

    Replaces VGG (which only works on pixels). 4 conv layers with downsampling.
    """

    def __init__(self, in_ch=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(),
        )

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


def gram_matrix(x):
    b, c, h, w = x.size()
    f = x.view(b, c, h * w)
    return f.bmm(f.transpose(1, 2)) / (c * h * w)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Data
    dataset = LatentStyleDataset(args.latent_root, args.style_names, args.max_per_style)
    print(f"[INFO] Dataset: {len(dataset)} samples, {len(args.style_names)} styles")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, pin_memory=False)

    style_num = len(args.style_names)

    # Model
    model = TransformerNetLatent(style_num=style_num, in_channels=4, latent_channels=4)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] TransformerNetLatent params: {n_params:,}")

    # Feature extractor for style loss (fixed, not trained)
    feat_extractor = LatentFeatureExtractor(in_ch=4).to(device).eval()
    for p in feat_extractor.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[INFO] Training {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"[INFO] Save dir: {args.save_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        agg_content = 0.0
        agg_style = 0.0
        agg_ae = 0.0
        n_batches = 0
        t_epoch = time.time()

        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            n_batch = x.size(0)

            # Random style ids (1..style_num); 0 = AE/identity
            style_ids = [random.randint(1, style_num) for _ in range(n_batch)]
            # Double batch: first half = stylize, second half = identity (style_id=0)
            all_style_ids = style_ids + [0] * n_batch
            x_doubled = x.repeat(2, 1, 1, 1)

            y, _ = model(x_doubled, style_id=all_style_ids)
            y_stylized, y_identity = torch.split(y, n_batch, dim=0)

            # Content loss: MSE on latent directly (no VGG)
            content_loss = args.content_weight * mse_loss(y_stylized, x)

            # Style loss: Gram matrix on latent features
            # Sample one style image per batch item for style target
            style_targets = []
            for sid in style_ids:
                sname = args.style_names[sid - 1]
                style_tensor = dataset.style_tensors[sid - 1]
                rand_idx = random.randint(0, style_tensor.size(0) - 1)
                style_targets.append(style_tensor[rand_idx])
            style_batch = torch.stack(style_targets).to(device)

            feats_y = feat_extractor(y_stylized)
            feats_s = feat_extractor(style_batch)
            style_loss = 0.0
            for fy, fs in zip(feats_y, feats_s):
                gy = gram_matrix(fy)
                gs = gram_matrix(fs)
                style_loss = style_loss + mse_loss(gy, gs)
            style_loss = style_loss * args.style_weight

            # AE loss: identity reconstruction
            ae_loss = args.ae_weight * mse_loss(y_identity, x)

            total_loss = content_loss + style_loss + ae_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            agg_content += content_loss.item()
            agg_style += style_loss.item()
            agg_ae += ae_loss.item()
            n_batches += 1

            if (batch_idx + 1) % args.log_interval == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                      f"content={agg_content/n_batches:.4f} "
                      f"style={agg_style/n_batches:.4f} "
                      f"ae={agg_ae/n_batches:.4f} "
                      f"total={ (agg_content+agg_style+agg_ae)/n_batches:.4f}")

        # LR step decay
        if epoch % args.step_size == 0:
            for pg in optimizer.param_groups:
                pg['lr'] *= args.weight_decay
            print(f"  LR -> {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch:02d}.model")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved {ckpt_path}")

        elapsed = time.time() - t_epoch
        vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"  Epoch {epoch} done in {elapsed:.1f}s, VRAM peak={vram:.2f}GB")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-root", required=True)
    p.add_argument("--style-names", default=",".join(STYLE_NAMES_DEFAULT))
    p.add_argument("--save-dir", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--content-weight", type=float, default=1.0)
    p.add_argument("--style-weight", type=float, default=10.0)
    p.add_argument("--ae-weight", type=float, default=1.0)
    p.add_argument("--step-size", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=0.5)
    p.add_argument("--save-interval", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--max-per-style", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    train(args)
