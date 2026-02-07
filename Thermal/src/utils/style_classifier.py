import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.dataset import LatentDataset

logger = logging.getLogger(__name__)


# -------------------------
# Dataset
# -------------------------
class LatentStyleDataset(Dataset):
    def __init__(self, latents: torch.Tensor, style_ids: torch.Tensor):
        self.latents = latents  # CPU tensor [N,C,H,W]
        self.style_ids = style_ids  # CPU tensor [N]

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.style_ids[idx]


def build_latent_dataset(config: dict) -> LatentStyleDataset:
    dataset = LatentDataset(
        data_root=config["data"]["data_root"],
        num_styles=config["model"]["num_styles"],
        style_subdirs=config["data"].get("style_subdirs"),
        config=config,
    )

    latents = dataset.latents_tensor.cpu()
    style_ids = torch.empty((latents.shape[0],), dtype=torch.long)
    for style_id, indices in dataset.style_indices.items():
        style_ids[indices] = style_id
    style_ids = style_ids.cpu()
    return LatentStyleDataset(latents, style_ids)


# -------------------------
# Basic CNN Classifier
# -------------------------
class BasicLatentCNN(nn.Module):
    """
    Simple CNN classifier for latent maps [B,C,H,W].
    No content-removal tricks. Just learn a strong baseline.
    """
    def __init__(self, in_channels: int, num_classes: int, width: int = 64, dropout: float = 0.1):
        super().__init__()

        w = int(width)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, w, 3, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),

            nn.Conv2d(w, w, 3, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),

            # downsample once (helps even if H/W small)
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(w * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(w * 2, w * 2, 3, padding=1),
            nn.BatchNorm2d(w * 2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Dropout(p=float(dropout)),
            nn.Linear(w * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyleClassifier(BasicLatentCNN):
    """
    Interface-compatible classifier for trainer.py.
    Accepts extra kwargs (style stats/lowpass settings) and ignores them.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        width: int = 64,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes, width=width, dropout=dropout)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    correct = 0
    total = 0
    tp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.numel()

        for c in range(num_classes):
            tp[c] += ((preds == c) & (y == c)).sum()
            fn[c] += ((preds != c) & (y == c)).sum()

    recall_per_class = (tp / (tp + fn + 1e-8)).tolist()
    recall_macro = float(torch.mean(tp / (tp + fn + 1e-8)).item())
    acc = correct / max(total, 1)
    return {"acc": acc, "recall_macro": recall_macro, "recall_per_class": recall_per_class}


def train(
    config_path: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    width: int,
    dropout: float,
    label_smoothing: float,
    use_amp: bool,
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_latent_dataset(config)

    num_classes = int(config["model"]["num_styles"])
    in_channels = int(config["model"]["latent_channels"])

    # train/val split
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = BasicLatentCNN(in_channels=in_channels, num_classes=num_classes, width=width, dropout=dropout).to(device)

    # class-weighted CE (stable for imbalance)
    labels_all = dataset.style_ids
    class_counts = torch.bincount(labels_all, minlength=num_classes).float()
    class_weights = (class_counts.sum() / torch.clamp(class_counts, min=1.0))
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        tp = torch.zeros(num_classes, device=device)
        fn = torch.zeros(num_classes, device=device)

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(
                    logits,
                    y,
                    weight=class_weights,
                    label_smoothing=float(label_smoothing),
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()

            for c in range(num_classes):
                tp[c] += ((preds == c) & (y == c)).sum()
                fn[c] += ((preds != c) & (y == c)).sum()

        scheduler.step()

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)
        train_recall_macro = float(torch.mean(tp / (tp + fn + 1e-8)).item())

        val_metrics = evaluate(model, val_loader, device=device, num_classes=num_classes)

        logger.info(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | train_recall_macro={train_recall_macro:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | val_recall_macro={val_metrics['recall_macro']:.4f} | "
            f"val_recall_per_class={val_metrics['recall_per_class']}"
        )

        # save best by val_recall_macro
        if val_metrics["recall_macro"] > best_val:
            best_val = val_metrics["recall_macro"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": {
                        "in_channels": in_channels,
                        "num_classes": num_classes,
                        "best_val_recall_macro": best_val,
                    },
                },
                output_path,
            )
            logger.info(f"Saved BEST checkpoint to {output_path} (val_recall_macro={best_val:.4f})")


def main():
    parser = argparse.ArgumentParser("Train basic latent style classifier (no content removal)")
    parser.add_argument("--config", type=str, default=str(_ROOT / "config.json"))
    parser.add_argument("--output", type=str, default=str(_ROOT / "style_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--no_amp", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    train(
        config_path=Path(args.config),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        width=args.width,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
