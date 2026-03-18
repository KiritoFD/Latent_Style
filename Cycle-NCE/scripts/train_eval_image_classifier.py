#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EVAL_CLS = _ROOT / "artifacts" / "eval_classifier" / "eval_style_image_classifier.pt"
import sys

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from utils.eval_image_classifier import build_model, write_report_json


@dataclass
class EpochStats:
    loss: float
    acc: float
    macro_recall: float


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _confusion_and_stats(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, float, float]:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(target.view(-1).tolist(), pred.view(-1).tolist()):
        cm[t, p] += 1
    total = int(cm.sum().item())
    acc = float(torch.trace(cm).item()) / float(max(1, total))
    recalls = []
    for i in range(num_classes):
        tp = float(cm[i, i].item())
        denom = float(cm[i, :].sum().item())
        recalls.append(tp / denom if denom > 0 else 0.0)
    macro_recall = float(sum(recalls) / max(1, len(recalls)))
    return cm, acc, macro_recall


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    num_classes: int,
) -> EpochStats:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    preds_all = []
    targets_all = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(train_mode):
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * imgs.size(0)
        preds_all.append(logits.argmax(dim=1).detach().cpu())
        targets_all.append(labels.detach().cpu())

    pred = torch.cat(preds_all, dim=0) if preds_all else torch.empty(0, dtype=torch.long)
    target = torch.cat(targets_all, dim=0) if targets_all else torch.empty(0, dtype=torch.long)
    _, acc, macro_recall = _confusion_and_stats(pred, target, num_classes=num_classes)
    denom = max(1, len(loader.dataset))
    return EpochStats(loss=total_loss / denom, acc=acc, macro_recall=macro_recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a robust image style classifier for evaluation.")
    parser.add_argument("--train_dir", type=Path, required=True, help="ImageFolder root for training.")
    parser.add_argument("--val_dir", type=Path, default=None, help="Optional ImageFolder root for validation.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_EVAL_CLS)
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    _seed_all(int(args.seed))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = T.Compose(
        [
            T.RandomResizedCrop(args.image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.3, 0.3, 0.2, 0.05),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    if not args.train_dir.is_dir():
        raise NotADirectoryError(f"train_dir not found: {args.train_dir}")

    full_train = ImageFolder(root=str(args.train_dir), transform=train_tf)
    class_names = list(full_train.classes)
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError(f"Need >=2 classes, got {num_classes}")

    if args.val_dir:
        if not args.val_dir.is_dir():
            raise NotADirectoryError(f"val_dir not found: {args.val_dir}")
        train_set = full_train
        val_set = ImageFolder(root=str(args.val_dir), transform=eval_tf)
        if list(val_set.classes) != class_names:
            raise ValueError("train_dir and val_dir classes mismatch.")
    else:
        full_eval = ImageFolder(root=str(args.train_dir), transform=eval_tf)
        n_total = len(full_eval)
        n_val = max(1, int(n_total * float(args.val_ratio)))
        n_train = max(1, n_total - n_val)
        gen = torch.Generator().manual_seed(int(args.seed))
        train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=gen)
        train_indices = list(train_idx.indices)
        val_indices = list(val_idx.indices)

        class _Subset(torch.utils.data.Dataset):
            def __init__(self, base: ImageFolder, indices: list[int]):
                self.base = base
                self.indices = indices
            def __len__(self) -> int:
                return len(self.indices)
            def __getitem__(self, idx: int):
                return self.base[self.indices[idx]]

        train_set = _Subset(full_train, train_indices)
        val_set = _Subset(full_eval, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch=str(args.arch), num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))

    best_macro_recall = -1.0
    best_epoch = -1
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        tr = _run_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        va = _run_epoch(model, val_loader, criterion, None, device, num_classes)
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": tr.loss,
                "train_acc": tr.acc,
                "train_macro_recall": tr.macro_recall,
                "val_loss": va.loss,
                "val_acc": va.acc,
                "val_macro_recall": va.macro_recall,
            }
        )
        print(
            f"[{epoch:03d}/{int(args.epochs):03d}] "
            f"train_acc={tr.acc:.4f} train_recall={tr.macro_recall:.4f} "
            f"val_acc={va.acc:.4f} val_recall={va.macro_recall:.4f}"
        )
        if va.macro_recall > best_macro_recall:
            best_macro_recall = va.macro_recall
            best_epoch = epoch
            args.output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_state_dict": model.state_dict(),
                "meta": {
                    "arch": str(args.arch),
                    "classes": class_names,
                    "image_size": int(args.image_size),
                    "mean": mean,
                    "std": std,
                    "best_epoch": int(best_epoch),
                    "best_val_macro_recall": float(best_macro_recall),
                },
            }
            torch.save(payload, args.output)

    report = {
        "output": str(args.output),
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir) if args.val_dir else "",
        "best_epoch": int(best_epoch),
        "best_val_macro_recall": float(best_macro_recall),
        "classes": class_names,
        "history": history,
    }
    write_report_json(args.output.with_suffix(".report.json"), report)
    print(f"Saved best checkpoint: {args.output}")
    print(f"Saved report: {args.output.with_suffix('.report.json')}")


if __name__ == "__main__":
    main()
