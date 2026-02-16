from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
from torchvision.transforms import v2 as T

from eval_image_classifier import build_model


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Sample:
    path: Path
    label: int


class StyleImageDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: T.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        # read_image returns uint8 [C,H,W]
        img = read_image(str(s.path)).float() / 255.0
        img = self.transform(img)
        return img, s.label


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_samples(data_root: Path, style_subdirs: Iterable[str]) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    classes = [str(x) for x in style_subdirs]
    for sid, style_name in enumerate(classes):
        sdir = data_root / style_name
        if not sdir.exists():
            continue
        for p in sorted(sdir.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                samples.append(Sample(path=p, label=sid))
    return samples, classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.json")
    parser.add_argument("--data_root", type=str, default="", help="Override style image root")
    parser.add_argument("--output", type=str, default="", help="Override output checkpoint path")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    cfg_path = (script_dir / args.config).resolve()
    cfg = _load_json(cfg_path)

    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    style_subdirs = list(data_cfg.get("style_subdirs", []))
    if not style_subdirs:
        raise ValueError("config.data.style_subdirs is empty")

    if args.data_root:
        data_root = Path(args.data_root).expanduser()
        if not data_root.is_absolute():
            data_root = (script_dir / data_root).resolve()
    else:
        root_raw = str(train_cfg.get("full_eval_classifier_data_dir", "")).strip()
        if root_raw:
            data_root = Path(root_raw).expanduser()
            if not data_root.is_absolute():
                data_root = (cfg_path.parent / data_root).resolve()
        else:
            test_root = str(train_cfg.get("test_image_dir", "")).strip()
            if not test_root:
                raise ValueError("Need training.full_eval_classifier_data_dir or training.test_image_dir in config.")
            data_root = Path(test_root).expanduser()
            if not data_root.is_absolute():
                data_root = (cfg_path.parent / data_root).resolve()

    if args.output:
        out_path = Path(args.output).expanduser()
        if not out_path.is_absolute():
            out_path = (script_dir / out_path).resolve()
    else:
        out_path = (
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "eval_classifier"
            / "eval_style_image_classifier.pt"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    image_size = int(train_cfg.get("full_eval_classifier_image_size", 224))
    batch_size = int(train_cfg.get("full_eval_classifier_batch_size", 32))
    epochs = int(train_cfg.get("full_eval_classifier_epochs", 10))
    lr = float(train_cfg.get("full_eval_classifier_lr", 1e-4))
    val_ratio = float(train_cfg.get("full_eval_classifier_val_ratio", 0.2))
    arch = str(train_cfg.get("full_eval_classifier_arch", "resnet50"))
    pretrained = bool(train_cfg.get("full_eval_classifier_pretrained", True))
    mean = list(train_cfg.get("full_eval_classifier_mean", [0.485, 0.456, 0.406]))
    std = list(train_cfg.get("full_eval_classifier_std", [0.229, 0.224, 0.225]))
    num_workers = int(train_cfg.get("full_eval_classifier_num_workers", 4))

    samples, classes = _collect_samples(data_root, style_subdirs)
    if not samples:
        raise RuntimeError(f"No style images found under {data_root}")

    tf_train = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
            T.Normalize(mean=mean, std=std),
        ]
    )
    tf_val = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.Normalize(mean=mean, std=std),
        ]
    )

    full_ds = StyleImageDataset(samples=samples, transform=tf_train)
    val_len = max(1, int(round(len(full_ds) * val_ratio)))
    train_len = max(1, len(full_ds) - val_len)
    val_len = len(full_ds) - train_len
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=g)
    # Use eval transform for val split.
    val_ds.dataset = StyleImageDataset(samples=samples, transform=tf_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch=arch, num_classes=len(classes), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = -1.0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x).argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        acc = float(correct / max(total, 1))
        print(f"[{ep}/{epochs}] val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training finished without valid checkpoint state.")

    payload = {
        "model_state_dict": best_state,
        "meta": {
            "arch": arch,
            "classes": classes,
            "image_size": image_size,
            "mean": mean,
            "std": std,
            "best_val_acc": best_acc,
            "data_root": str(data_root),
        },
    }
    torch.save(payload, out_path)
    print(f"Saved image classifier: {out_path} (best_val_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()

