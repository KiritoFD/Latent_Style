from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MODEL_ARCH = "styleeval_smallcnn_v1"
DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT = Path("../../artifacts/eval_classifier/eval_style_image_classifier.pt")
DEFAULT_REPORT_PATH = Path("../../artifacts/eval_classifier/eval_style_image_classifier_report.json")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_num_workers(num_workers: int) -> int:
    if int(num_workers) >= 0:
        return int(num_workers)
    cpu = os.cpu_count() or 8
    return max(2, min(8, cpu // 2))


def _list_image_paths_by_class(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        imgs = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if imgs:
            out[d.name] = imgs
    return out


def _resolve_image_root(cfg: dict, config_path: Path) -> Path:
    raw = str(cfg.get("data", {}).get("image_root", "")).strip()
    if not raw:
        raise FileNotFoundError("config.data.image_root is empty")
    p = Path(raw).expanduser()
    cfg_dir = config_path.resolve().parent
    bases = [cfg_dir, cfg_dir.parent, cfg_dir.parent.parent, Path.cwd(), Path(__file__).resolve().parents[2]]
    cands = [p] if p.is_absolute() else [(b / p).resolve() for b in bases]

    seen = set()
    uniq = []
    for c in cands:
        k = str(c)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)

    print(f"[classify] config path: {config_path}")
    print(f"[classify] raw image_root: {raw}")
    for c in uniq:
        exists = c.exists()
        is_dir = c.is_dir() if exists else False
        cls_count = len(_list_image_paths_by_class(c)) if (exists and is_dir) else 0
        print(f"  - {c} | exists={exists} is_dir={is_dir} class_folders_with_images={cls_count}")
        if exists and is_dir and cls_count > 0:
            return c
    raise FileNotFoundError(f"No valid image-root for config.data.image_root={raw}")


class SmallBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyleEvalClassifier(nn.Module):
    """
    Lightweight classifier for style evaluation.
    - Small parameter count
    - Stable FP32 training
    - Good enough capacity for 5-way style classification
    """

    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            SmallBlock(3, 24, stride=2),   # 256 -> 128
            SmallBlock(24, 48, stride=2),  # 128 -> 64
            SmallBlock(48, 80, stride=2),  # 64 -> 32
            SmallBlock(80, 128, stride=2), # 32 -> 16
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    # Keep signature for compatibility with callers.
    _ = pretrained
    return StyleEvalClassifier(num_classes=num_classes)


class EvalImageClassifier:
    def __init__(
        self,
        model: nn.Module,
        classes: list[str],
        mean: list[float] | tuple[float, ...],
        std: list[float] | tuple[float, ...],
        image_size: int,
        device: str,
    ):
        self.model = model.to(device).eval()
        self.classes = list(classes)
        self.device = str(device)
        self.image_size = int(image_size)
        self.preprocess = T.Compose([T.Resize((self.image_size, self.image_size)), T.Normalize(mean=list(mean), std=list(std))])

    @torch.no_grad()
    def predict_indices(self, batch_01: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(batch_01.to(self.device, dtype=torch.float32))
        logits = self.model(x)
        return logits.argmax(dim=1)


def load_eval_image_classifier(ckpt_path: Path, device: str) -> EvalImageClassifier:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid classifier checkpoint format")
    meta = payload.get("meta", {})
    classes = list(meta.get("classes", []))
    if not classes:
        raise ValueError("Checkpoint missing meta.classes")
    image_size = int(meta.get("image_size", 256))
    mean = meta.get("mean", [0.485, 0.456, 0.406])
    std = meta.get("std", [0.229, 0.224, 0.225])
    arch = str(meta.get("arch", MODEL_ARCH))
    if arch != MODEL_ARCH:
        raise ValueError(f"Unsupported arch in checkpoint: {arch}")
    model = build_model(num_classes=len(classes), pretrained=False)
    state = payload.get("model_state_dict")
    if state is None:
        raise ValueError("Checkpoint missing model_state_dict")
    model.load_state_dict(state, strict=True)
    return EvalImageClassifier(model=model, classes=classes, mean=mean, std=std, image_size=image_size, device=device)


def write_report_json(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class _ImageClsDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), int(y)


def _stratified_split(cls_map: dict[str, list[Path]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    classes = sorted(cls_map.keys())
    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    for cid, c in enumerate(classes):
        paths = cls_map[c][:]
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_ratio)))
        n_val = min(n_val, max(1, len(paths) - 1))
        for p in paths[:n_val]:
            val_items.append((p, cid))
        for p in paths[n_val:]:
            train_items.append((p, cid))
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items, classes


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, Any]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    n = 0
    loss_sum = 0.0
    correct = 0
    conf_sum = 0.0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            loss_sum += float(loss.item()) * int(x.size(0))
            n += int(x.size(0))
            correct += int((pred == y).sum().item())
            conf_sum += float(probs.max(dim=1).values.sum().item())

            yt = y.detach().cpu().long()
            yp = pred.detach().cpu().long()
            for i in range(yt.numel()):
                conf[int(yt[i].item()), int(yp[i].item())] += 1

    recalls = []
    for c in range(num_classes):
        tp = float(conf[c, c].item())
        fn = float(conf[c, :].sum().item() - conf[c, c].item())
        recalls.append(tp / max(1.0, tp + fn))
    macro_recall = float(sum(recalls) / max(1, len(recalls)))
    return {
        "loss": loss_sum / max(1, n),
        "acc": correct / max(1, n),
        "macro_recall": macro_recall,
        "mean_confidence": conf_sum / max(1, n),
        "per_class_recall": recalls,
    }


def train_from_config(
    config_path: Path,
    out_ckpt: Path = DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT,
    out_report: Path = DEFAULT_REPORT_PATH,
    *,
    epochs: int = 120,
    batch_size: int = 128,
    lr: float = 2e-4,
    weight_decay: float = 5e-2,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = -1,
    min_epochs: int = 10,
    target_acc: float = 0.95,
    target_recall: float = 0.95,
    target_confidence_min: float = 0.70,
    target_confidence_max: float = 0.98,
) -> dict[str, Any]:
    cfg_path = Path(config_path).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    data_root = _resolve_image_root(cfg, cfg_path)

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cls_map = _list_image_paths_by_class(data_root)
    print(f"[classify] data root: {data_root}")
    total_images = 0
    for cname in sorted(cls_map.keys()):
        n = len(cls_map[cname])
        total_images += n
        print(f"[classify] class='{cname}' images={n}")
    print(f"[classify] classes={len(cls_map)} total_images={total_images}")

    train_items, val_items, classes = _stratified_split(cls_map, val_ratio=val_ratio, seed=seed)
    if len(classes) < 2:
        raise RuntimeError(f"Need >=2 classes, got {len(classes)} from {data_root}")

    image_size = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tfm = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    val_tfm = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

    num_workers_eff = _resolve_num_workers(num_workers)
    loader_kwargs = {
        "num_workers": num_workers_eff,
        "pin_memory": bool(device.type == "cuda"),
        "persistent_workers": bool(num_workers_eff > 0),
    }
    if num_workers_eff > 0:
        loader_kwargs["prefetch_factor"] = 4

    dl_train = DataLoader(
        _ImageClsDataset(train_items, train_tfm),
        batch_size=max(1, batch_size),
        shuffle=True,
        **loader_kwargs,
    )
    dl_val = DataLoader(
        _ImageClsDataset(val_items, val_tfm),
        batch_size=max(1, batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(num_classes=len(classes), pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.05)

    class_counts = np.zeros((len(classes),), dtype=np.float64)
    for _, y in train_items:
        class_counts[int(y)] += 1.0
    inv = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
    inv = inv / max(1e-12, inv.mean())
    class_weights = torch.tensor(inv, dtype=torch.float32, device=device)
    print(f"[classify] class_weights={class_weights.detach().cpu().tolist()}")
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    best = {"epoch": 0, "val_loss": 1e9, "val_acc": 0.0, "val_macro_recall": 0.0, "val_mean_confidence": 0.0}
    history = []

    for ep in range(1, max(1, epochs) + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * int(x.size(0))
            n_seen += int(x.size(0))

        scheduler.step()
        v = _evaluate(model, dl_val, device=device, num_classes=len(classes))
        tr = train_loss / max(1, n_seen)
        history.append(
            {
                "epoch": ep,
                "train_loss": tr,
                "val_loss": v["loss"],
                "val_acc": v["acc"],
                "val_macro_recall": v["macro_recall"],
                "val_mean_confidence": v["mean_confidence"],
                "val_per_class_recall": v["per_class_recall"],
            }
        )
        print(
            f"[{ep:03d}/{epochs}] train_loss={tr:.4f} val_loss={v['loss']:.4f} "
            f"val_acc={v['acc']:.4f} val_recall={v['macro_recall']:.4f} val_conf={v['mean_confidence']:.4f}"
        )

        if v["macro_recall"] > best["val_macro_recall"] + 1e-5:
            best = {
                "epoch": ep,
                "val_loss": float(v["loss"]),
                "val_acc": float(v["acc"]),
                "val_macro_recall": float(v["macro_recall"]),
                "val_mean_confidence": float(v["mean_confidence"]),
                "val_per_class_recall": [float(x) for x in v["per_class_recall"]],
            }
            out_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": {
                        "arch": MODEL_ARCH,
                        "classes": classes,
                        "image_size": image_size,
                        "mean": mean,
                        "std": std,
                    },
                    "best": best,
                },
                out_ckpt,
            )

        if ep >= max(1, int(min_epochs)):
            reached = (
                float(v["acc"]) >= float(target_acc)
                and float(v["macro_recall"]) >= float(target_recall)
                and float(v["mean_confidence"]) >= float(target_confidence_min)
                and float(v["mean_confidence"]) <= float(target_confidence_max)
            )
            if reached:
                print(
                    f"Target reached at epoch {ep}: "
                    f"acc={v['acc']:.4f}, recall={v['macro_recall']:.4f}, conf={v['mean_confidence']:.4f}"
                )
                break

    report = {
        "model_arch": MODEL_ARCH,
        "default_ckpt": str(DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT),
        "trained_ckpt": str(out_ckpt),
        "data_root": str(data_root),
        "classes": classes,
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "best": best,
        "history": history,
        "infra": {
            "device": str(device),
            "num_workers": num_workers_eff,
            "pin_memory": bool(device.type == "cuda"),
            "persistent_workers": bool(num_workers_eff > 0),
            "prefetch_factor": 4 if num_workers_eff > 0 else None,
        },
    }
    write_report_json(out_report, report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser("Train fixed eval image classifier")
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=-1, help="-1 means auto")
    ap.add_argument("--min_epochs", type=int, default=10)
    ap.add_argument("--target_acc", type=float, default=0.95)
    ap.add_argument("--target_recall", type=float, default=0.95)
    ap.add_argument("--target_confidence_min", type=float, default=0.70)
    ap.add_argument("--target_confidence_max", type=float, default=0.98)
    ap.add_argument("--out_ckpt", type=str, default=str(DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT))
    ap.add_argument("--out_report", type=str, default=str(DEFAULT_REPORT_PATH))
    args = ap.parse_args()

    print(f"Model architecture: {MODEL_ARCH}")
    print(f"Default checkpoint path: {DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT}")
    report = train_from_config(
        config_path=Path(args.config),
        out_ckpt=Path(args.out_ckpt),
        out_report=Path(args.out_report),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        min_epochs=int(args.min_epochs),
        target_acc=float(args.target_acc),
        target_recall=float(args.target_recall),
        target_confidence_min=float(args.target_confidence_min),
        target_confidence_max=float(args.target_confidence_max),
    )
    print(f"Saved checkpoint: {report['trained_ckpt']}")
    print(f"Saved report: {args.out_report}")


if __name__ == "__main__":
    main()

