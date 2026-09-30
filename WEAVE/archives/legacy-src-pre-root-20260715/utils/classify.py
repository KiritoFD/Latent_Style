from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode

MODEL_ARCH = "styleeval_convnext_tiny_v1"
DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT = Path("../../eval_cache/eval_style_image_classifier.pt")
DEFAULT_REPORT_PATH = Path("../../eval_cache/eval_style_image_classifier_report.json")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        imgs = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if imgs:
            out[d.name] = imgs
    return out


def _candidate_paths(raw: str, config_path: Path) -> list[Path]:
    p = Path(raw).expanduser()
    cfg_dir = config_path.resolve().parent
    bases = [cfg_dir, cfg_dir.parent, cfg_dir.parent.parent, Path.cwd(), Path(__file__).resolve().parents[2]]
    if p.is_absolute():
        return [p]
    out: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        cand = (base / p).resolve()
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _resolve_image_root_from_raw(raw: str, config_path: Path, *, label: str) -> Path:
    raw = str(raw or "").strip()
    if not raw:
        raise FileNotFoundError(f"{label} is empty")
    print(f"[classify] resolving {label}: {raw}")
    for cand in _candidate_paths(raw, config_path):
        exists = cand.exists()
        is_dir = cand.is_dir() if exists else False
        cls_count = len(_list_image_paths_by_class(cand)) if (exists and is_dir) else 0
        print(f"  - {cand} | exists={exists} is_dir={is_dir} class_folders_with_images={cls_count}")
        if exists and is_dir and cls_count > 0:
            return cand
    raise FileNotFoundError(f"No valid image root for {label}={raw}")


def _resolve_train_val_roots(
    cfg: dict[str, Any],
    config_path: Path,
    *,
    train_root: str = "",
    val_root: str = "",
) -> tuple[Path, Path | None]:
    train_raw = train_root.strip() or str(cfg.get("data", {}).get("image_root", "")).strip()
    train_dir = _resolve_image_root_from_raw(train_raw, config_path, label="train_root")

    if val_root.strip():
        return train_dir, _resolve_image_root_from_raw(val_root.strip(), config_path, label="val_root")

    sibling_candidates = [
        train_dir.parent / "val_clean50",
        train_dir.parent / "val",
    ]
    for cand in sibling_candidates:
        cls_map = _list_image_paths_by_class(cand)
        if cls_map:
            print(f"[classify] using fixed validation root: {cand}")
            return train_dir, cand

    print("[classify] fixed validation root not found, fallback to stratified split from train_root")
    return train_dir, None


def build_model(num_classes: int, pretrained: bool = False, dropout: float = 0.25) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = models.convnext_tiny(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        print(f"[classify] warning: failed to load pretrained ConvNeXt weights, fallback to random init: {exc}")
        model = models.convnext_tiny(weights=None)

    in_features = int(model.classifier[2].in_features)
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(float(dropout)),
        nn.Linear(in_features, int(num_classes)),
    )
    return model


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
        self.preprocess = T.Compose(
            [
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.Normalize(mean=list(mean), std=list(std)),
            ]
        )

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
    image_size = int(meta.get("image_size", 224))
    mean = meta.get("mean", IMAGENET_MEAN)
    std = meta.get("std", IMAGENET_STD)
    arch = str(meta.get("arch", MODEL_ARCH))
    if arch != MODEL_ARCH:
        raise ValueError(f"Unsupported arch in checkpoint: {arch}")
    dropout = float(meta.get("dropout", 0.25))
    model = build_model(num_classes=len(classes), pretrained=False, dropout=dropout)
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
        with Image.open(p) as img:
            img = img.convert("RGB")
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


def _items_from_separate_roots(
    train_cls_map: dict[str, list[Path]],
    val_cls_map: dict[str, list[Path]],
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[str]]:
    train_classes = sorted(train_cls_map.keys())
    val_classes = sorted(val_cls_map.keys())
    if train_classes != val_classes:
        raise RuntimeError(
            "Train/val class folders differ: "
            f"train={train_classes}, val={val_classes}"
        )
    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    for cid, cls_name in enumerate(train_classes):
        for p in train_cls_map[cls_name]:
            train_items.append((p, cid))
        for p in val_cls_map[cls_name]:
            val_items.append((p, cid))
    return train_items, val_items, train_classes


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
        "confusion_matrix": conf.tolist(),
    }


def _better_metrics(cur: dict[str, Any], best: dict[str, Any]) -> bool:
    eps = 1e-6
    if float(cur["val_macro_recall"]) > float(best["val_macro_recall"]) + eps:
        return True
    if float(cur["val_macro_recall"]) + eps < float(best["val_macro_recall"]):
        return False
    if float(cur["val_acc"]) > float(best["val_acc"]) + eps:
        return True
    if float(cur["val_acc"]) + eps < float(best["val_acc"]):
        return False
    if float(cur["val_loss"]) < float(best["val_loss"]) - eps:
        return True
    if float(cur["val_loss"]) - eps > float(best["val_loss"]):
        return False
    return float(cur["val_mean_confidence"]) > float(best["val_mean_confidence"]) + eps


def train_from_config(
    config_path: Path,
    out_ckpt: Path = DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT,
    out_report: Path = DEFAULT_REPORT_PATH,
    *,
    train_root: str = "",
    val_root: str = "",
    epochs: int = 40,
    batch_size: int = 48,
    lr: float = 3e-5,
    head_lr_multiplier: float = 5.0,
    weight_decay: float = 2e-4,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = -1,
    min_epochs: int = 8,
    patience: int = 12,
    target_acc: float = 0.97,
    target_recall: float = 0.97,
    target_confidence_min: float = 0.75,
    target_confidence_max: float = 0.995,
    pretrained: bool = True,
    image_size: int = 224,
    dropout: float = 0.25,
) -> dict[str, Any]:
    cfg_path = Path(config_path).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    train_dir, val_dir = _resolve_train_val_roots(cfg, cfg_path, train_root=train_root, val_root=val_root)

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_cls_map = _list_image_paths_by_class(train_dir)
    print(f"[classify] train root: {train_dir}")
    for cname in sorted(train_cls_map.keys()):
        print(f"[classify] train class='{cname}' images={len(train_cls_map[cname])}")

    if val_dir is not None:
        val_cls_map = _list_image_paths_by_class(val_dir)
        print(f"[classify] val root: {val_dir}")
        for cname in sorted(val_cls_map.keys()):
            print(f"[classify] val class='{cname}' images={len(val_cls_map[cname])}")
        train_items, val_items, classes = _items_from_separate_roots(train_cls_map, val_cls_map)
        split_mode = "fixed_val_root"
    else:
        train_items, val_items, classes = _stratified_split(train_cls_map, val_ratio=val_ratio, seed=seed)
        split_mode = "stratified_split"

    if len(classes) < 2:
        raise RuntimeError(f"Need >=2 classes, got {len(classes)} from {train_dir}")

    train_tfm = T.Compose(
        [
            T.RandomResizedCrop(
                int(image_size),
                scale=(0.70, 1.0),
                ratio=(0.90, 1.10),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
        ]
    )
    val_tfm = T.Compose(
        [
            T.Resize(int(image_size) + 32, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(int(image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    num_workers_eff = _resolve_num_workers(num_workers)
    loader_kwargs = {
        "num_workers": num_workers_eff,
        "pin_memory": bool(device.type == "cuda"),
        "persistent_workers": bool(num_workers_eff > 0),
    }
    if num_workers_eff > 0:
        loader_kwargs["prefetch_factor"] = 4

    class_counts = np.zeros((len(classes),), dtype=np.float64)
    for _, y in train_items:
        class_counts[int(y)] += 1.0
    inv = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
    inv = inv / max(1e-12, inv.mean())
    class_weights = torch.tensor(inv, dtype=torch.float32, device=device)
    sample_weights = [float(inv[int(y)]) for _, y in train_items]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_items), replacement=True)
    print(f"[classify] class_weights={class_weights.detach().cpu().tolist()}")

    dl_train = DataLoader(
        _ImageClsDataset(train_items, train_tfm),
        batch_size=max(1, batch_size),
        sampler=sampler,
        shuffle=False,
        **loader_kwargs,
    )
    dl_val = DataLoader(
        _ImageClsDataset(val_items, val_tfm),
        batch_size=max(1, batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(num_classes=len(classes), pretrained=pretrained, dropout=dropout).to(device)

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": float(lr)},
            {"params": head_params, "lr": float(lr) * float(head_lr_multiplier)},
        ],
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(lr) * 0.1)
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    amp_enabled = device.type == "cuda"
    amp_ctx = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        if amp_enabled
        else nullcontext()
    )

    best = {
        "epoch": 0,
        "val_loss": 1e9,
        "val_acc": 0.0,
        "val_macro_recall": 0.0,
        "val_mean_confidence": 0.0,
        "val_per_class_recall": [],
        "confusion_matrix": [],
    }
    history = []
    epochs_since_improve = 0

    for ep in range(1, max(1, epochs) + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        correct = 0

        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(x)
                loss = ce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += float(loss.item()) * int(x.size(0))
            n_seen += int(x.size(0))
            correct += int((logits.detach().argmax(dim=1) == y).sum().item())

        scheduler.step()
        v = _evaluate(model, dl_val, device=device, num_classes=len(classes))
        tr = train_loss / max(1, n_seen)
        train_acc = correct / max(1, n_seen)
        current = {
            "epoch": ep,
            "train_loss": tr,
            "train_acc": train_acc,
            "val_loss": float(v["loss"]),
            "val_acc": float(v["acc"]),
            "val_macro_recall": float(v["macro_recall"]),
            "val_mean_confidence": float(v["mean_confidence"]),
            "val_per_class_recall": [float(x) for x in v["per_class_recall"]],
            "confusion_matrix": v["confusion_matrix"],
            "lr_backbone": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]),
        }
        history.append(current)
        print(
            f"[{ep:03d}/{epochs}] train_loss={tr:.4f} train_acc={train_acc:.4f} "
            f"val_loss={v['loss']:.4f} val_acc={v['acc']:.4f} "
            f"val_recall={v['macro_recall']:.4f} val_conf={v['mean_confidence']:.4f}"
        )

        if _better_metrics(current, best):
            best = {
                "epoch": ep,
                "val_loss": float(v["loss"]),
                "val_acc": float(v["acc"]),
                "val_macro_recall": float(v["macro_recall"]),
                "val_mean_confidence": float(v["mean_confidence"]),
                "val_per_class_recall": [float(x) for x in v["per_class_recall"]],
                "confusion_matrix": v["confusion_matrix"],
            }
            epochs_since_improve = 0
            out_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": {
                        "arch": MODEL_ARCH,
                        "classes": classes,
                        "image_size": int(image_size),
                        "mean": IMAGENET_MEAN,
                        "std": IMAGENET_STD,
                        "dropout": float(dropout),
                    },
                    "best": best,
                },
                out_ckpt,
            )
        else:
            epochs_since_improve += 1

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
            if int(patience) > 0 and epochs_since_improve >= int(patience):
                print(f"Early stop at epoch {ep}: no validation improvement for {epochs_since_improve} epochs")
                break

    report = {
        "model_arch": MODEL_ARCH,
        "default_ckpt": str(DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT),
        "trained_ckpt": str(out_ckpt),
        "train_root": str(train_dir),
        "val_root": str(val_dir) if val_dir is not None else "",
        "split_mode": split_mode,
        "classes": classes,
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "best": best,
        "history": history,
        "train_class_counts": {cls_name: len(train_cls_map[cls_name]) for cls_name in classes},
        "val_class_counts": (
            {cls_name: len(_list_image_paths_by_class(val_dir)[cls_name]) for cls_name in classes}
            if val_dir is not None
            else {}
        ),
        "infra": {
            "device": str(device),
            "num_workers": num_workers_eff,
            "pin_memory": bool(device.type == "cuda"),
            "persistent_workers": bool(num_workers_eff > 0),
            "prefetch_factor": 4 if num_workers_eff > 0 else None,
            "amp": amp_enabled,
            "pretrained": bool(pretrained),
            "image_size": int(image_size),
            "batch_size": int(batch_size),
        },
    }
    write_report_json(out_report, report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser("Train strong eval image classifier")
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--train_root", type=str, default="")
    ap.add_argument("--val_root", type=str, default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_lr_multiplier", type=float, default=5.0)
    ap.add_argument("--weight_decay", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=-1, help="-1 means auto")
    ap.add_argument("--min_epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--target_acc", type=float, default=0.97)
    ap.add_argument("--target_recall", type=float, default=0.97)
    ap.add_argument("--target_confidence_min", type=float, default=0.75)
    ap.add_argument("--target_confidence_max", type=float, default=0.995)
    ap.add_argument("--pretrained", action="store_true", help="Use cached torchvision pretrained ConvNeXt-Tiny weights")
    ap.add_argument("--no_pretrained", action="store_true", help="Disable pretrained weights even if available")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--out_ckpt", type=str, default=str(DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT))
    ap.add_argument("--out_report", type=str, default=str(DEFAULT_REPORT_PATH))
    args = ap.parse_args()

    use_pretrained = True
    if args.no_pretrained:
        use_pretrained = False
    elif args.pretrained:
        use_pretrained = True

    print(f"Model architecture: {MODEL_ARCH}")
    print(f"Default checkpoint path: {DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT}")
    report = train_from_config(
        config_path=Path(args.config),
        out_ckpt=Path(args.out_ckpt),
        out_report=Path(args.out_report),
        train_root=str(args.train_root),
        val_root=str(args.val_root),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        head_lr_multiplier=float(args.head_lr_multiplier),
        weight_decay=float(args.weight_decay),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        min_epochs=int(args.min_epochs),
        patience=int(args.patience),
        target_acc=float(args.target_acc),
        target_recall=float(args.target_recall),
        target_confidence_min=float(args.target_confidence_min),
        target_confidence_max=float(args.target_confidence_max),
        pretrained=bool(use_pretrained),
        image_size=int(args.image_size),
        dropout=float(args.dropout),
    )
    print(f"Saved checkpoint: {report['trained_ckpt']}")
    print(f"Saved report: {args.out_report}")


if __name__ == "__main__":
    main()
