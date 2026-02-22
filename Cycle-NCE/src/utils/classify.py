from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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


def _parse_src_tgt_styles_from_name(stem: str, classes: list[str]) -> tuple[str, str] | None:
    # Preferred form from run_evaluation: "{src}_{name}_to_{target}"
    classes_norm = {c.lower(): c for c in classes}
    m = re.search(r"_to_([^_]+)$", stem)
    if m:
        cand_tgt = m.group(1).strip().lower()
        if cand_tgt in classes_norm:
            tgt = classes_norm[cand_tgt]
            left = stem[: m.start()]
            # Parse source style from prefix before first "_" in left part.
            src_raw = left.split("_", 1)[0].strip().lower()
            if src_raw in classes_norm:
                return classes_norm[src_raw], tgt
    # Fallback: match suffix by known style names, then infer source by prefix.
    stem_l = stem.lower()
    for c in sorted(classes, key=len, reverse=True):
        key = f"_to_{c.lower()}"
        if stem_l.endswith(key):
            left = stem_l[: -len(key)]
            src_raw = left.split("_", 1)[0].strip().lower()
            if src_raw in classes_norm:
                return classes_norm[src_raw], c
    return None


@torch.no_grad()
def evaluate_generated_dir(
    generated_dir: Path,
    ckpt_path: Path,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 128,
) -> dict[str, Any]:
    clf = load_eval_image_classifier(ckpt_path=ckpt_path, device=device)
    files = sorted([p for p in generated_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not files:
        raise FileNotFoundError(f"No images found in {generated_dir}")

    class_to_idx = {c: i for i, c in enumerate(clf.classes)}
    y_true: list[int] = []
    src_true: list[int] = []
    kept_files: list[Path] = []
    skipped = 0
    for p in files:
        parsed = _parse_src_tgt_styles_from_name(p.stem, clf.classes)
        if parsed is None:
            skipped += 1
            continue
        src, tgt = parsed
        src_true.append(class_to_idx[src])
        y_true.append(class_to_idx[tgt])
        kept_files.append(p)

    if not kept_files:
        raise RuntimeError(f"No parsable files with '_to_*' target style found in {generated_dir}")

    tfm = T.Compose(
        [
            T.Resize((clf.image_size, clf.image_size)),
            T.ToTensor(),
        ]
    )

    y_pred: list[int] = []
    for s in range(0, len(kept_files), max(1, int(batch_size))):
        e = min(s + max(1, int(batch_size)), len(kept_files))
        xs = []
        for p in kept_files[s:e]:
            img = Image.open(p).convert("RGB")
            xs.append(tfm(img))
        xb = torch.stack(xs, dim=0)
        preds = clf.predict_indices(xb).detach().cpu().tolist()
        y_pred.extend([int(v) for v in preds])

    total = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    acc = float(correct / max(1, total))

    per_class = {}
    for idx, name in enumerate(clf.classes):
        idxs = [i for i, t in enumerate(y_true) if t == idx]
        c = sum(int(y_pred[i] == idx) for i in idxs) if idxs else 0
        per_class[name] = {
            "n": len(idxs),
            "acc": float(c / len(idxs)) if idxs else 0.0,
        }

    # Art-only metrics: exclude "photo" to expose 4-style transfer quality directly.
    art_class_names = [c for c in clf.classes if c.lower() != "photo"]
    art_idxs = [class_to_idx[c] for c in art_class_names]
    art_mask = [i for i, t in enumerate(y_true) if t in art_idxs]
    art_total = len(art_mask)
    art_correct = sum(int(y_pred[i] == y_true[i]) for i in art_mask)
    art_acc = float(art_correct / max(1, art_total))
    per_class_art = {}
    for name in art_class_names:
        idx = class_to_idx[name]
        idxs = [i for i, t in enumerate(y_true) if t == idx]
        c = sum(int(y_pred[i] == idx) for i in idxs) if idxs else 0
        per_class_art[name] = {
            "n": len(idxs),
            "acc": float(c / len(idxs)) if idxs else 0.0,
        }

    report = {
        "generated_dir": str(generated_dir),
        "ckpt_path": str(ckpt_path),
        "total_images": len(files),
        "parsed_images": total,
        "skipped_unparsable": skipped,
        "accuracy": acc,
        "per_class": per_class,
        "art_only_accuracy": art_acc,
        "art_only_total": art_total,
        "per_class_art_only": per_class_art,
    }

    # 5x5 source->target breakdown (or NxN for generic class count).
    pair_stats = {}
    for src_name in clf.classes:
        for tgt_name in clf.classes:
            s_idx = class_to_idx[src_name]
            t_idx = class_to_idx[tgt_name]
            idxs = [i for i in range(total) if src_true[i] == s_idx and y_true[i] == t_idx]
            c = sum(int(y_pred[i] == t_idx) for i in idxs) if idxs else 0
            pair_stats[f"{src_name}->{tgt_name}"] = {
                "n": len(idxs),
                "acc": float(c / len(idxs)) if idxs else 0.0,
            }
    report["pair_accuracy"] = pair_stats

    print(
        f"[classify][generated-test] total={len(files)} parsed={total} skipped={skipped} "
        f"acc={acc:.4f} ({correct}/{total})"
    )
    for k in sorted(per_class.keys()):
        v = per_class[k]
        print(f"[classify][generated-test] class={k:>10s} n={v['n']:4d} acc={v['acc']:.4f}")
    if art_class_names:
        print(
            f"[classify][generated-test][art-only] total={art_total} "
            f"acc={art_acc:.4f} ({art_correct}/{art_total})"
        )
        for k in sorted(per_class_art.keys()):
            v = per_class_art[k]
            print(f"[classify][generated-test][art-only] class={k:>10s} n={v['n']:4d} acc={v['acc']:.4f}")
    print("[classify][generated-test][pair-acc] source->target (25 combos):")
    for src_name in clf.classes:
        for tgt_name in clf.classes:
            k = f"{src_name}->{tgt_name}"
            v = pair_stats[k]
            print(f"[classify][generated-test][pair-acc] {k:>20s} n={v['n']:4d} acc={v['acc']:.4f}")
    return report


def write_report_json(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_existing_best(ckpt_path: Path) -> dict[str, Any] | None:
    if not ckpt_path.exists():
        return None
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    best = payload.get("best")
    if not isinstance(best, dict):
        return None
    return {
        "epoch": int(best.get("epoch", 0)),
        "val_loss": float(best.get("val_loss", 1e9)),
        "val_acc": float(best.get("val_acc", 0.0)),
        "val_macro_recall": float(best.get("val_macro_recall", 0.0)),
        "val_mean_confidence": float(best.get("val_mean_confidence", 0.0)),
        "val_per_class_recall": [float(x) for x in best.get("val_per_class_recall", [])],
    }


def _is_better_best(candidate: dict[str, Any], baseline: dict[str, Any], tol: float = 1e-5) -> bool:
    # Priority: macro_recall -> acc -> lower loss.
    if float(candidate["val_macro_recall"]) > float(baseline["val_macro_recall"]) + tol:
        return True
    if float(candidate["val_macro_recall"]) < float(baseline["val_macro_recall"]) - tol:
        return False
    if float(candidate["val_acc"]) > float(baseline["val_acc"]) + tol:
        return True
    if float(candidate["val_acc"]) < float(baseline["val_acc"]) - tol:
        return False
    return float(candidate["val_loss"]) < float(baseline["val_loss"]) - tol


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
    weight_decay: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = -1,
    min_epochs: int = 10,
    target_acc: float = 0.95,
    target_recall: float = 0.95,
    target_confidence_min: float = 0.70,
    target_confidence_max: float = 0.98,
    class_weight_power: float = 1.0,
    label_smoothing: float = 0.0,
    use_balanced_sampler: bool = False,
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

    train_ds = _ImageClsDataset(train_items, train_tfm)
    train_sampler = None
    if use_balanced_sampler:
        class_counts_for_sampler = np.zeros((len(classes),), dtype=np.float64)
        for _, y in train_items:
            class_counts_for_sampler[int(y)] += 1.0
        sample_weights = [1.0 / max(1.0, class_counts_for_sampler[int(y)]) for _, y in train_items]
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_items),
            replacement=True,
        )

    dl_train = DataLoader(
        train_ds,
        batch_size=max(1, batch_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
    power = float(class_weight_power)
    inv = 1.0 / np.power(np.maximum(class_counts, 1.0), power)
    inv = inv / max(1e-12, inv.mean())
    class_weights = torch.tensor(inv, dtype=torch.float32, device=device)
    print(f"[classify] class_weight_power={power:.2f} class_weights={class_weights.detach().cpu().tolist()}")
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing))

    best = {"epoch": 0, "val_loss": 1e9, "val_acc": 0.0, "val_macro_recall": 0.0, "val_mean_confidence": 0.0}
    existing_best = _load_existing_best(out_ckpt)
    if existing_best is not None:
        best = existing_best
        print(
            "[classify] loaded historical best from ckpt: "
            f"epoch={best['epoch']} recall={best['val_macro_recall']:.4f} "
            f"acc={best['val_acc']:.4f} loss={best['val_loss']:.4f}"
        )
    history = []
    ckpt_updated = False

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

        candidate_best = {
            "epoch": ep,
            "val_loss": float(v["loss"]),
            "val_acc": float(v["acc"]),
            "val_macro_recall": float(v["macro_recall"]),
            "val_mean_confidence": float(v["mean_confidence"]),
            "val_per_class_recall": [float(x) for x in v["per_class_recall"]],
        }
        if _is_better_best(candidate_best, best):
            best = candidate_best
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
            ckpt_updated = True

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
        "ckpt_updated": ckpt_updated,
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
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=-1, help="-1 means auto")
    ap.add_argument("--min_epochs", type=int, default=10)
    ap.add_argument("--target_acc", type=float, default=0.95)
    ap.add_argument("--target_recall", type=float, default=0.95)
    ap.add_argument("--target_confidence_min", type=float, default=0.70)
    ap.add_argument("--target_confidence_max", type=float, default=0.98)
    ap.add_argument("--class_weight_power", type=float, default=1.0, help="Use 1/(count^power) for class weighting")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--use_balanced_sampler", action="store_true", default=False)
    ap.add_argument("--no_balanced_sampler", action="store_false", dest="use_balanced_sampler")
    ap.add_argument("--out_ckpt", type=str, default=str(DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT))
    ap.add_argument("--out_report", type=str, default=str(DEFAULT_REPORT_PATH))
    ap.add_argument("--test_generated_dir", type=str, default="", help="Directory containing run_evaluation generated images")
    ap.add_argument("--test_ckpt", type=str, default="", help="Classifier checkpoint for --test_generated_dir; default uses --out_ckpt")
    ap.add_argument("--test_only", action="store_true", help="Only run generated-dir classification test")
    ap.add_argument(
        "--train_then_test",
        action="store_true",
        help="When --test_generated_dir is set, train first then test (default is test-only).",
    )
    args = ap.parse_args()

    print(f"Model architecture: {MODEL_ARCH}")
    print(f"Default checkpoint path: {DEFAULT_EVAL_IMAGE_CLASSIFIER_CKPT}")
    has_test_dir = bool(str(args.test_generated_dir).strip())
    should_test_only = bool(args.test_only or (has_test_dir and not args.train_then_test))
    if should_test_only:
        if not has_test_dir:
            raise ValueError("Test-only mode requires --test_generated_dir")
        ckpt = Path(args.test_ckpt) if str(args.test_ckpt).strip() else Path(args.out_ckpt)
        rep = evaluate_generated_dir(
            generated_dir=Path(args.test_generated_dir),
            ckpt_path=ckpt,
            batch_size=max(1, int(args.batch_size)),
        )
        print(f"[classify][generated-test] accuracy={rep['accuracy']:.4f}")
        return

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
        class_weight_power=float(args.class_weight_power),
        label_smoothing=float(args.label_smoothing),
        use_balanced_sampler=bool(args.use_balanced_sampler),
    )
    if bool(report.get("ckpt_updated", False)):
        print(f"Saved checkpoint (updated): {report['trained_ckpt']}")
    else:
        print(f"Checkpoint kept (no improvement): {report['trained_ckpt']}")
    print(f"Saved report: {args.out_report}")
    if str(args.test_generated_dir).strip():
        ckpt = Path(args.test_ckpt) if str(args.test_ckpt).strip() else Path(report["trained_ckpt"])
        rep = evaluate_generated_dir(
            generated_dir=Path(args.test_generated_dir),
            ckpt_path=ckpt,
            batch_size=max(1, int(args.batch_size)),
        )
        print(f"[classify][generated-test] accuracy={rep['accuracy']:.4f}")


if __name__ == "__main__":
    main()
