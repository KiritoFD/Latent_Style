from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.classify import load_eval_image_classifier


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _batched(items: list[Any], n: int) -> list[list[Any]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def _source_image_path(source_root: Path, src_style: str, src_image: str, src_stem: str | None = None) -> Path:
    if src_image:
        direct = source_root / src_style / Path(src_image).name
        if direct.exists():
            return direct
    if src_stem:
        for ext in IMAGE_EXTS:
            guess = source_root / src_style / f"{src_stem}{ext}"
            if guess.exists():
                return guess
    raise FileNotFoundError(f"missing source image for style={src_style} image={src_image or src_stem}")


def _infer_rows_from_images(images_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for image_path in sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name.lower()):
        stem = image_path.stem
        if "__to__" in stem:
            left, tgt_style = stem.rsplit("__to__", 1)
            if "__" not in left:
                continue
            src_style, src_stem = left.split("__", 1)
        elif "_to_" in stem:
            left, tgt_style = stem.rsplit("_to_", 1)
            if "_" not in left:
                continue
            src_style, suffix = left.split("_", 1)
            src_stem = f"{src_style}_{suffix}"
        else:
            continue
        rows.append(
            {
                "src_style": src_style,
                "tgt_style": tgt_style,
                "src_image": f"{src_stem}.jpg",
                "src_stem": src_stem,
                "gen_image": image_path.name,
            }
        )
    return rows


def _resolve_gen_path(images_dir: Path | None, row: dict[str, str], source_root: Path, generated_mode: str) -> Path:
    if generated_mode == "source_copy":
        return _source_image_path(source_root, row["src_style"], row.get("src_image", ""), row.get("src_stem"))
    if images_dir is None:
        raise FileNotFoundError("images_dir missing")
    gen_name = Path(str(row.get("gen_image") or row.get("image") or "")).name
    direct = images_dir / gen_name
    if direct.exists():
        return direct
    raw = images_dir / str(row.get("gen_image") or row.get("image") or "")
    if raw.exists():
        return raw
    raise FileNotFoundError(gen_name)


def evaluate_one(
    *,
    classifier,
    method: str,
    run: str,
    images_dir: Path | None,
    metrics_csv: Path | None,
    source_root: Path,
    generated_mode: str,
    batch_size: int,
) -> dict[str, Any]:
    rows = _read_rows(metrics_csv) if metrics_csv is not None else _infer_rows_from_images(images_dir or source_root)
    class_to_idx = {name: i for i, name in enumerate(classifier.classes)}
    preprocess = T.Compose(
        [
            T.Resize((classifier.image_size, classifier.image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
        ]
    )

    all_records: list[dict[str, Any]] = []
    for chunk in _batched(rows, max(1, batch_size)):
        images = []
        metas = []
        for row in chunk:
            path = _resolve_gen_path(images_dir, row, source_root, generated_mode)
            with Image.open(path) as img:
                images.append(preprocess(img.convert("RGB")))
            metas.append(row)
        batch = torch.stack(images, dim=0)
        x = classifier.preprocess(batch.to(classifier.device, dtype=torch.float32))
        with torch.no_grad():
            logits = classifier.model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu()
            preds = probs.argmax(dim=1)
        for row, prob, pred in zip(metas, probs, preds):
            src = row["src_style"]
            tgt = row["tgt_style"]
            src_idx = class_to_idx[src]
            tgt_idx = class_to_idx[tgt]
            all_records.append(
                {
                    "src_style": src,
                    "tgt_style": tgt,
                    "pred_style": classifier.classes[int(pred.item())],
                    "target_prob": float(prob[tgt_idx].item()),
                    "source_prob": float(prob[src_idx].item()),
                    "target_correct": float(int(pred.item() == tgt_idx)),
                    "source_correct": float(int(pred.item() == src_idx)),
                    "is_transfer": float(src != tgt),
                }
            )

    def mean(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    transfer = [r for r in all_records if r["is_transfer"] > 0.5]
    identity = [r for r in all_records if r["is_transfer"] < 0.5]
    return {
        "method": method,
        "run": run,
        "images": len(all_records),
        "transfer_target_acc": mean([r["target_correct"] for r in transfer]),
        "identity_source_acc": mean([r["source_correct"] for r in identity]),
        "transfer_target_prob": mean([r["target_prob"] for r in transfer]),
        "transfer_source_prob": mean([r["source_prob"] for r in transfer]),
        "transfer_target_source_margin": mean([r["target_prob"] - r["source_prob"] for r in transfer]),
        "all_pairs_target_source_margin": mean([r["target_prob"] - r["source_prob"] for r in all_records]),
        "images_dir": "" if images_dir is None else str(images_dir),
        "metrics_csv": "" if metrics_csv is None else str(metrics_csv),
        "source_root": str(source_root),
        "generated_mode": generated_mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate representative outputs with the existing Distinct5 style classifier.")
    parser.add_argument("--classifier-ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    classifier = load_eval_image_classifier(args.classifier_ckpt, device=args.device)
    manifest_rows = _read_rows(args.manifest)
    results: list[dict[str, Any]] = []
    for row in manifest_rows:
        method = str(row["method"]).strip()
        run = str(row["run"]).strip()
        images_dir_raw = str(row.get("images_dir", "")).strip()
        metrics_csv_raw = str(row.get("metrics_csv", "")).strip()
        source_root_raw = str(row.get("source_root", "")).strip()
        generated_mode = str(row.get("generated_mode", "")).strip() or "generated"
        source_root = Path(source_root_raw)
        images_dir = Path(images_dir_raw) if images_dir_raw else None
        metrics_csv = Path(metrics_csv_raw) if metrics_csv_raw else None
        results.append(
            evaluate_one(
                classifier=classifier,
                method=method,
                run=run,
                images_dir=images_dir,
                metrics_csv=metrics_csv,
                source_root=source_root,
                generated_mode=generated_mode,
                batch_size=int(args.batch_size),
            )
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "run",
                "images",
                "transfer_target_acc",
                "identity_source_acc",
                "transfer_target_prob",
                "transfer_source_prob",
                "transfer_target_source_margin",
                "all_pairs_target_source_margin",
                "images_dir",
                "metrics_csv",
                "source_root",
                "generated_mode",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output_csv)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
