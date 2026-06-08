from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image

from utils.classify import load_eval_image_classifier


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_rows(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_gen_path(images_dir: Path, gen_image: str) -> Path:
    name = Path(str(gen_image)).name
    direct = images_dir / name
    if direct.exists():
        return direct
    raw = images_dir / str(gen_image)
    if raw.exists():
        return raw
    raise FileNotFoundError(name)


def batched(iterable: list[Any], n: int) -> list[list[Any]]:
    return [iterable[i : i + n] for i in range(0, len(iterable), n)]


def evaluate_one(
    *,
    classifier,
    method: str,
    run: str,
    images_dir: Path,
    metrics_csv: Path,
    batch_size: int,
) -> dict[str, Any]:
    rows = load_rows(metrics_csv)
    class_to_idx = {name: i for i, name in enumerate(classifier.classes)}
    preprocess = T.Compose(
        [
            T.Resize((classifier.image_size, classifier.image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
        ]
    )

    all_records: list[dict[str, Any]] = []
    for chunk in batched(rows, max(1, batch_size)):
        images = []
        metas = []
        for row in chunk:
            path = resolve_gen_path(images_dir, row["gen_image"])
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
    confusion = Counter((r["tgt_style"], r["pred_style"]) for r in transfer)
    confusion_rows = [
        {"target_style": tgt, "pred_style": pred, "count": count}
        for (tgt, pred), count in sorted(confusion.items())
    ]

    by_target = defaultdict(list)
    for r in transfer:
        by_target[r["tgt_style"]].append(r)

    per_target = []
    for tgt, bucket in sorted(by_target.items()):
        per_target.append(
            {
                "target_style": tgt,
                "target_acc": mean([b["target_correct"] for b in bucket]),
                "target_prob": mean([b["target_prob"] for b in bucket]),
                "source_prob": mean([b["source_prob"] for b in bucket]),
                "target_source_margin": mean([b["target_prob"] - b["source_prob"] for b in bucket]),
                "count": len(bucket),
            }
        )

    return {
        "method": method,
        "run": run,
        "images": len(all_records),
        "all_pairs_target_acc": mean([r["target_correct"] for r in all_records]),
        "transfer_target_acc": mean([r["target_correct"] for r in transfer]),
        "identity_source_acc": mean([r["source_correct"] for r in identity]),
        "transfer_target_prob": mean([r["target_prob"] for r in transfer]),
        "transfer_source_prob": mean([r["source_prob"] for r in transfer]),
        "transfer_target_source_margin": mean([r["target_prob"] - r["source_prob"] for r in transfer]),
        "all_pairs_target_source_margin": mean([r["target_prob"] - r["source_prob"] for r in all_records]),
        "per_target": per_target,
        "transfer_confusion": confusion_rows,
        "images_dir": str(images_dir),
        "metrics_csv": str(metrics_csv),
    }


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Non-CLIP Style Verification",
        "",
        "| Method | Run | Transfer target acc | Transfer target prob | Transfer source prob | Transfer margin | Identity source acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['run']} | {row['transfer_target_acc']:.4f} | "
            f"{row['transfer_target_prob']:.4f} | {row['transfer_source_prob']:.4f} | "
            f"{row['transfer_target_source_margin']:.4f} | {row['identity_source_acc']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    classifier = load_eval_image_classifier(args.classifier_ckpt, device=args.device)
    manifest_rows = read_manifest(args.manifest)
    results: list[dict[str, Any]] = []
    for row in manifest_rows:
        method = str(row["method"]).strip()
        run = str(row["run"]).strip()
        images_dir = Path(str(row["images_dir"]).strip())
        metrics_csv = Path(str(row["metrics_csv"]).strip()) if str(row.get("metrics_csv", "")).strip() else images_dir.parent / "metrics.csv"
        if not images_dir.exists() or not metrics_csv.exists():
            print(f"SKIP {method}/{run}: missing images or metrics")
            continue
        print(f"Evaluating {method}/{run}")
        results.append(
            evaluate_one(
                classifier=classifier,
                method=method,
                run=run,
                images_dir=images_dir,
                metrics_csv=metrics_csv,
                batch_size=args.batch_size,
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
                "all_pairs_target_acc",
                "transfer_target_acc",
                "identity_source_acc",
                "transfer_target_prob",
                "transfer_source_prob",
                "transfer_target_source_margin",
                "all_pairs_target_source_margin",
                "images_dir",
                "metrics_csv",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(results, args.output_csv.with_suffix(".md"))
    print(args.output_csv)
    print(args.output_json)
    print(args.output_csv.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
