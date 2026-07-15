from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.modern_metrics import ClipEmbedder


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _batched[T](items: list[T], n: int) -> list[list[T]]:
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
            rows.append(
                {
                    "src_style": src_style,
                    "tgt_style": tgt_style,
                    "src_image": f"{src_stem}.jpg",
                    "src_stem": src_stem,
                    "gen_image": image_path.name,
                }
            )
            continue
        marker = "_to_"
        if marker not in stem:
            continue
        left, tgt_style = stem.rsplit(marker, 1)
        if "_" not in left:
            continue
        src_style, suffix = left.split("_", 1)
        src_stem = f"{src_style}_{suffix}"
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


def _prompt_for_style(style: str, template: str) -> str:
    return template.format(style=style.replace("_", " "))


def evaluate_one(
    *,
    clip: ClipEmbedder,
    method: str,
    run: str,
    images_dir: Path | None,
    metrics_csv: Path | None,
    source_root: Path,
    generated_mode: str,
    prompt_template: str,
    batch_size: int,
) -> dict[str, Any]:
    rows = _read_rows(metrics_csv) if metrics_csv is not None else _infer_rows_from_images(images_dir or source_root)
    styles = sorted({str(row["src_style"]) for row in rows} | {str(row["tgt_style"]) for row in rows})
    prompts = [_prompt_for_style(style, prompt_template) for style in styles]
    text_bank = clip.encode_texts(prompts, batch_size=batch_size)
    style_to_idx = {style: idx for idx, style in enumerate(styles)}

    all_records: list[dict[str, Any]] = []
    for chunk in _batched(rows, max(1, batch_size)):
        gen_paths = [
            _resolve_gen_path(images_dir, row, source_root, generated_mode)
            for row in chunk
        ]
        img_feats = clip.encode_paths(gen_paths, batch_size=batch_size)
        sims = torch.matmul(img_feats, text_bank.T)
        preds = sims.argmax(dim=1)
        for row, sim_row, pred_idx in zip(chunk, sims, preds):
            src = str(row["src_style"])
            tgt = str(row["tgt_style"])
            src_idx = style_to_idx[src]
            tgt_idx = style_to_idx[tgt]
            all_records.append(
                {
                    "src_style": src,
                    "tgt_style": tgt,
                    "pred_style": styles[int(pred_idx.item())],
                    "target_score": float(sim_row[tgt_idx].item()),
                    "source_score": float(sim_row[src_idx].item()),
                    "target_correct": float(int(pred_idx.item() == tgt_idx)),
                    "source_correct": float(int(pred_idx.item() == src_idx)),
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
        "transfer_target_text": mean([r["target_score"] for r in transfer]),
        "transfer_source_text": mean([r["source_score"] for r in transfer]),
        "transfer_target_source_margin": mean([r["target_score"] - r["source_score"] for r in transfer]),
        "all_pairs_target_source_margin": mean([r["target_score"] - r["source_score"] for r in all_records]),
        "prompt_template": prompt_template,
        "styles": styles,
        "images_dir": "" if images_dir is None else str(images_dir),
        "metrics_csv": "" if metrics_csv is None else str(metrics_csv),
        "source_root": str(source_root),
        "generated_mode": generated_mode,
    }


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# CLIP-T Style Probe",
        "",
        "| Method | Run | Transfer acc | Target text | Source text | Margin | Identity acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['run']} | {row['transfer_target_acc']:.4f} | "
            f"{row['transfer_target_text']:.4f} | {row['transfer_source_text']:.4f} | "
            f"{row['transfer_target_source_margin']:.4f} | {row['identity_source_acc']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CLIP image-to-style-text alignment on representative style-transfer outputs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt-template", default="a painting in {style} style")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    clip = ClipEmbedder(args.clip_model, args.device)
    manifest_rows = _read_rows(args.manifest)
    results: list[dict[str, Any]] = []
    for row in manifest_rows:
        method = str(row["method"]).strip()
        run = str(row["run"]).strip()
        images_dir_raw = str(row.get("images_dir", "")).strip()
        metrics_csv_raw = str(row.get("metrics_csv", "")).strip()
        source_root_raw = str(row.get("source_root", "")).strip()
        generated_mode = str(row.get("generated_mode", "")).strip() or "generated"
        if not source_root_raw:
            raise ValueError(f"Missing source_root for {method}/{run}")
        source_root = Path(source_root_raw)
        images_dir = Path(images_dir_raw) if images_dir_raw else None
        metrics_csv = Path(metrics_csv_raw) if metrics_csv_raw else None
        results.append(
            evaluate_one(
                clip=clip,
                method=method,
                run=run,
                images_dir=images_dir,
                metrics_csv=metrics_csv,
                source_root=source_root,
                generated_mode=generated_mode,
                prompt_template=str(args.prompt_template),
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
                "transfer_target_text",
                "transfer_source_text",
                "transfer_target_source_margin",
                "all_pairs_target_source_margin",
                "prompt_template",
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
    write_markdown(results, args.output_csv.with_suffix(".md"))
    print(args.output_csv)
    print(args.output_json)
    print(args.output_csv.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
