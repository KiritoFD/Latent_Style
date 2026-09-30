from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _load_images(paths: list[Path]) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


@torch.inference_mode()
def _ssm_distance(
    model: AutoModel,
    processor: AutoImageProcessor,
    src_paths: list[Path],
    gen_paths: list[Path],
    *,
    device: torch.device,
    batch_size: int,
) -> float | None:
    vals: list[float] = []
    for start in range(0, len(src_paths), max(1, int(batch_size))):
        src_batch = src_paths[start:start + max(1, int(batch_size))]
        gen_batch = gen_paths[start:start + max(1, int(batch_size))]
        src_images = _load_images(src_batch)
        gen_images = _load_images(gen_batch)
        src_inputs = processor(images=src_images, return_tensors="pt")
        gen_inputs = processor(images=gen_images, return_tensors="pt")
        src_inputs = {k: v.to(device) for k, v in src_inputs.items()}
        gen_inputs = {k: v.to(device) for k, v in gen_inputs.items()}
        src_out = model(**src_inputs, output_hidden_states=True)
        gen_out = model(**gen_inputs, output_hidden_states=True)
        src_tokens = F.normalize(src_out.hidden_states[-2][:, 1:, :].float(), p=2, dim=-1)
        gen_tokens = F.normalize(gen_out.hidden_states[-2][:, 1:, :].float(), p=2, dim=-1)
        src_ssm = torch.bmm(src_tokens, src_tokens.transpose(1, 2))
        gen_ssm = torch.bmm(gen_tokens, gen_tokens.transpose(1, 2))
        vals.extend(F.mse_loss(gen_ssm, src_ssm, reduction="none").mean(dim=(1, 2)).cpu().tolist())
        for img in src_images + gen_images:
            img.close()
    return _mean(vals)


def _image_name(row: dict[str, str]) -> str:
    return str(row.get("gen_image") or row.get("image") or "").strip()


def _resolve_source_image_path(source_root: Path, src_style: str, src_image: str, src_stem: str) -> Path | None:
    style_dir = source_root / src_style
    if not style_dir.is_dir():
        return None

    candidates: list[Path] = []
    if src_image:
        image_name = Path(src_image).name
        candidates.append(style_dir / image_name)
        if "__" not in image_name:
            candidates.append(style_dir / f"{src_style}__{image_name}")
    if src_stem:
        stem = str(src_stem).strip()
        if stem:
            candidates.append(style_dir / f"{stem}.jpg")
            candidates.append(style_dir / f"{stem}.png")
            if "__" not in stem:
                candidates.append(style_dir / f"{src_style}__{stem}.jpg")
                candidates.append(style_dir / f"{src_style}__{stem}.png")
    for path in candidates:
        if path.exists():
            return path
    return None


def _summary_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (payload.get("analysis") or {}).get("style_transfer_ability") or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute DINO structure for a manifest of metrics/images pairs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-test-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(str(args.device))
    processor = AutoImageProcessor.from_pretrained(str(args.model_name), local_files_only=bool(args.local_files_only))
    model = AutoModel.from_pretrained(str(args.model_name), local_files_only=bool(args.local_files_only)).to(device).eval()

    manifest_rows = _read_rows(args.manifest)
    out_rows: list[dict[str, Any]] = []
    for item in manifest_rows:
        label = str(item["label"])
        run = str(item.get("run", label))
        images_dir = Path(str(item["images_dir"]))
        metrics_csv = Path(str(item["metrics_csv"]))
        introstyle_summary = Path(str(item.get("introstyle_summary", "")).strip()) if str(item.get("introstyle_summary", "")).strip() else None
        source_root = Path(str(item.get("source_root", args.source_test_dir)))
        rows = _read_rows(metrics_csv)
        src_paths: list[Path] = []
        gen_paths: list[Path] = []
        for row in rows:
            src_style = str(row.get("src_style", "")).strip()
            src_image = str(row.get("src_image", "")).strip()
            src_stem = str(row.get("src_stem", "")).strip()
            src = _resolve_source_image_path(source_root, src_style, src_image, src_stem)
            if src is None:
                continue
            gen = images_dir / Path(_image_name(row)).name
            if src.exists() and gen.exists():
                src_paths.append(src)
                gen_paths.append(gen)
        dino_val = _ssm_distance(
            model,
            processor,
            src_paths,
            gen_paths,
            device=device,
            batch_size=max(1, int(args.batch_size)),
        )
        intro = _summary_metrics(introstyle_summary) if introstyle_summary is not None else {}
        out_rows.append(
            {
                "label": label,
                "run": run,
                "n_pairs": len(src_paths),
                "dino_structure": dino_val,
                "introstyle_target_style_score": intro.get("introstyle_target_style_score"),
                "introstyle_delta_idt": intro.get("introstyle_delta_idt"),
                "introstyle_style_margin": intro.get("introstyle_style_margin"),
                "images_dir": str(images_dir),
                "source_root": str(source_root),
                "metrics_csv": str(metrics_csv),
                "introstyle_summary": str(introstyle_summary) if introstyle_summary is not None else "",
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
