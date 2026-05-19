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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _summary_metrics(eval_dir: Path) -> dict[str, Any]:
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return ((payload.get("analysis") or {}).get("all_pairs_overview") or {})


def _image_path(eval_dir: Path, rel_path: str) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        return p
    direct = eval_dir / p
    if direct.exists():
        return direct
    images = eval_dir / "images" / p.name
    return images


def _source_path(test_dir: Path, style: str, filename: str) -> Path:
    return test_dir / style / filename


@torch.inference_mode()
def _ssm_distance(
    model: AutoModel,
    processor: AutoImageProcessor,
    src_paths: list[Path],
    gen_paths: list[Path],
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    vals: list[float] = []
    for start in range(0, len(src_paths), batch_size):
        src_batch = src_paths[start:start + batch_size]
        gen_batch = gen_paths[start:start + batch_size]
        src_images = [Image.open(p).convert("RGB") for p in src_batch]
        gen_images = [Image.open(p).convert("RGB") for p in gen_batch]
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
    return float(sum(vals) / max(len(vals), 1))


def eval_dir(
    eval_dir: Path,
    *,
    label: str,
    test_dir: Path,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    metrics_csv = eval_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"missing metrics.csv: {metrics_csv}")
    rows = _read_rows(metrics_csv)
    src_paths: list[Path] = []
    gen_paths: list[Path] = []
    for row in rows:
        src = _source_path(test_dir, row["src_style"], row["src_image"])
        gen = _image_path(eval_dir, row["gen_image"])
        if src.exists() and gen.exists():
            src_paths.append(src)
            gen_paths.append(gen)
    if not src_paths:
        raise RuntimeError(f"no valid image pairs under {eval_dir}")
    summary = _summary_metrics(eval_dir)
    dino_structure = _ssm_distance(
        model,
        processor,
        src_paths,
        gen_paths,
        device=device,
        batch_size=batch_size,
    )
    return {
        "label": label,
        "eval_dir": str(eval_dir),
        "n_pairs": len(src_paths),
        "dino_structure": dino_structure,
        "clip_style_all": summary.get("clip_style"),
        "clip_content_all": summary.get("clip_content"),
        "content_lpips_all": summary.get("content_lpips"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("../style_data/overfit50"))
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--model-name", type=str, default="facebook/dinov2-small")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("items", nargs="+", help="label=eval_dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_name} on {device}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    rows: list[dict[str, Any]] = []
    for item in args.items:
        if "=" not in item:
            raise ValueError(f"Expected label=eval_dir, got {item}")
        label, raw_dir = item.split("=", 1)
        print(f"Evaluating {label}: {raw_dir}", flush=True)
        rows.append(
            eval_dir(
                Path(raw_dir),
                label=label,
                test_dir=args.test_dir,
                model=model,
                processor=processor,
                device=device,
                batch_size=max(1, int(args.batch_size)),
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "n_pairs", "dino_structure", "clip_style_all", "clip_content_all", "content_lpips_all", "eval_dir"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
