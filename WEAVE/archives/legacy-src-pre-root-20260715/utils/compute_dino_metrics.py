"""Compute the paper-canonical DINO metrics for an existing eval directory.

DINO-C: cosine between generated and source DINOv2 CLS embeddings.
DINO-S: maximum cosine between generated CLS and target-style reference CLS.
DINO-structure: penultimate patch-token self-similarity MSE (lower is better).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DINO_TRANSFORM = T.Compose(
    [
        T.Resize(224, interpolation=Image.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def list_style_images(style_dir: Path) -> list[Path]:
    return sorted(
        path for path in style_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def load_dino(model_name: str, device: str, cache_dir: str, allow_network: bool):
    os.environ["HF_HUB_OFFLINE"] = "0" if allow_network else "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0" if allow_network else "1"
    # If cache_dir has a local snapshot, use it directly to avoid network
    if not allow_network and cache_dir:
        from pathlib import Path as _P
        # Try standard HF hub layout: <cache_dir>/hub/models--<org>--<name>/snapshots/<rev>
        parts = model_name.split("/")
        if len(parts) == 2:
            repo_dir = _P(cache_dir) / "hub" / f"models--{parts[0]}--{parts[1]}"
            snap_root = repo_dir / "snapshots"
            if snap_root.exists():
                revisions = [p for p in snap_root.iterdir() if p.is_dir()]
                if revisions:
                    local_path = str(revisions[0])
                    print(f"[INFO] Loading DINOv2 from local snapshot: {local_path}")
                    return AutoModel.from_pretrained(local_path).to(device).eval()
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    return AutoModel.from_pretrained(model_name, **kwargs).to(device).eval()


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


@torch.inference_mode()
def extract_features(
    paths: list[Path],
    model,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    cls_features: list[torch.Tensor] = []
    patch_features: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        pixels = torch.stack([DINO_TRANSFORM(load_image(path)) for path in batch_paths]).to(device)
        output = model(pixels, output_hidden_states=True)
        cls_features.append(F.normalize(output.last_hidden_state[:, 0, :].float(), dim=-1).cpu())
        patches = F.normalize(output.hidden_states[-2][:, 1:, :].float(), dim=-1).cpu()
        patch_features.extend(patches[index] for index in range(patches.shape[0]))
    return torch.cat(cls_features, dim=0), patch_features


def resolve_generated_path(eval_dir: Path, raw_path: str) -> Path:
    direct = eval_dir / raw_path
    if direct.exists():
        return direct
    return eval_dir / "images" / raw_path


def resolve_source_path(test_dir: Path, row: dict[str, str]) -> Path:
    style_dir = test_dir / row["src_style"]
    direct = style_dir / row["src_image"]
    if direct.exists():
        return direct
    return style_dir / f"{row['src_style']}__{row['src_image']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--dino_model_name", default="facebook/dinov2-small")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_refs_per_style", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow_network", action="store_true")
    parser.add_argument("--exclude_source_from_style_refs", action="store_true")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    test_dir = Path(args.test_dir)
    rows = list(csv.DictReader((eval_dir / "metrics.csv").open(encoding="utf-8-sig")))
    generated_paths = [resolve_generated_path(eval_dir, row["gen_image"]) for row in rows]
    source_paths = [resolve_source_path(test_dir, row) for row in rows]
    missing = [path for path in generated_paths + source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} DINO input images; first: {missing[0]}")

    model = load_dino(args.dino_model_name, args.device, args.cache_dir, args.allow_network)
    batch_size = max(1, int(args.batch_size))
    generated_cls, generated_patches = extract_features(generated_paths, model, args.device, batch_size)

    unique_sources = list(dict.fromkeys(source_paths))
    source_cls_all, source_patches_all = extract_features(unique_sources, model, args.device, batch_size)
    source_index = {path: index for index, path in enumerate(unique_sources)}

    style_reference_paths: dict[str, list[Path]] = {}
    style_reference_cls: dict[str, torch.Tensor] = {}
    for style_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        ref_paths = list_style_images(style_dir)[: max(1, int(args.max_refs_per_style))]
        if not ref_paths:
            continue
        ref_cls, _ = extract_features(ref_paths, model, args.device, batch_size)
        style_reference_paths[style_dir.name] = ref_paths
        style_reference_cls[style_dir.name] = ref_cls

    dino_style: list[float] = []
    dino_content: list[float] = []
    dino_structure: list[float] = []
    for index, row in enumerate(rows):
        source_pos = source_index[source_paths[index]]
        content_score = float((generated_cls[index] * source_cls_all[source_pos]).sum().item())
        dino_content.append(content_score)

        generated_ssm = generated_patches[index] @ generated_patches[index].T
        source_patches = source_patches_all[source_pos]
        source_ssm = source_patches @ source_patches.T
        dino_structure.append(float(F.mse_loss(generated_ssm, source_ssm).item()))

        target_style = row["tgt_style"]
        reference_cls = style_reference_cls[target_style]
        if args.exclude_source_from_style_refs:
            keep = torch.tensor(
                [path.resolve() != source_paths[index].resolve() for path in style_reference_paths[target_style]],
                dtype=torch.bool,
            )
            if keep.any():
                reference_cls = reference_cls[keep]
        style_scores = reference_cls @ generated_cls[index]
        dino_style.append(float(style_scores.max().item()))

    output_csv = Path(args.output_csv) if args.output_csv else eval_dir / "dino_metrics.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src_style", "tgt_style", "src_image", "gen_image", "dino_s", "dino_c", "dino_structure"])
        for row, style, content, structure in zip(rows, dino_style, dino_content, dino_structure):
            writer.writerow(
                [row["src_style"], row["tgt_style"], row["src_image"], row["gen_image"], style, content, structure]
            )

    def mean(values: list[float]) -> float:
        return float(sum(values) / max(1, len(values)))

    off_indices = [index for index, row in enumerate(rows) if row["src_style"] != row["tgt_style"]]
    summary = {
        "protocol": "paper_canonical_dinov2_small",
        "dino_s_definition": "max cosine CLS(gen), CLS(target-style reference)",
        "dino_c_definition": "cosine CLS(gen), CLS(source)",
        "dino_structure_definition": "penultimate patch self-similarity MSE",
        "exclude_source_from_style_refs": bool(args.exclude_source_from_style_refs),
        "n_all": len(rows),
        "n_off_diagonal": len(off_indices),
        "all_dino_s": mean(dino_style),
        "all_dino_c": mean(dino_content),
        "all_dino_structure": mean(dino_structure),
        "off_dino_s": mean([dino_style[index] for index in off_indices]),
        "off_dino_c": mean([dino_content[index] for index in off_indices]),
        "off_dino_structure": mean([dino_structure[index] for index in off_indices]),
        "all_clip_s": mean([float(row["clip_style"]) for row in rows]),
        "all_lpips": mean([float(row["content_lpips"]) for row in rows]),
    }
    output_json = Path(args.output_json) if args.output_json else eval_dir / "dino_summary.json"
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
