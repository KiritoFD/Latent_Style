from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


CLASSES = ["Realism", "Impressionism", "Post_Impressionism", "Expressionism", "Symbolism"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _generated_records(output_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tgt_style in CLASSES:
        for path in sorted((output_root / tgt_style).glob("*.png")):
            stem = path.stem
            if "__to__" not in stem:
                continue
            src_key, parsed_tgt = stem.split("__to__", 1)
            if "__" not in src_key:
                continue
            src_style, src_stem = src_key.split("__", 1)
            rows.append(
                {
                    "src_style": src_style,
                    "src_stem": src_stem,
                    "tgt_style": parsed_tgt or tgt_style,
                    "path": str(path),
                    "source_key": src_key,
                }
            )
    return rows


def _complete_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_key"]].append(row)
    complete_sources = {
        key
        for key, items in by_source.items()
        if set(CLASSES).issubset({item["tgt_style"] for item in items})
    }
    return [row for row in rows if row["source_key"] in complete_sources and row["tgt_style"] in CLASSES]


def _load_tensor(path: Path, size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device=device, dtype=dtype)


def _clip_image_features(
    paths: list[Path],
    *,
    model,
    processor,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    feats = []
    for start in range(0, len(paths), batch_size):
        imgs = [Image.open(path).convert("RGB") for path in paths[start : start + batch_size]]
        batch = processor(images=imgs, return_tensors="pt")
        batch = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model.get_image_features(**batch)
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            feat = F.normalize(feat.float(), dim=-1).cpu()
        feats.append(feat)
        del batch, out, feat
        if device == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--result-csv", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clip-cache", type=Path, default=Path(r"G:\GitHub\Latent_Style\Cycle-NCE\eval_cache\manual_clip\openai-clip-vit-base-patch32"))
    args = parser.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    rows_all = _generated_records(args.output_root)
    rows = _complete_records(rows_all)
    if not rows:
        raise RuntimeError(f"No complete 5-target source groups found under {args.output_root}")

    source_lookup: dict[tuple[str, str], Path] = {}
    for style in CLASSES:
        for path in _image_paths(args.image_root / style):
            source_lookup[(style, path.stem)] = path

    style_paths = {style: _image_paths(args.image_root / style) for style in CLASSES}

    print(f"device={device} dtype={dtype} generated={len(rows_all)} complete_eval_images={len(rows)}")
    print(f"complete_sources={len(set(row['source_key'] for row in rows))}")

    import lpips
    from transformers import CLIPModel, CLIPProcessor

    lpips_model = lpips.LPIPS(net="alex").to(device=device, dtype=dtype).eval()
    clip_src = str(args.clip_cache) if args.clip_cache.exists() else "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_src).to(device=device, dtype=dtype).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)

    gen_paths = [Path(row["path"]) for row in rows]
    gen_feats = _clip_image_features(
        gen_paths,
        model=clip_model,
        processor=clip_processor,
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
    )
    style_feats = {
        style: _clip_image_features(
            paths,
            model=clip_model,
            processor=clip_processor,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
        )
        for style, paths in style_paths.items()
    }
    source_feat_cache: dict[Path, torch.Tensor] = {}

    out_rows: list[dict[str, object]] = []
    lpips_scores: list[float] = []
    clip_style_scores: list[float] = []
    clip_content_scores: list[float] = []

    for idx, row in enumerate(rows):
        gen_path = Path(row["path"])
        src_path = source_lookup.get((row["src_style"], row["src_stem"]))
        if src_path is None:
            print(f"[WARN] missing source for {row['source_key']}")
            continue

        with torch.no_grad():
            src_tensor = _load_tensor(src_path, args.image_size, device, dtype)
            gen_tensor = _load_tensor(gen_path, args.image_size, device, dtype)
            lp = float(lpips_model(src_tensor, gen_tensor).squeeze().detach().cpu().item())
        lpips_scores.append(lp)
        del src_tensor, gen_tensor
        if device == "cuda":
            torch.cuda.empty_cache()

        gen_feat = gen_feats[idx : idx + 1]
        sf = style_feats[row["tgt_style"]]
        clip_style = float((gen_feat @ sf.T).mean().item())
        clip_style_scores.append(clip_style)

        if src_path not in source_feat_cache:
            source_feat_cache[src_path] = _clip_image_features(
                [src_path],
                model=clip_model,
                processor=clip_processor,
                device=device,
                dtype=dtype,
                batch_size=1,
            )
        clip_content = float((gen_feat @ source_feat_cache[src_path].T).mean().item())
        clip_content_scores.append(clip_content)

        out_rows.append(
            {
                **row,
                "src_path": str(src_path),
                "lpips": lp,
                "clip_style": clip_style,
                "clip_content": clip_content,
            }
        )

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    by_tgt: dict[str, dict[str, float | int | None]] = {}
    for style in CLASSES:
        style_rows = [r for r in out_rows if r["tgt_style"] == style]
        by_tgt[style] = {
            "count": len(style_rows),
            "lpips": mean([float(r["lpips"]) for r in style_rows]),
            "clip_style": mean([float(r["clip_style"]) for r in style_rows]),
            "clip_content": mean([float(r["clip_content"]) for r in style_rows]),
        }

    summary = {
        "output_root": str(args.output_root),
        "image_root": str(args.image_root),
        "device": device,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "generated_png_count": len(rows_all),
        "complete_eval_images": len(out_rows),
        "complete_sources": len(set(row["source_key"] for row in rows)),
        "target_counts": dict(Counter(row["tgt_style"] for row in out_rows)),
        "overall": {
            "lpips": mean(lpips_scores),
            "clip_style": mean(clip_style_scores),
            "clip_content": mean(clip_content_scores),
        },
        "by_target": by_tgt,
        "elapsed_sec": round(time.time() - t0, 3),
    }

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.result_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    print(f"summary={args.result_json}")
    print(f"rows={args.result_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
