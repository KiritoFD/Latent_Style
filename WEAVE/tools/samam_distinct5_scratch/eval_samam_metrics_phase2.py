"""SaMam metric-only: batch evaluate CLIP-S/LPIPS for all checkpoints with images.

Phase 2 of two-phase eval: GPU-intensive metric computation only.
All inference already done in Phase 1, so this is pure GPU-batched CLIP+LPIPS.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def step_from_ckpt(path: Path) -> int:
    m = re.search(r"step=(\d+)", path.name)
    if m:
        return int(m.group(1))
    if path.name == "last.ckpt":
        return 10**12
    return -1


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_rgb_batch(paths: list[Path]) -> list[Image.Image]:
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as ex:
        return list(ex.map(load_rgb, paths))


def clip_features_batch(paths: list[Path], model, processor, device, dtype, batch_size: int) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        imgs = load_rgb_batch(chunk)
        batch = processor(images=imgs, return_tensors="pt")
        batch = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in batch.items()
        }
        with torch.inference_mode():
            out = model.get_image_features(**batch)
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            feats.append(F.normalize(feat.float(), dim=-1))
        for img in imgs:
            img.close()
    return torch.cat(feats, dim=0)


def evaluate_one_checkpoint(args, image_dir: Path, sources_by_key, style_paths, clip_model, clip_processor, lpips_model, device, clip_dtype) -> dict:
    gen_files = sorted(image_dir.glob("*.png"))
    triples: list[tuple[Path, Path, str]] = []
    for gen_path in gen_files:
        stem = gen_path.stem
        if "__to__" in stem:
            src_key, tgt_style = stem.split("__to__", 1)
        elif "_to_" in stem:
            src_key, tgt_style = stem.split("_to_", 1)
        else:
            continue
        if "__" not in src_key:
            continue
        src_style, src_stem = src_key.split("__", 1)
        src_path = sources_by_key[(src_style, src_stem)]
        triples.append((src_path, gen_path, tgt_style))

    all_src_paths = list({src for src, _, _ in triples})
    all_style_names = list({tgt for _, _, tgt in triples})

    # CLIP features
    style_feat_map = {s: clip_features_batch(style_paths[s], clip_model, clip_processor, device, clip_dtype, args.metric_batch_size) for s in all_style_names}
    gen_feats = clip_features_batch(gen_files, clip_model, clip_processor, device, clip_dtype, args.metric_batch_size)
    src_feats_all = clip_features_batch(all_src_paths, clip_model, clip_processor, device, clip_dtype, args.metric_batch_size)
    src_feat_map = {p: src_feats_all[i] for i, p in enumerate(all_src_paths)}

    clip_style_values = []
    clip_content_values = []
    rows = []
    for i, (src_path, gen_path, tgt_style) in enumerate(triples):
        gen_feat = gen_feats[i]
        clip_style = float((gen_feat @ style_feat_map[tgt_style].T).mean().item())
        clip_content = float((gen_feat @ src_feat_map[src_path].T).mean().item())
        clip_style_values.append(clip_style)
        clip_content_values.append(clip_content)
        rows.append({
            "image": str(gen_path),
            "src_style": src_path.parent.name,
            "src_stem": src_path.stem,
            "tgt_style": tgt_style,
            "clip_style": clip_style,
            "clip_content": clip_content,
        })

    # LPIPS
    lpips_dtype = torch.float32
    lpips_batch = max(64, args.metric_batch_size * 2)
    tr_metric = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    def load_pair(t):
        src_path, gen_path, _ = t
        return tr_metric(load_rgb(src_path)), tr_metric(load_rgb(gen_path))

    lpips_values = []
    with torch.inference_mode():
        for start in range(0, len(triples), lpips_batch):
            chunk = triples[start : start + lpips_batch]
            with ThreadPoolExecutor(max_workers=8) as ex:
                pairs = list(ex.map(load_pair, chunk))
            src_batch = torch.stack([p[0] for p in pairs], dim=0).to(device=device, dtype=lpips_dtype)
            gen_batch = torch.stack([p[1] for p in pairs], dim=0).to(device=device, dtype=lpips_dtype)
            lp = lpips_model(src_batch, gen_batch).squeeze().detach().cpu()
            if lp.dim() == 0:
                lp = lp.unsqueeze(0)
            for j, v in enumerate(lp):
                rows[start + j]["lpips"] = float(torch.nan_to_num(v, nan=0.0))
                lpips_values.append(float(torch.nan_to_num(v, nan=0.0)))

    metrics_csv = image_dir.parent / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "count": float(len(rows)),
        "content_lpips": sum(lpips_values) / len(lpips_values),
        "clip_style": sum(clip_style_values) / len(clip_style_values),
        "clip_content": sum(clip_content_values) / len(clip_content_values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--clip-cache", type=Path, default=REPO_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32")
    parser.add_argument("--style-names", type=str, required=True)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_dtype = torch.float16 if device.type == "cuda" else torch.float32

    import lpips
    from transformers import CLIPModel, CLIPProcessor

    lpips_model = lpips.LPIPS(net="alex").to(device=device, dtype=torch.float32).eval()
    clip_src = str(args.clip_cache) if args.clip_cache.exists() else "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_src).to(device=device, dtype=clip_dtype).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)

    style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    sources_by_key = {}
    style_paths = {}
    import random
    for style in style_names:
        paths = image_paths(args.image_root / style)
        rng = random.Random(42)
        selected = paths[:]
        rng.shuffle(selected)
        selected = selected[: args.max_src_per_style]
        for path in selected:
            sources_by_key[(style, path.stem)] = path
        style_paths[style] = paths

    # Find all step dirs with images
    step_dirs = sorted([d for d in args.output_root.iterdir() if d.is_dir() and (d / "images").exists() and len(list((d / "images").glob("*.png"))) >= 750],
                       key=lambda d: step_from_ckpt(Path(d.name + ".ckpt")) if "step_" in d.name else 10**12)

    print(f"=== Phase 2: Metric eval only ({len(step_dirs)} checkpoints with images) ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)

    summary_rows = []
    for i, step_dir in enumerate(step_dirs):
        step = step_from_ckpt(Path(step_dir.name + ".ckpt")) if "step_" in step_dir.name else 10**12
        image_dir = step_dir / "images"
        print(f"[ckpt] step={step} ({i+1}/{len(step_dirs)})", flush=True)
        t_metric = time.time()
        metrics = evaluate_one_checkpoint(args, image_dir, sources_by_key, style_paths, clip_model, clip_processor, lpips_model, device, clip_dtype)
        row = {"step": step, "image_dir": str(image_dir), "metric_wall_seconds": time.time() - t_metric, **metrics}
        summary_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    csv_path = args.output_root / "curve_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.output_root / "curve_metrics.json").write_text(json.dumps(summary_rows, indent=2) + "\n")

    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Phase 2 done: {len(summary_rows)} checkpoints, {time.time()-t0:.1f}s total", flush=True)
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
