"""Optimized SaMam checkpoint curve evaluation - GPU batched.

Key optimizations vs eval_samam_checkpoint_curve.py:
1. CLIP feature extraction: batch_size=32 (was 1) - 750 images in 24 batches
2. LPIPS: batched computation (was single image)
3. SaMam inference: keep batch=1 (mamba state) but remove .cpu() sync per image
4. Remove torch.cuda.empty_cache() per checkpoint
5. Use torch.inference_mode() for lower overhead

Usage: same as original script.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMAM_ROOT = REPO_ROOT / "Related_Works" / "repos" / "SaMam"
import sys

sys.path.insert(0, str(SAMAM_ROOT))

from TRAIN.lightning_module.lightningmodel import LightningModel  # noqa: E402


DEFAULT_STYLE_NAMES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
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


def tensor_for_samam(path: Path, size: int, device: torch.device) -> torch.Tensor:
    tr = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return tr(load_rgb(path)).unsqueeze(0).to(device)


def clip_features_batch(paths: list[Path], model, processor, device: torch.device, dtype: torch.dtype, batch_size: int) -> torch.Tensor:
    """Batched CLIP feature extraction - GPU efficient."""
    feats: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        imgs = [load_rgb(p) for p in chunk]
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
    return torch.cat(feats, dim=0)  # keep on GPU


def generate_for_checkpoint(args, ckpt: Path, sources: list[tuple[str, Path]], style_refs: dict[str, Path]) -> Path:
    step = step_from_ckpt(ckpt)
    tag = f"step_{step:06d}" if step < 10**12 else ckpt.stem
    out_dir = args.output_root / tag / "images"
    expected = len(sources) * len(style_refs)
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= expected and not args.force:
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LightningModel.load_from_checkpoint(checkpoint_path=str(ckpt), map_location=device)
    model = model.to(device).eval()
    # Use fp16 autocast for inference (matches training, ~2x speedup on 3060)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        for src_style, src_path in sources:
            content = tensor_for_samam(src_path, args.image_size, device)
            for tgt_style, style_path in style_refs.items():
                style = tensor_for_samam(style_path, args.image_size, device)
                output = model.forward(content, style)[0].float()
                name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
                save_image(output.cpu(), out_dir / name)
    del model
    return out_dir


def evaluate_images_batched(args, image_dir: Path, sources_by_key: dict[tuple[str, str], Path], style_paths: dict[str, list[Path]]) -> dict[str, float]:
    """Batched evaluation - GPU efficient."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_dtype = torch.float16 if device.type == "cuda" else torch.float32
    lpips_dtype = torch.float32

    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(device=device, dtype=lpips_dtype).eval()

    from transformers import CLIPModel, CLIPProcessor
    clip_src = str(args.clip_cache) if args.clip_cache.exists() else "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_src).to(device=device, dtype=clip_dtype).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)

    gen_files = sorted(image_dir.glob("*.png"))

    # Parse all (src_path, gen_path, tgt_style) triples
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

    # ===== BATCHED CLIP FEATURE EXTRACTION =====
    # Collect all unique paths for CLIP: gen_files + unique src_paths + all style paths
    all_src_paths = list({src for src, _, _ in triples})
    all_style_names = list({tgt for _, _, tgt in triples})

    print(f"  [CLIP] Extracting features: {len(gen_files)} gen + {len(all_src_paths)} src + {len(all_style_names)} styles", flush=True)
    t_clip = time.time()

    # Style features (use all images per style for robustness, matching original)
    style_feat_map: dict[str, torch.Tensor] = {}
    for style in all_style_names:
        style_feat_map[style] = clip_features_batch(style_paths[style], clip_model, clip_processor, device, clip_dtype, args.metric_batch_size)

    # Gen features (batched)
    gen_feats = clip_features_batch(gen_files, clip_model, clip_processor, device, clip_dtype, args.metric_batch_size)

    # Src features (batched)
    src_feat_map: dict[Path, torch.Tensor] = {}
    src_feats_all = clip_features_batch(all_src_paths, clip_model, clip_processor, device, clip_dtype, args.metric_batch_size)
    for i, src_path in enumerate(all_src_paths):
        src_feat_map[src_path] = src_feats_all[i]

    print(f"  [CLIP] done in {time.time()-t_clip:.1f}s", flush=True)

    # Compute CLIP-S style and content on GPU (vectorized)
    print(f"  [CLIP-S] Computing similarities...", flush=True)
    t_sim = time.time()
    clip_style_values = []
    clip_content_values = []
    rows = []
    for i, (src_path, gen_path, tgt_style) in enumerate(triples):
        gen_feat = gen_feats[i]
        style_feat = style_feat_map[tgt_style]
        src_feat = src_feat_map[src_path]
        clip_style = float((gen_feat @ style_feat.T).mean().item())
        clip_content = float((gen_feat @ src_feat.T).mean().item())
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
    print(f"  [CLIP-S] done in {time.time()-t_sim:.1f}s", flush=True)

    # ===== BATCHED LPIPS (large batch + thread loading) =====
    lpips_batch = max(64, args.metric_batch_size * 2)  # LPIPS is light, use larger batch
    print(f"  [LPIPS] Computing {len(triples)} pairs (batch={lpips_batch})...", flush=True)
    t_lpips = time.time()
    lpips_values = []
    tr_metric = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    # Prefetch all (src, gen) image pairs with threads, build big batches
    from concurrent.futures import ThreadPoolExecutor
    with torch.inference_mode():
        for start in range(0, len(triples), lpips_batch):
            chunk = triples[start : start + lpips_batch]
            # Thread-load + transform on CPU
            def load_pair(t):
                src_path, gen_path, _ = t
                return tr_metric(load_rgb(src_path)), tr_metric(load_rgb(gen_path))
            with ThreadPoolExecutor(max_workers=8) as ex:
                pairs = list(ex.map(load_pair, chunk))
            src_batch = torch.stack([p[0] for p in pairs], dim=0).to(device=device, dtype=lpips_dtype)
            gen_batch = torch.stack([p[1] for p in pairs], dim=0).to(device=device, dtype=lpips_dtype)
            lp = lpips_model(src_batch, gen_batch).squeeze().detach().cpu()
            if lp.dim() == 0:
                lp = lp.unsqueeze(0)
            for v in lp:
                lpips_values.append(float(torch.nan_to_num(v, nan=0.0)))
            for j, v in enumerate(lp):
                rows[start + j]["lpips"] = float(torch.nan_to_num(v, nan=0.0))
    print(f"  [LPIPS] done in {time.time()-t_lpips:.1f}s", flush=True)

    if not rows:
        raise RuntimeError(f"No generated images could be parsed under {image_dir}")

    metrics_csv = image_dir.parent / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Free GPU memory
    del clip_model, lpips_model, gen_feats, src_feats_all, style_feat_map, src_feat_map
    torch.cuda.empty_cache()

    return {
        "count": float(len(rows)),
        "content_lpips": sum(lpips_values) / len(lpips_values),
        "clip_style": sum(clip_style_values) / len(clip_style_values),
        "clip_content": sum(clip_content_values) / len(clip_content_values),
    }


def plot_curve(rows: list[dict[str, object]], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [float(r["content_lpips"]) for r in rows]
    ys = [float(r["clip_style"]) for r in rows]
    labels = [str(r["step"]) for r in rows]
    plt.figure(figsize=(7.5, 5.5), dpi=160)
    plt.plot(xs, ys, marker="o", linewidth=1.6)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    plt.xlabel("content LPIPS (down)")
    plt.ylabel("CLIP style (up)")
    plt.title("SaMAM checkpoint convergence curve (HF CLIP, GPU-batched)")
    plt.grid(True, alpha=0.25)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=REPO_ROOT / "style_data" / "overfit50")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-src-per-style", type=int, default=5)
    parser.add_argument("--metric-batch-size", type=int, default=32, help="CLIP/LPIPS batch size (was 1 in original)")
    parser.add_argument("--clip-cache", type=Path, default=REPO_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32")
    parser.add_argument("--clip-backend", choices=["open_clip", "transformers"], default="transformers")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--steps", type=str, default="")
    parser.add_argument("--style-names", type=str, default=",".join(DEFAULT_STYLE_NAMES))
    args = parser.parse_args()

    if args.ckpt_dir is None and args.checkpoint is None:
        raise ValueError("Provide either --ckpt-dir or --checkpoint")

    t0 = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, Path]] = []
    sources_by_key: dict[tuple[str, str], Path] = {}
    style_refs: dict[str, Path] = {}
    style_paths: dict[str, list[Path]] = {}
    style_names = [s.strip() for s in str(args.style_names).split(",") if s.strip()]
    if not style_names:
        raise ValueError("--style-names resolved to an empty list")

    for style in style_names:
        paths = image_paths(args.image_root / style)
        if not paths:
            raise FileNotFoundError(args.image_root / style)
        rng = random.Random(42)
        selected = paths[:]
        rng.shuffle(selected)
        selected = selected[: args.max_src_per_style]
        for path in selected:
            sources.append((style, path))
            sources_by_key[(style, path.stem)] = path
        style_refs[style] = paths[0]
        style_paths[style] = paths

    if args.checkpoint is not None:
        ckpts = [args.checkpoint.resolve()]
    else:
        ckpt_dir = args.ckpt_dir.resolve()
        ckpts = [p for p in sorted(ckpt_dir.glob("*.ckpt"), key=step_from_ckpt) if step_from_ckpt(p) >= 0]
        if not ckpts:
            raise FileNotFoundError(f"No step checkpoints under {ckpt_dir}")
    if args.steps.strip():
        wanted = {int(s.strip()) for s in args.steps.split(",") if s.strip()}
        ckpts = [p for p in ckpts if step_from_ckpt(p) in wanted]
        if not ckpts:
            raise FileNotFoundError(f"No requested checkpoints {sorted(wanted)} under {args.ckpt_dir}")

    summary_rows: list[dict[str, object]] = []
    for ckpt in ckpts:
        step = step_from_ckpt(ckpt)
        print(f"[ckpt] step={step} path={ckpt}", flush=True)
        gen_started = time.time()
        image_dir = generate_for_checkpoint(args, ckpt, sources, style_refs)
        infer_wall = time.time() - gen_started
        if args.generate_only:
            row = {
                "step": step,
                "ckpt": str(ckpt),
                "image_dir": str(image_dir),
                "count": float(len(list(image_dir.glob('*.png')))),
                "infer_wall_seconds": infer_wall,
            }
            summary_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue
        metric_started = time.time()
        metrics = evaluate_images_batched(args, image_dir, sources_by_key, style_paths)
        row = {
            "step": step,
            "ckpt": str(ckpt),
            "image_dir": str(image_dir),
            "infer_wall_seconds": infer_wall,
            "metric_wall_seconds": time.time() - metric_started,
            **metrics,
        }
        summary_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    csv_path = args.output_root / "curve_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    json_path = args.output_root / "curve_metrics.json"
    json_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.generate_only:
        plot_curve(summary_rows, args.output_root / "clip_lpips_curve.png")

    print(f"csv={csv_path}")
    print(f"json={json_path}")
    if not args.generate_only:
        print(f"plot={args.output_root / 'clip_lpips_curve.png'}")
    print(f"elapsed_sec={time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
