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


def tensor_for_metric(path: Path, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tr = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    return tr(load_rgb(path)).unsqueeze(0).to(device=device, dtype=dtype)


def clip_features(paths: list[Path], model, processor, device: torch.device, dtype: torch.dtype, batch_size: int) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        imgs = [load_rgb(p) for p in paths[start : start + batch_size]]
        batch = processor(images=imgs, return_tensors="pt")
        batch = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in batch.items()
        }
        with torch.no_grad():
            out = model.get_image_features(**batch)
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            feats.append(F.normalize(feat.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)


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
    with torch.no_grad():
        for src_style, src_path in sources:
            content = tensor_for_samam(src_path, args.image_size, device)
            for tgt_style, style_path in style_refs.items():
                style = tensor_for_samam(style_path, args.image_size, device)
                output = model.forward(content, style)[0].detach().cpu()
                # Double separators keep style names with underscores unambiguous.
                name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
                save_image(output, out_dir / name)
    del model
    torch.cuda.empty_cache()
    return out_dir


def evaluate_images(args, image_dir: Path, sources_by_key: dict[tuple[str, str], Path], style_paths: dict[str, list[Path]]) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    import lpips
    from transformers import CLIPModel, CLIPProcessor

    lpips_model = lpips.LPIPS(net="alex").to(device=device, dtype=dtype).eval()
    clip_src = str(args.clip_cache) if args.clip_cache.exists() else "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_src).to(device=device, dtype=dtype).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)

    gen_files = sorted(image_dir.glob("*.png"))
    style_feat_cache = {
        style: clip_features(paths, clip_model, clip_processor, device, dtype, args.metric_batch_size)
        for style, paths in style_paths.items()
    }
    src_feat_cache: dict[Path, torch.Tensor] = {}

    lpips_values: list[float] = []
    clip_style_values: list[float] = []
    clip_content_values: list[float] = []
    rows: list[dict[str, object]] = []

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

        with torch.no_grad():
            src_t = tensor_for_metric(src_path, args.image_size, device, dtype)
            gen_t = tensor_for_metric(gen_path, args.image_size, device, dtype)
            lp = float(lpips_model(src_t, gen_t).squeeze().detach().cpu().item())
        lpips_values.append(lp)

        gen_feat = clip_features([gen_path], clip_model, clip_processor, device, dtype, 1)
        style_feat = style_feat_cache[tgt_style]
        clip_style = float((gen_feat @ style_feat.T).mean().item())
        clip_style_values.append(clip_style)

        if src_path not in src_feat_cache:
            src_feat_cache[src_path] = clip_features([src_path], clip_model, clip_processor, device, dtype, 1)
        clip_content = float((gen_feat @ src_feat_cache[src_path].T).mean().item())
        clip_content_values.append(clip_content)
        rows.append(
            {
                "image": str(gen_path),
                "src_style": src_style,
                "src_stem": src_stem,
                "tgt_style": tgt_style,
                "lpips": lp,
                "clip_style": clip_style,
                "clip_content": clip_content,
            }
        )

    if not rows:
        raise RuntimeError(f"No generated images could be parsed under {image_dir}")

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
    plt.title("SaMAM checkpoint convergence curve")
    plt.grid(True, alpha=0.25)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=REPO_ROOT / "style_data" / "overfit50")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-src-per-style", type=int, default=5)
    parser.add_argument("--metric-batch-size", type=int, default=4)
    parser.add_argument("--clip-cache", type=Path, default=REPO_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--generate-only", action="store_true", help="Only generate SB-compatible images; do not compute ad-hoc metrics.")
    parser.add_argument(
        "--steps",
        type=str,
        default="",
        help="Optional comma-separated checkpoint steps to generate, for example 14000,15000.",
    )
    parser.add_argument(
        "--style-names",
        type=str,
        default=",".join(DEFAULT_STYLE_NAMES),
        help="Comma-separated style folder names used for generation/evaluation.",
    )
    args = parser.parse_args()

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

    ckpts = [p for p in sorted(args.ckpt_dir.glob("step-step=*.ckpt"), key=step_from_ckpt)]
    if not ckpts:
        raise FileNotFoundError(f"No step checkpoints under {args.ckpt_dir}")
    if args.steps.strip():
        wanted = {int(s.strip()) for s in args.steps.split(",") if s.strip()}
        ckpts = [p for p in ckpts if step_from_ckpt(p) in wanted]
        if not ckpts:
            raise FileNotFoundError(f"No requested checkpoints {sorted(wanted)} under {args.ckpt_dir}")

    summary_rows: list[dict[str, object]] = []
    for ckpt in ckpts:
        step = step_from_ckpt(ckpt)
        print(f"[ckpt] step={step} path={ckpt}", flush=True)
        image_dir = generate_for_checkpoint(args, ckpt, sources, style_refs)
        if args.generate_only:
            row = {"step": step, "ckpt": str(ckpt), "image_dir": str(image_dir), "count": float(len(list(image_dir.glob('*.png'))))}
            summary_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue
        metrics = evaluate_images(args, image_dir, sources_by_key, style_paths)
        row = {"step": step, "ckpt": str(ckpt), "image_dir": str(image_dir), **metrics}
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
