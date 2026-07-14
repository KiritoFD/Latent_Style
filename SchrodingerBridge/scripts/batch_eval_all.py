"""Batch eval all checkpoints after training: CLIP/LPIPS eval + DINO eval.

Infra fix: training saves checkpoints only (full_eval_each_epoch=false).
Run this script after training to evaluate all checkpoints.

DINO optimization: load DINOv2 once, extract source/reference features once,
only re-extract generated features per epoch. ~5x faster than per-epoch subprocess.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DINO_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_dino(model_name: str, device: str, cache_dir: str, allow_network: bool):
    os.environ["HF_HUB_OFFLINE"] = "0" if allow_network else "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0" if allow_network else "1"
    if not allow_network and cache_dir:
        parts = model_name.split("/")
        if len(parts) == 2:
            repo_dir = Path(cache_dir) / "hub" / f"models--{parts[0]}--{parts[1]}"
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
def extract_features(paths: list[Path], model, device: str, batch_size: int):
    cls_features = []
    patch_features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        pixels = torch.stack([DINO_TRANSFORM(load_image(p)) for p in batch_paths]).to(device)
        output = model(pixels, output_hidden_states=True)
        cls_features.append(F.normalize(output.last_hidden_state[:, 0, :].float(), dim=-1).cpu())
        patches = F.normalize(output.hidden_states[-2][:, 1:, :].float(), dim=-1).cpu()
        patch_features.extend(patches[i] for i in range(patches.shape[0]))
    return torch.cat(cls_features, dim=0), patch_features


def list_style_images(style_dir: Path) -> list[Path]:
    return sorted(p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


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


def run_clip_eval(checkpoint_path: Path, out_dir: Path, eval_script: Path, train_cfg, extra_args: list[str], config_override: str = "") -> bool:
    """Run run_evaluation.py for one checkpoint. Returns True if success."""
    if (out_dir / "summary.json").exists():
        print(f"  [skip] summary.json exists")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(eval_script),
        "--checkpoint", str(checkpoint_path),
        "--output", str(out_dir),
        "--test_dir", str(train_cfg.test_image_dir),
        "--cache_dir", str(train_cfg.full_eval_cache_dir),
        "--clip_hf_cache_dir", str(train_cfg.full_eval_clip_hf_cache_dir),
        "--batch_size", str(int(train_cfg.full_eval_batch_size)),
        "--save_generated_images",
    ] + extra_args
    if config_override:
        cmd += ["--config_override", config_override]
    print(f"  [eval] running run_evaluation.py...")
    t0 = time.time()
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if ret.returncode != 0:
        print(f"  [ERROR] {ret.stderr[:300]}")
        return False
    print(f"  [eval] done in {dt:.1f}s")
    return True


def run_dino_for_epoch(eval_dir: Path, test_dir: Path, model, device: str, batch_size: int,
                       source_cls_all: torch.Tensor, source_patches_all: list[torch.Tensor],
                       source_index: dict, style_reference_cls: dict, style_reference_paths: dict,
                       max_refs_per_style: int) -> dict | None:
    """Run DINO eval for one epoch. Returns summary dict or None on error."""
    metrics_csv = eval_dir / "metrics.csv"
    if not metrics_csv.exists():
        print(f"  [dino] no metrics.csv, skipping")
        return None
    rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8-sig")))
    generated_paths = [resolve_generated_path(eval_dir, row["gen_image"]) for row in rows]
    source_paths = [resolve_source_path(test_dir, row) for row in rows]
    missing = [p for p in generated_paths + source_paths if not p.exists()]
    if missing:
        print(f"  [dino] missing {len(missing)} images, skipping")
        return None

    # Extract generated features (per epoch)
    generated_cls, generated_patches = extract_features(generated_paths, model, device, batch_size)

    dino_style, dino_content, dino_structure = [], [], []
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
        style_scores = reference_cls @ generated_cls[index]
        dino_style.append(float(style_scores.max().item()))

    def mean(values):
        return float(sum(values) / max(1, len(values)))

    off_indices = [i for i, row in enumerate(rows) if row["src_style"] != row["tgt_style"]]
    summary = {
        "protocol": "paper_canonical_dinov2_small",
        "n_all": len(rows),
        "n_off_diagonal": len(off_indices),
        "all_dino_s": mean(dino_style),
        "all_dino_c": mean(dino_content),
        "all_dino_structure": mean(dino_structure),
        "off_dino_s": mean([dino_style[i] for i in off_indices]),
        "off_dino_c": mean([dino_content[i] for i in off_indices]),
        "all_clip_s": mean([float(row["clip_style"]) for row in rows]),
        "all_lpips": mean([float(row["content_lpips"]) for row in rows]),
    }
    output_json = eval_dir / "dino_summary.json"
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch eval all checkpoints")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory with epoch_XXXX.pt files")
    parser.add_argument("--test_dir", required=True, help="Test image directory")
    parser.add_argument("--config", default="default_config.json", help="Config for eval params")
    parser.add_argument("--dino_model_name", default="facebook/dinov2-small")
    parser.add_argument("--dino_cache_dir", default="I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf")
    parser.add_argument("--dino_batch_size", type=int, default=8)
    parser.add_argument("--max_refs_per_style", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_network", action="store_true")
    parser.add_argument("--skip_clip", action="store_true", help="Skip CLIP/LPIPS eval (only DINO)")
    parser.add_argument("--skip_dino", action="store_true", help="Skip DINO eval")
    parser.add_argument("--config_override", default="", help="Config override json (e.g. eval_adain_20.json)")
    parser.add_argument("--output_subdir", default="full_eval", help="Output subdir name")
    args = parser.parse_args()

    # Load config for eval params
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from config_schema import load_experiment_config
    config = load_experiment_config(args.config)
    train_cfg = config.training

    ckpt_dir = Path(args.checkpoint_dir)
    test_dir = Path(args.test_dir)
    eval_subdir = args.output_subdir
    eval_script = Path(__file__).resolve().parent.parent / "src" / "utils" / "run_evaluation.py"

    # Find all checkpoints
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not checkpoints:
        print(f"No epoch_*.pt found in {ckpt_dir}")
        return
    print(f"Found {len(checkpoints)} checkpoints")

    # Extra eval args from config
    extra_args = []
    if train_cfg.full_eval_num_steps is not None:
        extra_args += ["--num_steps", str(int(train_cfg.full_eval_num_steps))]
    if train_cfg.full_eval_max_src_samples is not None:
        extra_args += ["--max_src_samples", str(int(train_cfg.full_eval_max_src_samples))]
    if train_cfg.full_eval_max_ref_compare is not None:
        extra_args += ["--max_ref_compare", str(int(train_cfg.full_eval_max_ref_compare))]
    if train_cfg.full_eval_max_ref_cache is not None:
        extra_args += ["--max_ref_cache", str(int(train_cfg.full_eval_max_ref_cache))]

    # Phase 1: CLIP/LPIPS eval (serial, with --save_generated_images)
    if not args.skip_clip:
        print(f"\n=== Phase 1: CLIP/LPIPS eval ({len(checkpoints)} checkpoints) ===")
        for i, ckpt in enumerate(checkpoints):
            print(f"[{i+1}/{len(checkpoints)}] {ckpt.name}")
            out_dir = ckpt.parent / eval_subdir / ckpt.stem
            run_clip_eval(ckpt, out_dir, eval_script, train_cfg, extra_args, args.config_override)

    # Phase 2: DINO eval (batch, reuse model + source/reference features)
    if not args.skip_dino:
        print(f"\n=== Phase 2: DINO eval (batch, reuse model) ===")
        print("[dino] Loading DINOv2 model...")
        model = load_dino(args.dino_model_name, args.device, args.dino_cache_dir, args.allow_network)
        batch_size = args.dino_batch_size

        # Extract source features once (shared across all epochs)
        # We need metrics.csv to know source paths; use first available
        sample_eval_dir = None
        for ckpt in checkpoints:
            d = ckpt.parent / eval_subdir / ckpt.stem
            if (d / "metrics.csv").exists():
                sample_eval_dir = d
                break
        if sample_eval_dir is None:
            print("[dino] No metrics.csv found, cannot extract source paths")
            return

        sample_rows = list(csv.DictReader((sample_eval_dir / "metrics.csv").open(encoding="utf-8-sig")))
        source_paths_all = [resolve_source_path(test_dir, row) for row in sample_rows]
        unique_sources = list(dict.fromkeys(source_paths_all))
        print(f"[dino] Extracting {len(unique_sources)} source features (once)...")
        source_cls_all, source_patches_all = extract_features(unique_sources, model, args.device, batch_size)
        source_index = {p: i for i, p in enumerate(unique_sources)}

        # Extract style reference features once (shared across all epochs)
        print(f"[dino] Extracting style reference features (once)...")
        style_reference_cls = {}
        style_reference_paths = {}
        for style_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            ref_paths = list_style_images(style_dir)[:args.max_refs_per_style]
            if not ref_paths:
                continue
            ref_cls, _ = extract_features(ref_paths, model, args.device, batch_size)
            style_reference_cls[style_dir.name] = ref_cls
            style_reference_paths[style_dir.name] = ref_paths
        print(f"[dino] {len(style_reference_cls)} style references loaded")

        # Run DINO for each epoch (only generated features re-extracted)
        for i, ckpt in enumerate(checkpoints):
            eval_dir = ckpt.parent / eval_subdir / ckpt.stem
            dino_json = eval_dir / "dino_summary.json"
            if dino_json.exists():
                print(f"[{i+1}/{len(checkpoints)}] {ckpt.name} [skip] dino_summary.json exists")
                continue
            print(f"[{i+1}/{len(checkpoints)}] {ckpt.name} [dino] running...")
            t0 = time.time()
            summary = run_dino_for_epoch(
                eval_dir, test_dir, model, args.device, batch_size,
                source_cls_all, source_patches_all, source_index,
                style_reference_cls, style_reference_paths, args.max_refs_per_style,
            )
            dt = time.time() - t0
            if summary:
                print(f"  [dino] DINO-S={summary['all_dino_s']:.4f} DINO-C={summary['all_dino_c']:.4f} ({dt:.1f}s)")

    # Final summary
    print(f"\n=== Final Summary ===")
    print(f"{'Ep':>3} | {'CLIP-S':>7} | {'LPIPS':>7} | {'DINO-S':>7} | {'DINO-C':>7}")
    print("-" * 50)
    for i, ckpt in enumerate(checkpoints):
        ep = i + 1
        eval_dir = ckpt.parent / eval_subdir / ckpt.stem
        summary_path = eval_dir / "summary.json"
        dino_path = eval_dir / "dino_summary.json"
        clip_s = lpips = ds = dc = 0.0
        if summary_path.exists():
            with open(summary_path) as f:
                d = json.load(f)
            ov = d.get("analysis", {}).get("all_pairs_overview", {})
            clip_s = float(ov.get("clip_style", 0) or 0)
            lpips = float(ov.get("content_lpips", 0) or 0)
        if dino_path.exists():
            with open(dino_path) as f:
                d = json.load(f)
            ds = float(d.get("all_dino_s", 0) or 0)
            dc = float(d.get("all_dino_c", 0) or 0)
        print(f"{ep:>3} | {clip_s:>7.4f} | {lpips:>7.4f} | {ds:>7.4f} | {dc:>7.4f}")


if __name__ == "__main__":
    main()
