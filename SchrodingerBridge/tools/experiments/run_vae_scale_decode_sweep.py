from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.inference import LGTInference, encode_image, load_vae  # noqa: E402


STYLE_SUBDIRS = ["photo", "Hayao", "monet", "vangogh", "cezanne"]


def _pil_to_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size))
    arr = torch.from_numpy(np.array(image)).float() / 255.0
    return arr.permute(2, 0, 1)


@torch.no_grad()
def _decode_with_scale(vae, latent: torch.Tensor, scale: float, device: str) -> torch.Tensor:
    latent = latent.to(device, dtype=torch.float16) / max(float(scale), 1e-12)
    image = vae.decode(latent).sample
    image = (image + 1.0) / 2.0
    return torch.clamp(image, 0.0, 1.0)


def _source_items(test_dir: Path, style_names: list[str], max_src_samples: int) -> list[dict]:
    items: list[dict] = []
    for sid, style in enumerate(style_names):
        paths = sorted(
            p for p in (test_dir / style).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        rng = random.Random(42)
        indices = list(range(len(paths)))
        rng.shuffle(indices)
        if max_src_samples > 0:
            indices = indices[:max_src_samples]
        sampled = [paths[i] for i in indices]
        for p in sampled:
            items.append({"path": p, "style_id": sid, "style_name": style})
    return items


def _scale_tag(scale: float) -> str:
    return f"s{scale:.5f}".replace(".", "p")


@torch.no_grad()
def generate_scale_decodes(
    checkpoint: Path,
    test_dir: Path,
    output_root: Path,
    scales: list[float],
    *,
    style_names: list[str],
    batch_size: int,
    max_src_samples: int,
    image_size: int,
    device: str,
    num_steps: int,
    step_size: float | None,
    style_strength: float | None,
    seed: int,
) -> dict:
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    sources = _source_items(test_dir, style_names, max_src_samples)
    lgt = LGTInference(
        str(checkpoint),
        device=device,
        num_steps=num_steps,
        step_size=step_size,
        style_strength=style_strength,
    )
    vae = load_vae(device=device)
    vae_scale = float(getattr(vae.config, "scaling_factor", 0.18215))
    model_scale = float(getattr(lgt.model, "latent_scale_factor", vae_scale))
    scale_in = model_scale / max(vae_scale, 1e-12)

    for scale in scales:
        (output_root / _scale_tag(scale) / "images").mkdir(parents=True, exist_ok=True)

    total_batches = (len(sources) + batch_size - 1) // batch_size
    for bidx, start in enumerate(range(0, len(sources), batch_size), start=1):
        batch = sources[start : start + batch_size]
        print(f"  generate {checkpoint.parent.name} batch {bidx}/{total_batches}")
        src = torch.stack([_pil_to_tensor(x["path"], image_size) for x in batch]).to(device)
        autocast_enabled = str(device).startswith("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            z_src = encode_image(vae, src, device=device).float()
            if abs(scale_in - 1.0) > 1e-5:
                z_src = z_src * scale_in
            z0 = lgt.inversion(z_src)
            for tgt_id, tgt_name in enumerate(style_names):
                tgt_ids = torch.full((len(batch),), tgt_id, dtype=torch.long, device=device)
                z_gen = lgt.generation(z0, tgt_ids)
                for scale in scales:
                    imgs = _decode_with_scale(vae, z_gen, scale, device=device).cpu()
                    out_dir = output_root / _scale_tag(scale) / "images"
                    for i, item in enumerate(batch):
                        out_name = f"{item['style_name']}_{item['path'].stem}_to_{tgt_name}.jpg"
                        save_image(imgs[i], out_dir / out_name)

    return {
        "checkpoint": str(checkpoint),
        "test_dir": str(test_dir),
        "num_sources": len(sources),
        "num_pairs": len(sources) * len(style_names),
        "vae_config_scaling_factor": vae_scale,
        "model_latent_scale_factor": model_scale,
        "decode_scales": scales,
        "seed": seed,
    }


def run_full_eval(
    output_dir: Path,
    test_dir: Path,
    *,
    style_names: list[str],
    batch_size: int,
    cache_dir: str,
    clip_hf_cache_dir: str,
    eval_lpips_chunk_size: int,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "src/utils/run_evaluation.py"),
        "--output",
        str(output_dir),
        "--reuse_generated",
        "--force_regen",
        "--test_dir",
        str(test_dir),
        "--style_subdirs",
        ",".join(style_names),
        "--batch_size",
        str(batch_size),
        "--cache_dir",
        cache_dir,
        "--clip_hf_cache_dir",
        clip_hf_cache_dir,
        "--clip_backend",
        "hf",
        "--eval_lpips_chunk_size",
        str(eval_lpips_chunk_size),
    ]
    print("[eval]", output_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key, "") not in {"", "N/A"}]
    return sum(vals) / max(1, len(vals))


def summarize_twofold(output_root: Path, scales: list[float]) -> dict:
    summary = {}
    for scale in scales:
        tag = _scale_tag(scale)
        metrics_path = output_root / tag / "metrics.csv"
        rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
        folds = {
            "fold0": [r for i, r in enumerate(rows) if i % 2 == 0],
            "fold1": [r for i, r in enumerate(rows) if i % 2 == 1],
            "all": rows,
        }
        summary[tag] = {"scale": scale}
        for fold_name, fold_rows in folds.items():
            summary[tag][fold_name] = {
                "count": len(fold_rows),
                "content_lpips": _mean(fold_rows, "content_lpips"),
                "clip_style": _mean(fold_rows, "clip_style"),
                "clip_content": _mean(fold_rows, "clip_content"),
            }
    return summary


def summarize_across_checkpoints(manifest: dict) -> dict:
    buckets: dict[str, list[dict]] = {}
    for ckpt in manifest.get("checkpoints", []):
        ckpt_name = Path(str(ckpt.get("checkpoint", ""))).parent.name
        for tag, item in ckpt.get("twofold", {}).items():
            row = dict(item.get("all", {}))
            row["scale"] = float(item.get("scale"))
            row["checkpoint"] = ckpt_name
            buckets.setdefault(tag, []).append(row)

    aggregate = {}
    for tag, rows in buckets.items():
        if not rows:
            continue
        keys = ["content_lpips", "clip_style", "clip_content"]
        aggregate[tag] = {
            "scale": rows[0]["scale"],
            "num_checkpoints": len(rows),
            "checkpoints": [r["checkpoint"] for r in rows],
        }
        for key in keys:
            vals = [float(r[key]) for r in rows if key in r]
            aggregate[tag][key] = sum(vals) / max(1, len(vals))
        aggregate[tag]["score_style_minus_lpips"] = aggregate[tag]["clip_style"] - aggregate[tag]["content_lpips"]
    ranked = sorted(
        aggregate.items(),
        key=lambda kv: (
            kv[1]["score_style_minus_lpips"],
            kv[1]["clip_style"],
            -kv[1]["content_lpips"],
        ),
        reverse=True,
    )
    return {
        "by_scale": aggregate,
        "ranked_by_style_minus_lpips": [{"tag": tag, **vals} for tag, vals in ranked],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode-scale sweep using fixed generated latents.")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=ROOT.parent / "style_data/overfit50")
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/vae_scale_decode_sweep")
    parser.add_argument("--scales", type=str, default="0.18215,0.20500,0.22319,0.23218")
    parser.add_argument("--style-subdirs", type=str, default=",".join(STYLE_SUBDIRS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-src-samples", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--style-strength", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Seed for VAE latent sampling and generation. Set <0 to leave RNG untouched.")
    parser.add_argument("--cache-dir", type=str, default="../Cycle-NCE/eval_cache")
    parser.add_argument("--clip-hf-cache-dir", type=str, default="../Cycle-NCE/eval_cache/hf")
    parser.add_argument("--eval-lpips-chunk-size", type=int, default=2)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    style_names = [x.strip() for x in args.style_subdirs.split(",") if x.strip()]
    manifest = {"scales": scales, "checkpoints": []}

    for checkpoint in args.checkpoints:
        ckpt_name = checkpoint.parent.name
        ckpt_out = args.out_root / ckpt_name
        ckpt_out.mkdir(parents=True, exist_ok=True)
        if not args.skip_generate:
            meta = generate_scale_decodes(
                checkpoint,
                args.test_dir,
                ckpt_out,
                scales,
                style_names=style_names,
                batch_size=args.batch_size,
                max_src_samples=args.max_src_samples,
                image_size=args.image_size,
                device=args.device,
                num_steps=args.num_steps,
                step_size=args.step_size,
                style_strength=args.style_strength,
                seed=args.seed,
            )
            with (ckpt_out / "generation_manifest.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        if not args.skip_eval:
            for scale in scales:
                run_full_eval(
                    ckpt_out / _scale_tag(scale),
                    args.test_dir,
                    style_names=style_names,
                    batch_size=args.eval_batch_size,
                    cache_dir=args.cache_dir,
                    clip_hf_cache_dir=args.clip_hf_cache_dir,
                    eval_lpips_chunk_size=args.eval_lpips_chunk_size,
                )
        twofold = summarize_twofold(ckpt_out, scales)
        with (ckpt_out / "twofold_summary.json").open("w", encoding="utf-8") as f:
            json.dump(twofold, f, indent=2)
        manifest["checkpoints"].append({"checkpoint": str(checkpoint), "output": str(ckpt_out), "twofold": twofold})

    manifest["aggregate_across_checkpoints"] = summarize_across_checkpoints(manifest)
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
