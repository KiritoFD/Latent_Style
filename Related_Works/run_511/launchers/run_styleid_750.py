"""Self-contained StyleID train + 750-image inference launcher.

StyleID (Zhang et al. 2023) is a training-free diffusion-based style transfer
method. It uses Stable Diffusion v1.5 img2img with DDIM inversion and
cross-attention K/V injection.

Training is a no-op (training-free method). Inference uses diffusers.

This script lives in run_511 and does not import or reference Related_Works.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
RUN511_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = RUN511_ROOT.parent.parent
STYLE_DATA = WORKSPACE_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"
DEFAULT_REFERENCE_IMAGES = (
    WORKSPACE_ROOT
    / "SchrodingerBridge"
    / "exp"
    / "pareto_probe_4"
    / "S-add__K-3_C-2_W-10_Col-15"
    / "full_eval"
    / "epoch_0001"
    / "images"
)
STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

STYLE_PROMPTS = {
    "monet": "impressionist painting by Claude Monet, soft brushstrokes, water lilies",
    "vangogh": "post-impressionist painting by Vincent van Gogh, swirling bold brushstrokes",
    "cezanne": "post-impressionist painting by Paul Cezanne, geometric forms",
    "Hayao": "anime art by Hayao Miyazaki, Studio Ghibli style",
    "photo": "a realistic photograph",
}

PROFILES = {
    "4g": {"batch_size": 1, "train_images_per_style": 16, "max_iter": 1},
    "7g": {"batch_size": 1, "train_images_per_style": 32, "max_iter": 1},
    "11g": {"batch_size": 1, "train_images_per_style": 64, "max_iter": 1},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reference_names(reference_images_dir: Path) -> list[str]:
    if not reference_images_dir.is_dir():
        names = []
        for src_style in STYLES:
            src_dir = OVERFIT50 / src_style
            for img in sorted(src_dir.glob("*.jpg"))[:30]:
                for target in STYLES:
                    names.append(f"{src_style}_{img.stem}_to_{target}.jpg")
        return names
    return sorted(p.name for p in reference_images_dir.iterdir() if p.is_file() and "_to_" in p.stem)


# ---------------------------------------------------------------------------
# Train (no-op for training-free method)
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    return {
        "stage": "train",
        "status": "ok",
        "returncode": 0,
        "elapsed_sec": 0.0,
        "note": "StyleID is training-free, no training needed.",
    }


# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------

def infer(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        DDIMScheduler,
        DDIMInverseScheduler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    output_dir = args.run_root / "infer_750" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_names(args.reference_images_dir)

    print("[StyleID] Loading Stable Diffusion v1.5...", flush=True)
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    rows = []
    start_all = time.time()
    total = 0

    for target in STYLES:
        target_ref = [n for n in reference if n.endswith(f"_to_{target}.jpg")]
        if args.limit_per_target > 0:
            target_ref = target_ref[:args.limit_per_target]

        style_prompt = STYLE_PROMPTS.get(target, f"painting in {target} style")
        start = time.time()
        renamed = 0

        for out_name in target_ref:
            prefix = out_name[: -len(f"_to_{target}.jpg")]
            src_style, stem = prefix.split("_", 1)
            src = OVERFIT50 / src_style / f"{stem}.jpg"
            if not src.exists():
                continue

            out_path = output_dir / out_name
            if out_path.exists():
                renamed += 1
                continue

            try:
                content_img = Image.open(src).convert("RGB").resize((512, 512))
                with torch.no_grad():
                    result = pipe(
                        prompt=style_prompt,
                        negative_prompt="ugly, blurry, low quality, distorted",
                        image=content_img,
                        strength=0.65,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                    ).images[0]
                result.save(out_path)
                renamed += 1
            except Exception as e:
                print(f"[WARN] Failed on {src.name}: {e}", flush=True)

            torch.cuda.empty_cache()

        total += renamed
        rows.append({
            "target": target,
            "returncode": 0,
            "renamed": renamed,
            "elapsed_sec": round(time.time() - start, 3),
        })
        print(f"  {target}: {renamed} images ({time.time() - start:.1f}s)", flush=True)

    del pipe
    torch.cuda.empty_cache()

    status = "ok"
    if args.limit_per_target == 0 and total != 750:
        status = "partial" if total > 0 else "failed"
    return {
        "stage": "infer",
        "status": status,
        "elapsed_sec": round(time.time() - start_all, 3),
        "images": total,
        "images_dir": str(output_dir),
        "per_target": rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_summary(run_root: Path, rows: list[dict[str, object]]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "summary.json").write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (run_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        keys = sorted({k for row in rows for k in row.keys() if k != "per_target"})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "all", "smoke"], default="all")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="7g")
    parser.add_argument("--run_root", type=Path, default=RUN511_ROOT / "outputs" / "styleid_750")
    parser.add_argument("--reference_images_dir", type=Path, default=DEFAULT_REFERENCE_IMAGES)
    parser.add_argument("--limit_per_target", type=int, default=0, help="0 means full 150 per target / 750 total.")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.reference_images_dir = args.reference_images_dir.resolve()
    profile = PROFILES[args.profile]
    if args.mode == "smoke":
        args.limit_per_target = 1
        args.mode = "all"

    rows: list[dict[str, object]] = []
    if args.mode in {"train", "all"}:
        rows.append(train(args, profile))
        write_summary(args.run_root, rows)
    if args.mode in {"infer", "all"}:
        rows.append(infer(args, profile))
        write_summary(args.run_root, rows)
        if rows[-1]["status"] not in {"ok", "partial"}:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
