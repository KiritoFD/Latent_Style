"""Generate SaMST WikiArt512 checkpoint curves for SB reuse evaluation.

This script only runs SaMST inference. Metrics are intentionally delegated to
SchrodingerBridge/src/utils/run_evaluation.py so SaMST uses the same CLIP/LPIPS
protocol as the main experiments.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import yaml
from PIL import Image, ImageOps


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SAMST_REPO = WORKSPACE_ROOT / "Related_Works" / "repos" / "external" / "SaMST"
DEFAULT_CKPT_ROOT = SAMST_REPO / "checkpoint"
DEFAULT_IMAGE_ROOT = Path("F:/wikiart_images_512_ema_test")
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "Related_Works"
    / "baseline_pipeline"
    / "results"
    / "samst_wikiart512_curve"
)
DEFAULT_STYLES = [
    "Realism",
    "Impressionism",
    "Post_Impressionism",
    "Expressionism",
    "Symbolism",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _parse_steps(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("--steps is empty")
    return out


def _parse_styles(value: str) -> list[str]:
    out = [p.strip() for p in value.split(",") if p.strip()]
    if not out:
        raise ValueError("--style-names is empty")
    return out


def _target_ckpt_dir(ckpt_root: Path, target_style: str) -> Path:
    return ckpt_root / f"wikiart5_3600_target_{target_style}_b2_e15"


def _safe_stem(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii")
    out = []
    prev_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch.lower())
            prev_sep = False
        else:
            if not prev_sep:
                out.append("-")
                prev_sep = True
    slug = "".join(out).strip("-")
    return slug or "img"


def _prepare_content_dir(image_root: Path, style_names: list[str], max_src_per_style: int, resize_content: int) -> Path:
    content_dir = SAMST_REPO / "content"
    shutil.rmtree(content_dir, ignore_errors=True)
    content_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    for style in style_names:
        src_dir = image_root / style
        if not src_dir.is_dir():
            raise FileNotFoundError(f"missing source style dir: {src_dir}")
        files = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if max_src_per_style > 0:
            files = files[:max_src_per_style]
        for idx, src in enumerate(files, start=1):
            base = f"{style}_{idx:03d}_{_safe_stem(src.stem)}"
            candidate = base
            suffix = 2
            while candidate in used_names:
                candidate = f"{base}-{suffix}"
                suffix += 1
            used_names.add(candidate)
            dst = content_dir / f"{candidate}.png"
            if resize_content > 0:
                with Image.open(src) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image = ImageOps.fit(
                        image,
                        (resize_content, resize_content),
                        method=Image.Resampling.LANCZOS,
                        centering=(0.5, 0.5),
                    )
                    image.save(dst)
            else:
                with Image.open(src) as image:
                    ImageOps.exif_transpose(image).convert("RGB").save(dst)
    return content_dir


def _run_one_target(
    *,
    target_style: str,
    epoch: int,
    image_root: Path,
    style_names: list[str],
    max_src_per_style: int,
    content_scale: int | None,
    resize_content: int,
    ckpt_root: Path,
    images_out: Path,
) -> int:
    ckpt = _target_ckpt_dir(ckpt_root, target_style) / f"epoch_{epoch}.model"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")

    test_dir = SAMST_REPO / "test_model" / "test"
    test_script = test_dir / "test.py"
    if not test_script.is_file():
        raise FileNotFoundError(f"missing SaMST test.py: {test_script}")

    content_dir = _prepare_content_dir(image_root, style_names, max_src_per_style, resize_content)
    raw_output = SAMST_REPO / "outputs"
    shutil.rmtree(raw_output, ignore_errors=True)
    raw_output.mkdir(parents=True, exist_ok=True)

    test_yml = {
        "content_image_dir": str(content_dir),
        "content_scale": content_scale,
        "output_image_dir": str(raw_output) + "/",
        "model": str(ckpt),
        "style_num": 1,
        "cuda": 1,
    }
    with (test_dir / "test.yml").open("w", encoding="utf-8") as f:
        yaml.dump(test_yml, f, default_flow_style=False, allow_unicode=True)

    print(f"[SaMST] epoch={epoch:04d} target={target_style} ckpt={ckpt.name}", flush=True)
    rc = subprocess.run([sys.executable, str(test_script)], cwd=str(test_dir)).returncode
    if rc != 0:
        return rc

    copied = 0
    for src in sorted(raw_output.glob("style1_*")):
        if not src.is_file():
            continue
        original = src.name[len("style1_") :]
        stem = Path(original).stem
        dst = images_out / f"{stem}_to_{target_style}.png"
        with Image.open(src) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(dst)
        copied += 1
    print(f"[SaMST] epoch={epoch:04d} target={target_style} copied={copied}", flush=True)
    return 0


def generate_epoch(args: argparse.Namespace, epoch: int, style_names: list[str]) -> Path:
    step_dir = args.output_root / f"epoch_{epoch:04d}"
    images_out = step_dir / "images"
    shutil.rmtree(images_out, ignore_errors=True)
    images_out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    for target in style_names:
        rc = _run_one_target(
            target_style=target,
            epoch=epoch,
            image_root=args.image_root,
            style_names=style_names,
            max_src_per_style=args.max_src_per_style,
            content_scale=args.content_scale if int(args.content_scale) > 0 else None,
            resize_content=max(0, int(args.resize_content)),
            ckpt_root=args.ckpt_root,
            images_out=images_out,
        )
        if rc != 0:
            raise RuntimeError(f"SaMST inference failed for epoch={epoch}, target={target}, rc={rc}")

    count = len(list(images_out.glob("*_to_*.*")))
    summary = {
        "epoch": epoch,
        "images": count,
        "style_names": style_names,
        "max_src_per_style": args.max_src_per_style,
        "elapsed_sec": round(time.time() - started, 3),
        "images_dir": str(images_out),
    }
    with (step_dir / "summary.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SaMST] epoch={epoch:04d} total={count} -> {images_out}", flush=True)
    return step_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True, help="Comma-separated SaMST epochs, e.g. 15,20,25,30")
    parser.add_argument("--style-names", default=",".join(DEFAULT_STYLES))
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--content-scale", type=int, default=1, help="SaMST scale divisor; <=0 keeps original size in SaMST loader.")
    parser.add_argument("--resize-content", type=int, default=512, help="Pre-resize copied content images to this square size; <=0 copies originals.")
    args = parser.parse_args()

    args.image_root = args.image_root.resolve()
    args.ckpt_root = args.ckpt_root.resolve()
    args.output_root = args.output_root.resolve()
    style_names = _parse_styles(args.style_names)
    steps = _parse_steps(args.steps)

    for epoch in steps:
        generate_epoch(args, epoch, style_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
