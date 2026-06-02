from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml
from PIL import Image, ImageOps


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SAMST_REPO = WORKSPACE_ROOT / "Related_Works" / "repos" / "SaMST-main"
DEFAULT_IMAGE_ROOT = WORKSPACE_ROOT / "Dataset" / "distinct5_512" / "test"
DEFAULT_CKPT_ROOT = (
    WORKSPACE_ROOT
    / "Related_Works"
    / "baseline_pipeline"
    / "results"
    / "samst_distinct5_512_real_b2_e15_20260602"
    / "checkpoints"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "Related_Works"
    / "baseline_pipeline"
    / "results"
    / "samst_distinct5_512_real_b2_e15_20260602"
    / "eval_epoch15"
)
DEFAULT_STYLES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _list_images(folder: Path, limit: int) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    return files[:limit] if limit > 0 else files


def _prepare_content_dir(image_root: Path, style_names: list[str], max_src_per_style: int, resize_content: int) -> Path:
    content_dir = SAMST_REPO / "content"
    shutil.rmtree(content_dir, ignore_errors=True)
    content_dir.mkdir(parents=True, exist_ok=True)

    for style in style_names:
        src_dir = image_root / style
        if not src_dir.is_dir():
            raise FileNotFoundError(f"missing source style dir: {src_dir}")
        files = _list_images(src_dir, max_src_per_style)
        for src in files:
            dst = content_dir / f"{style}_{src.stem}.png"
            with Image.open(src) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                if resize_content > 0:
                    image = ImageOps.fit(
                        image,
                        (resize_content, resize_content),
                        method=Image.Resampling.LANCZOS,
                        centering=(0.5, 0.5),
                    )
                image.save(dst)
    return content_dir


def _run_one_target(
    *,
    target_style: str,
    epoch: int,
    image_root: Path,
    style_names: list[str],
    max_src_per_style: int,
    resize_content: int,
    ckpt_root: Path,
    images_out: Path,
) -> int:
    ckpt = ckpt_root / target_style / f"epoch_{epoch}.model"
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
        "content_scale": None,
        "output_image_dir": str(raw_output) + "/",
        "model": str(ckpt),
        "style_num": 1,
        "cuda": 1,
    }
    with (test_dir / "test.yml").open("w", encoding="utf-8") as f:
        yaml.dump(test_yml, f, default_flow_style=False, allow_unicode=True)

    print(f"[SaMST distinct5] epoch={epoch:04d} target={target_style} ckpt={ckpt.name}", flush=True)
    rc = subprocess.run([sys.executable, str(test_script)], cwd=str(test_dir)).returncode
    if rc != 0:
        return rc

    copied = 0
    raw_files = sorted(p for p in raw_output.iterdir() if p.is_file() and p.name.startswith("style1_"))
    for src in raw_files:
        original = src.name[len("style1_") :]
        stem = Path(original).stem
        dst = images_out / f"{stem}_to_{target_style}.png"
        with Image.open(src) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(dst)
        copied += 1
    print(f"[SaMST distinct5] epoch={epoch:04d} target={target_style} copied={copied}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--style-names", default=",".join(DEFAULT_STYLES))
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--resize-content", type=int, default=512)
    args = parser.parse_args()

    style_names = [p.strip() for p in str(args.style_names).split(",") if p.strip()]
    if not style_names:
        raise ValueError("--style-names is empty")

    image_root = args.image_root.resolve()
    ckpt_root = args.ckpt_root.resolve()
    output_root = args.output_root.resolve()
    step_dir = output_root / f"epoch_{int(args.epoch):04d}"
    images_out = step_dir / "images"
    shutil.rmtree(images_out, ignore_errors=True)
    images_out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    for target in style_names:
        rc = _run_one_target(
            target_style=target,
            epoch=int(args.epoch),
            image_root=image_root,
            style_names=style_names,
            max_src_per_style=int(args.max_src_per_style),
            resize_content=max(0, int(args.resize_content)),
            ckpt_root=ckpt_root,
            images_out=images_out,
        )
        if rc != 0:
            raise RuntimeError(f"SaMST inference failed for epoch={args.epoch}, target={target}, rc={rc}")

    summary = {
        "epoch": int(args.epoch),
        "images": len(list(images_out.glob("*_to_*.*"))),
        "style_names": style_names,
        "max_src_per_style": int(args.max_src_per_style),
        "elapsed_sec": round(time.time() - started, 3),
        "images_dir": str(images_out),
    }
    with (step_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SaMST distinct5] total={summary['images']} -> {images_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
