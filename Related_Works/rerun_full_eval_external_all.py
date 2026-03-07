from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


_GEN_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _looks_like_generated_images_dir(images_dir: Path) -> bool:
    """
    External full_eval expects generated images to follow:
      {src_style}_{src_stem}_to_{tgt_style}.(jpg|png|...)
    Avoid running eval on unrelated folders like CUT 'web/images'.
    """
    try:
        for p in images_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in _GEN_EXTS:
                continue
            if "_to_" in p.stem:
                return True
    except Exception:
        return False
    return False


def _iter_image_output_dirs(scan_root: Path, *, include_non_matching: bool) -> list[Path]:
    out: list[Path] = []
    for images_dir in sorted(scan_root.rglob("images")):
        if not images_dir.is_dir():
            continue
        if not include_non_matching and not _looks_like_generated_images_dir(images_dir):
            continue
        out_dir = images_dir.parent
        out.append(out_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Re-run Cycle-NCE full_eval for all external image dirs under a root (KID enabled).")
    ap.add_argument("--scan_root", required=True, help="Root to scan (e.g. Related_Works/runs)")
    ap.add_argument(
        "--include_non_matching",
        action="store_true",
        help="Also include images/ dirs that do not look like external generated outputs (not recommended).",
    )
    ap.add_argument("--eval_py", required=True, help="Python exe that can run Cycle-NCE evaluation")
    ap.add_argument("--eval_script", required=True, help="Path to Cycle-NCE/src/utils/run_evaluation.py")
    ap.add_argument("--test_dir", required=True, help="Dataset dir with style subfolders")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_lpips_chunk_size", type=int, default=1)
    ap.add_argument("--max_ref_compare", type=int, default=50)
    ap.add_argument("--max_ref_cache", type=int, default=256)
    ap.add_argument("--force_regen", action="store_true")
    ap.add_argument("--kid_subset_size", type=int, default=50)
    ap.add_argument("--kid_max_gen", type=int, default=200)
    ap.add_argument("--kid_max_ref", type=int, default=200)
    ap.add_argument("--kid_batch_size", type=int, default=8)
    args = ap.parse_args()

    scan_root = Path(args.scan_root).resolve()
    eval_py = str(Path(args.eval_py).resolve())
    eval_script = str(Path(args.eval_script).resolve())
    test_dir = str(Path(args.test_dir).resolve())

    out_dirs = _iter_image_output_dirs(scan_root, include_non_matching=bool(args.include_non_matching))
    if not out_dirs:
        raise SystemExit(f"No images/ dirs found under: {scan_root}")

    for out_dir in out_dirs:
        images_dir = out_dir / "images"
        if not images_dir.is_dir():
            continue
        cmd = [
            eval_py,
            eval_script,
            "--checkpoint",
            str((out_dir / "fake_eval_checkpoint.pt").resolve()),
            "--output",
            str(out_dir),
            "--test_dir",
            test_dir,
            "--reuse_generated",
            "--batch_size",
            str(int(args.batch_size)),
            "--eval_lpips_chunk_size",
            str(int(args.eval_lpips_chunk_size)),
            "--max_ref_compare",
            str(int(args.max_ref_compare)),
            "--max_ref_cache",
            str(int(args.max_ref_cache)),
            "--eval_enable_kid",
            "--eval_kid_subset_size",
            str(int(args.kid_subset_size)),
            "--eval_kid_max_gen",
            str(int(args.kid_max_gen)),
            "--eval_kid_max_ref",
            str(int(args.kid_max_ref)),
            "--eval_kid_batch_size",
            str(int(args.kid_batch_size)),
        ]
        if args.force_regen:
            cmd += ["--force_regen"]

        # Ensure fake checkpoint exists by delegating to run_full_eval_external.py-like behavior:
        # if it's missing, create it using torch.save with minimal config payload.
        fake_ckpt = out_dir / "fake_eval_checkpoint.pt"
        if not fake_ckpt.exists():
            import torch

            style_subdirs = sorted([d.name for d in Path(test_dir).iterdir() if d.is_dir()], key=lambda x: x.lower())
            payload = {
                "config": {
                    "data": {"style_subdirs": list(style_subdirs)},
                    "training": {"test_image_dir": str(test_dir)},
                }
            }
            torch.save(payload, fake_ckpt)

        print(f"[full_eval] {out_dir}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
