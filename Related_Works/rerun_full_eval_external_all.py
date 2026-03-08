from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import re


_GEN_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_NAME_RE = re.compile(r"^(?P<src>[^_]+)_.+_to_(?P<tgt>.+)$")


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


def _infer_styles_from_images(images_dir: Path) -> list[str]:
    styles: set[str] = set()
    try:
        for p in images_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in _GEN_EXTS:
                continue
            m = _NAME_RE.match(p.stem)
            if not m:
                continue
            styles.add(m.group("src"))
            styles.add(m.group("tgt"))
    except Exception:
        return []
    return sorted(styles, key=lambda x: x.lower())


def _pick_test_dir_for_styles(test_dir_candidates: list[Path], styles: list[str]) -> Path | None:
    for cand in test_dir_candidates:
        if not cand.exists() or not cand.is_dir():
            continue
        ok = True
        for s in styles:
            if not (cand / s).is_dir():
                ok = False
                break
        if ok:
            return cand
    return None


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
    ap.add_argument(
        "--test_dir",
        default=None,
        help="Dataset dir with style subfolders. If omitted, pick from --test_dir_candidate based on each run's inferred styles.",
    )
    ap.add_argument(
        "--test_dir_candidate",
        action="append",
        default=[],
        help="Candidate dataset dirs (repeatable). First match that contains all inferred styles will be used.",
    )
    ap.add_argument("--cache_dir", default="Cycle-NCE/src/eval_cache", help="Shared cache dir")
    ap.add_argument("--clip_allow_network", action="store_true")
    ap.add_argument(
        "--clip_backend",
        default="openai",
        choices=["openai", "hf", "none"],
        help="CLIP backend forwarded to Cycle-NCE run_evaluation.py",
    )
    ap.add_argument("--clip_openai_model", default="ViT-B/32", help="OpenAI CLIP model name (if clip_backend=openai)")
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
    cache_dir = str(Path(args.cache_dir).resolve())

    test_dir_candidates = [Path(p).resolve() for p in (args.test_dir_candidate or [])]
    fixed_test_dir = Path(args.test_dir).resolve() if args.test_dir else None

    out_dirs = _iter_image_output_dirs(scan_root, include_non_matching=bool(args.include_non_matching))
    if not out_dirs:
        raise SystemExit(f"No images/ dirs found under: {scan_root}")

    for out_dir in out_dirs:
        images_dir = out_dir / "images"
        if not images_dir.is_dir():
            continue

        fake_ckpt = out_dir / "fake_eval_checkpoint.pt"
        ckpt_test_dir: Path | None = None
        ckpt_style_subdirs: list[str] | None = None
        if fake_ckpt.exists() and fixed_test_dir is None:
            try:
                import torch

                payload = torch.load(fake_ckpt, map_location="cpu")
                cfg = (payload or {}).get("config", {})
                ckpt_test_dir_raw = (cfg.get("training", {}) or {}).get("test_image_dir", None)
                if ckpt_test_dir_raw:
                    p = Path(str(ckpt_test_dir_raw)).expanduser()
                    if not p.is_absolute():
                        p = (out_dir / p).resolve()
                    ckpt_test_dir = p if p.exists() else None
                ckpt_style_subdirs = (cfg.get("data", {}) or {}).get("style_subdirs", None)
                if isinstance(ckpt_style_subdirs, list):
                    ckpt_style_subdirs = [str(s) for s in ckpt_style_subdirs if str(s)]
                else:
                    ckpt_style_subdirs = None
            except Exception:
                ckpt_test_dir = None
                ckpt_style_subdirs = None

        inferred_styles = _infer_styles_from_images(images_dir)
        if fixed_test_dir is not None:
            picked_test_dir = fixed_test_dir
        else:
            if ckpt_test_dir is not None:
                picked_test_dir = ckpt_test_dir
            else:
                picked_test_dir = None

            if picked_test_dir is None:
                picked_test_dir = (
                    _pick_test_dir_for_styles(test_dir_candidates, inferred_styles) if inferred_styles else None
                )
            if picked_test_dir is None:
                picked_test_dir = next((p for p in test_dir_candidates if p.exists() and p.is_dir()), None)
        if picked_test_dir is None:
            print(f"[full_eval] skip (no test_dir match): {out_dir}")
            continue

        test_dir = str(picked_test_dir)
        style_subdirs = ckpt_style_subdirs or inferred_styles
        if not style_subdirs:
            style_subdirs = sorted([d.name for d in picked_test_dir.iterdir() if d.is_dir()], key=lambda x: x.lower())
        cmd = [
            eval_py,
            eval_script,
            "--checkpoint",
            str((out_dir / "fake_eval_checkpoint.pt").resolve()),
            "--output",
            str(out_dir),
            "--test_dir",
            test_dir,
            "--cache_dir",
            cache_dir,
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
        if args.clip_allow_network:
            cmd += ["--clip_allow_network"]
        cmd += ["--clip_backend", str(args.clip_backend)]
        if str(args.clip_backend).lower() == "openai":
            cmd += ["--clip_openai_model", str(args.clip_openai_model)]

        # Ensure fake checkpoint exists by delegating to run_full_eval_external.py-like behavior:
        # if it's missing, create it using torch.save with minimal config payload.
        if not fake_ckpt.exists():
            import torch
            payload = {
                "config": {
                    "data": {"style_subdirs": list(style_subdirs)},
                    "training": {"test_image_dir": str(test_dir)},
                }
            }
            torch.save(payload, fake_ckpt)

        print(f"[full_eval] {out_dir} (test_dir={picked_test_dir.name}, styles={style_subdirs})")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
