from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


_GEN_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _count_generated(images_dir: Path) -> int:
    n = 0
    try:
        for p in images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in _GEN_EXTS and "_to_" in p.stem:
                n += 1
    except Exception:
        return 0
    return n


def _resolve_ckpt_from_summary(out_dir: Path) -> Path | None:
    summary = _read_json(out_dir / "summary.json")
    if not summary:
        return None
    ckpt = summary.get("checkpoint")
    if not ckpt or not isinstance(ckpt, str):
        return None
    p = Path(ckpt)
    if not p.is_absolute():
        p = (out_dir / p).resolve()
    return p if p.exists() else None


def _resolve_test_dir_from_ckpt(ckpt_path: Path) -> Path | None:
    try:
        import torch

        payload = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(payload, dict):
            return None
        cfg = payload.get("config") or {}
        if not isinstance(cfg, dict):
            return None
        raw = (cfg.get("training") or {}).get("test_image_dir", None)
        if not raw:
            return None
        p = Path(str(raw)).expanduser()
        return p if p.exists() else None
    except Exception:
        return None


def _infer_styles_from_images(images_dir: Path) -> list[str]:
    styles: set[str] = set()
    try:
        for p in images_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in _GEN_EXTS:
                continue
            stem = p.stem
            if "_to_" not in stem:
                continue
            left, right = stem.split("_to_", 1)
            src = left.split("_", 1)[0]
            tgt = right
            if src:
                styles.add(src)
            if tgt:
                styles.add(tgt)
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


def _ensure_fake_ckpt(fake_ckpt: Path, *, test_dir: Path, style_subdirs: list[str]) -> None:
    import torch

    fake_ckpt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "data": {"style_subdirs": list(style_subdirs)},
            "training": {"test_image_dir": str(test_dir)},
        }
    }
    torch.save(payload, fake_ckpt)


def _iter_out_dirs(scan_root: Path) -> list[Path]:
    out = []
    for images_dir in sorted(scan_root.rglob("images")):
        if images_dir.is_dir():
            out.append(images_dir.parent)
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Re-run Cycle-NCE full_eval (reuse only) for all images/ dirs under one or more roots.")
    ap.add_argument("--scan_root", action="append", required=True, help="Root(s) to scan (repeatable)")
    ap.add_argument(
        "--experiments_root",
        action="append",
        default=[],
        help="Treat each direct child dir as one experiment; experiments without any images/ dirs are skipped.",
    )
    ap.add_argument("--eval_py", required=True)
    ap.add_argument("--eval_script", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--test_dir_candidate", action="append", default=[], help="Candidate dataset dirs (repeatable)")
    ap.add_argument("--force_regen", action="store_true")
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
    ap.add_argument("--kid_subset_size", type=int, default=50)
    ap.add_argument("--kid_max_gen", type=int, default=50)
    ap.add_argument("--kid_max_ref", type=int, default=50)
    ap.add_argument("--kid_batch_size", type=int, default=16)
    ap.add_argument("--stop_on_error", action="store_true")
    args = ap.parse_args()

    eval_py = str(Path(args.eval_py).resolve())
    eval_script = str(Path(args.eval_script).resolve())
    cache_dir = str(Path(args.cache_dir).resolve())
    test_dir_candidates = [Path(p).resolve() for p in (args.test_dir_candidate or [])]

    scan_roots = [Path(p).resolve() for p in args.scan_root]
    out_dirs: list[Path] = []
    for r in scan_roots:
        out_dirs.extend(_iter_out_dirs(r))

    for exp_root_raw in args.experiments_root or []:
        exp_root = Path(exp_root_raw).resolve()
        if not exp_root.exists() or not exp_root.is_dir():
            print(f"[skip_root] experiments_root missing: {exp_root}")
            continue
        exp_dirs = sorted([d for d in exp_root.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
        for exp_dir in exp_dirs:
            exp_out_dirs = _iter_out_dirs(exp_dir)
            if not exp_out_dirs:
                print(f"[skip_exp] {exp_dir} (no images/ dirs)")
                continue
            out_dirs.extend(exp_out_dirs)
    # De-dup
    seen = set()
    uniq = []
    for p in out_dirs:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    out_dirs = uniq

    print(f"scan_roots: {[str(r) for r in scan_roots]}")
    print(f"out_dirs: {len(out_dirs)}")

    for out_dir in out_dirs:
        images_dir = out_dir / "images"
        if not images_dir.is_dir():
            continue
        n_gen = _count_generated(images_dir)
        if n_gen <= 0:
            print(f"[skip] {out_dir} (no *_to_* images)")
            continue

        ckpt = _resolve_ckpt_from_summary(out_dir)
        if ckpt is None:
            fake_ckpt = out_dir / "fake_eval_checkpoint.pt"
            if fake_ckpt.exists():
                ckpt = fake_ckpt

        inferred_styles = _infer_styles_from_images(images_dir)
        test_dir = _resolve_test_dir_from_ckpt(ckpt) if ckpt is not None else None
        if test_dir is None:
            test_dir = _pick_test_dir_for_styles(test_dir_candidates, inferred_styles) if inferred_styles else None
        if test_dir is None:
            test_dir = next((p for p in test_dir_candidates if p.exists() and p.is_dir()), None)
        if test_dir is None:
            print(f"[skip] {out_dir} (no test_dir)")
            continue

        if ckpt is None:
            fake_ckpt = out_dir / "fake_eval_checkpoint.pt"
            style_subdirs = inferred_styles or sorted([d.name for d in test_dir.iterdir() if d.is_dir()], key=str.lower)
            _ensure_fake_ckpt(fake_ckpt, test_dir=test_dir, style_subdirs=style_subdirs)
            ckpt = fake_ckpt

        cmd = [
            eval_py,
            eval_script,
            "--checkpoint",
            str(ckpt),
            "--output",
            str(out_dir),
            "--test_dir",
            str(test_dir),
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

        print(f"[full_eval] {out_dir} (n_images={n_gen}, test_dir={test_dir.name}, ckpt={ckpt.name})")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[error] {out_dir}: exit={e.returncode}")
            if args.stop_on_error:
                raise


if __name__ == "__main__":
    main()
