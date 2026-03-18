import argparse
import subprocess
import sys
from pathlib import Path

import torch


def make_fake_ckpt(path: Path, test_dir: Path, style_subdirs: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "data": {"style_subdirs": list(style_subdirs)},
            "training": {"test_image_dir": str(test_dir)},
        }
    }
    torch.save(payload, path)


def _default_repo_root() -> Path:
    # Related_Works/run_full_eval_external.py -> <repo_root>/Related_Works/...
    return Path(__file__).resolve().parents[1]


def _auto_eval_python(repo_root: Path) -> Path | None:
    cand = repo_root / "Cycle-NCE" / ".venv" / "Scripts" / "python.exe"
    return cand if cand.exists() else None


def _auto_eval_script(repo_root: Path) -> Path | None:
    cand = repo_root / "Cycle-NCE" / "src" / "utils" / "run_evaluation.py"
    return cand if cand.exists() else None


def main():
    ap = argparse.ArgumentParser("Run Cycle-NCE run_evaluation.py on external generated images")
    ap.add_argument(
        "--eval_py",
        default=None,
        help="Python exe that can run Cycle-NCE evaluation (default: auto-detect Cycle-NCE/.venv)",
    )
    ap.add_argument(
        "--eval_script",
        default=None,
        help="Path to Cycle-NCE/src/utils/run_evaluation.py (default: auto-detect under repo root)",
    )
    ap.add_argument("--test_dir", required=True, help="Dataset dir with style subfolders")
    ap.add_argument("--out_dir", required=True, help="Directory containing images/ to reuse and write summary.json")
    ap.add_argument("--cache_dir", default="Cycle-NCE/src/eval_cache", help="Shared cache dir")
    ap.add_argument("--clip_allow_network", action="store_true")
    ap.add_argument(
        "--clip_backend",
        type=str,
        default="hf",
        choices=["openai", "hf", "none"],
        help="CLIP backend for eval metrics (default: hf to avoid openai-clip dependency).",
    )
    ap.add_argument(
        "--clip_openai_model",
        type=str,
        default="ViT-B/32",
        help="OpenAI CLIP model name when --clip_backend openai.",
    )
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_lpips_chunk_size", type=int, default=1)
    ap.add_argument("--max_ref_compare", type=int, default=50)
    ap.add_argument("--max_ref_cache", type=int, default=256)
    ap.add_argument("--max_src_samples", type=int, default=0, help="<=0 means all")
    ap.add_argument("--force_regen", action="store_true", help="Overwrite metrics")
    ap.add_argument("--enable_kid", action="store_true", help="Compute KID (torchmetrics) in full_eval summary.json")
    ap.add_argument("--kid_subset_size", type=int, default=50)
    ap.add_argument("--kid_max_gen", type=int, default=200)
    ap.add_argument("--kid_max_ref", type=int, default=200)
    ap.add_argument("--kid_batch_size", type=int, default=8)
    args = ap.parse_args()

    repo_root = _default_repo_root()
    eval_py_path = Path(args.eval_py).resolve() if args.eval_py else _auto_eval_python(repo_root)
    eval_script = Path(args.eval_script).resolve() if args.eval_script else _auto_eval_script(repo_root)
    if eval_py_path is None or not eval_py_path.exists():
        raise FileNotFoundError(
            "Missing --eval_py and could not auto-detect Cycle-NCE/.venv python. "
            f"Expected: {repo_root / 'Cycle-NCE' / '.venv' / 'Scripts' / 'python.exe'}"
        )
    if eval_script is None or not eval_script.exists():
        raise FileNotFoundError(
            "Missing --eval_script and could not auto-detect Cycle-NCE evaluation script. "
            f"Expected: {repo_root / 'Cycle-NCE' / 'src' / 'utils' / 'run_evaluation.py'}"
        )

    eval_py = str(eval_py_path)
    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    images_dir = out_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    style_subdirs = sorted([d.name for d in test_dir.iterdir() if d.is_dir()], key=lambda x: x.lower())
    if not style_subdirs:
        raise RuntimeError(f"No style subdirs found under: {test_dir}")

    fake_ckpt = out_dir / "fake_eval_checkpoint.pt"
    make_fake_ckpt(fake_ckpt, test_dir, style_subdirs)

    cmd = [
        eval_py,
        str(eval_script),
        "--checkpoint",
        str(fake_ckpt),
        "--output",
        str(out_dir),
        "--test_dir",
        str(test_dir),
        "--cache_dir",
        str(Path(args.cache_dir).resolve()),
        "--reuse_generated",
        "--batch_size",
        str(int(args.batch_size)),
        "--eval_lpips_chunk_size",
        str(int(args.eval_lpips_chunk_size)),
        "--max_ref_compare",
        str(int(args.max_ref_compare)),
        "--max_ref_cache",
        str(int(args.max_ref_cache)),
        "--max_src_samples",
        str(int(args.max_src_samples)),
    ]
    if args.clip_allow_network:
        cmd += ["--clip_allow_network"]
    cmd += ["--clip_backend", str(args.clip_backend)]
    if str(args.clip_backend).lower() == "openai":
        cmd += ["--clip_openai_model", str(args.clip_openai_model)]
    if args.force_regen:
        cmd += ["--force_regen"]
    if args.enable_kid:
        cmd += [
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

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
