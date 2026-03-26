import argparse
import subprocess
from pathlib import Path
import shutil

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


def _auto_eval_workdir(repo_root: Path) -> Path | None:
    cand = repo_root / "Cycle-NCE" / "src"
    return cand if cand.exists() else None


def _auto_eval_script(repo_root: Path) -> Path | None:
    cand = repo_root / "Cycle-NCE" / "src" / "utils" / "run_evaluation.py"
    return cand if cand.exists() else None


def main():
    ap = argparse.ArgumentParser("Run Cycle-NCE run_evaluation.py on external generated images")
    ap.add_argument(
        "--eval_mode",
        default="auto",
        choices=["auto", "python", "uv"],
        help="How to launch Cycle-NCE evaluation: direct python, uv run from Cycle-NCE/src, or auto-detect.",
    )
    ap.add_argument(
        "--eval_py",
        default=None,
        help="Python exe that can run Cycle-NCE evaluation (default: auto-detect Cycle-NCE/.venv)",
    )
    ap.add_argument(
        "--eval_workdir",
        default=None,
        help="Working directory for `uv run` evaluation (default: auto-detect Cycle-NCE/src).",
    )
    ap.add_argument(
        "--eval_script",
        default=None,
        help="Path to run_evaluation.py. For uv mode, relative paths are resolved from eval_workdir.",
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
    auto_eval_py = _auto_eval_python(repo_root)
    auto_eval_workdir = _auto_eval_workdir(repo_root)
    auto_eval_script = _auto_eval_script(repo_root)

    eval_mode = args.eval_mode
    if eval_mode == "auto":
        if args.eval_py:
            eval_mode = "python"
        elif args.eval_workdir:
            eval_mode = "uv"
        elif shutil.which("uv") and auto_eval_workdir and auto_eval_script:
            eval_mode = "uv"
        elif auto_eval_py and auto_eval_script:
            eval_mode = "python"
        else:
            raise FileNotFoundError(
                "Could not auto-detect a Cycle-NCE evaluation runner. "
                "Need either uv + Cycle-NCE/src or Cycle-NCE/.venv/Scripts/python.exe."
            )

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

    if eval_mode == "python":
        eval_py_path = Path(args.eval_py).resolve() if args.eval_py else auto_eval_py
        eval_script_path = Path(args.eval_script).resolve() if args.eval_script else auto_eval_script
        if eval_py_path is None or not eval_py_path.exists():
            raise FileNotFoundError(
                "Missing --eval_py and could not auto-detect Cycle-NCE/.venv python. "
                f"Expected: {repo_root / 'Cycle-NCE' / '.venv' / 'Scripts' / 'python.exe'}"
            )
        if eval_script_path is None or not eval_script_path.exists():
            raise FileNotFoundError(
                "Missing --eval_script and could not auto-detect Cycle-NCE evaluation script. "
                f"Expected: {repo_root / 'Cycle-NCE' / 'src' / 'utils' / 'run_evaluation.py'}"
            )
        cmd = [str(eval_py_path), str(eval_script_path)]
        cmd_cwd = None
        cache_dir = str(Path(args.cache_dir).resolve())
    elif eval_mode == "uv":
        if shutil.which("uv") is None:
            raise FileNotFoundError("`uv` was not found on PATH, but --eval_mode uv was requested.")
        eval_workdir = Path(args.eval_workdir).resolve() if args.eval_workdir else auto_eval_workdir
        if eval_workdir is None or not eval_workdir.exists():
            raise FileNotFoundError(
                "Missing --eval_workdir and could not auto-detect Cycle-NCE/src. "
                f"Expected: {repo_root / 'Cycle-NCE' / 'src'}"
            )
        if args.eval_script:
            eval_script_path = Path(args.eval_script)
            if eval_script_path.is_absolute():
                resolved_eval_script = eval_script_path
                cmd_eval_script = str(eval_script_path)
            else:
                resolved_eval_script = (eval_workdir / eval_script_path).resolve()
                cmd_eval_script = str(eval_script_path)
        else:
            resolved_eval_script = auto_eval_script
            cmd_eval_script = str(Path("utils") / "run_evaluation.py")
        if resolved_eval_script is None or not resolved_eval_script.exists():
            raise FileNotFoundError(
                "Missing --eval_script and could not auto-detect Cycle-NCE evaluation script. "
                f"Expected: {repo_root / 'Cycle-NCE' / 'src' / 'utils' / 'run_evaluation.py'}"
            )
        cmd = ["uv", "run", cmd_eval_script]
        cmd_cwd = str(eval_workdir)
        if Path(args.cache_dir).is_absolute():
            cache_dir = str(Path(args.cache_dir).resolve())
        else:
            cache_dir = str((eval_workdir / args.cache_dir).resolve())
    else:
        raise ValueError(f"Unsupported eval mode: {eval_mode}")

    cmd += [
        "--checkpoint",
        str(fake_ckpt),
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
    subprocess.run(cmd, check=True, cwd=cmd_cwd)


if __name__ == "__main__":
    main()
