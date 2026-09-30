from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_AAAI_DIR = SB_ROOT / "aaai2027" / "dualpath_bestfew_localreview_20260609"
SOURCE_ROOT = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test")
INTROSTYLE_MODEL_ID = Path(r"G:\GitHub\Latent_Style\eval_cache\modelscope\stabilityai\stable-diffusion-2-1-base")


def _run(cmd: list[str]) -> int:
    print("[run_local_dualpath_bestfew_review] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    src_path = str(SB_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    proc = subprocess.run(cmd, check=False, cwd=str(SB_ROOT), env=env)
    return int(proc.returncode)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _images_present(handoff_csv: Path) -> bool:
    for row in _read_rows(handoff_csv):
        images_dir = Path(str(row.get("images_dir", "")).strip())
        if not images_dir.is_dir():
            return False
        if not any(images_dir.iterdir()):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull dualpath best-few locally and run local IntroStyle/DINO review.")
    parser.add_argument("--eval-subdir", default="full_eval_fresh_localreview")
    parser.add_argument("--local-root", type=Path, default=DEFAULT_AAAI_DIR)
    parser.add_argument("--skip-introstyle", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--introstyle-device", default="")
    parser.add_argument("--dino-device", default="")
    parser.add_argument("--dino-local-files-only", action="store_true")
    args = parser.parse_args()

    eval_subdir = str(args.eval_subdir).strip() or "full_eval_fresh_localreview"
    aaai_dir = Path(args.local_root)
    aaai_dir.mkdir(parents=True, exist_ok=True)
    handoff_csv = aaai_dir / f"{eval_subdir}_bestfew_handoff.csv"
    manifest_csv = aaai_dir / f"{eval_subdir}_bestfew_introstyle_manifest.csv"
    intro_csv = aaai_dir / f"{eval_subdir}_bestfew_introstyle.csv"
    intro_json = aaai_dir / f"{eval_subdir}_bestfew_introstyle.json"
    dino_csv = aaai_dir / f"{eval_subdir}_bestfew_dino.csv"

    if not bool(args.skip_pull):
        pull = [
            sys.executable,
            str(SCRIPT_DIR / "pull_remote_dualpath_bestfew_localreview.py"),
            "--eval-subdir",
            eval_subdir,
            "--local-root",
            str(aaai_dir),
        ]
        rc = _run(pull)
        if rc != 0:
            return rc

    manifest = [
        sys.executable,
        str(SCRIPT_DIR / "build_introstyle_manifest_from_handoff.py"),
        "--handoff-csv",
        str(handoff_csv),
        "--output-csv",
        str(manifest_csv),
        "--method",
        "LBM",
        "--label-prefix",
        "DualPathBestFew",
        "--source-root",
        str(SOURCE_ROOT),
    ]
    rc = _run(manifest)
    if rc != 0:
        return rc

    if not _images_present(handoff_csv):
        print(
            "[run_local_dualpath_bestfew_review] images are absent in the pulled eval dirs; "
            "skip local IntroStyle/DINO until an image-backed eval bundle is available.",
            flush=True,
        )
        print(manifest_csv)
        return 0

    intro_device = str(args.introstyle_device).strip() or ("cpu" if bool(args.cpu_only) else "cuda")
    dino_device = str(args.dino_device).strip() or ("cpu" if bool(args.cpu_only) else "cuda")

    if not bool(args.skip_introstyle):
        intro = [
            sys.executable,
            str(SB_ROOT / "tools" / "eval_introstyle_probe.py"),
            "--manifest",
            str(manifest_csv),
            "--style-bank-root",
            str(SOURCE_ROOT),
            "--output_csv",
            str(intro_csv),
            "--output_json",
            str(intro_json),
            "--model-id",
            str(INTROSTYLE_MODEL_ID),
            "--device",
            intro_device,
            "--batch_size",
            "1",
            "--ensemble_size",
            "1",
        ]
        rc = _run(intro)
        if rc != 0:
            return rc

    if not bool(args.skip_dino):
        dino = [
            sys.executable,
            str(SB_ROOT / "tools" / "eval_dino_manifest.py"),
            "--manifest",
            str(manifest_csv),
            "--source-test-dir",
            str(SOURCE_ROOT),
            "--output-csv",
            str(dino_csv),
            "--device",
            dino_device,
            "--batch-size",
            "2",
        ]
        if bool(args.dino_local_files_only):
            dino.append("--local-files-only")
        rc = _run(dino)
        if rc != 0:
            return rc

    if not bool(args.skip_introstyle):
        print(intro_csv)
    if not bool(args.skip_dino):
        print(dino_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
