from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
SOURCE_ROOT = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test")
INTROSTYLE_MODEL_ID = Path(r"G:\GitHub\Latent_Style\eval_cache\modelscope\stabilityai\stable-diffusion-2-1-base")
INTROSTYLE_BANK_CACHE_ROOT = WORKSPACE / "eval_cache" / "introstyle_bank_vectors"
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config
from local_gpu_lock import run_with_local_gpu_lock
from round1_paths import infer_round1_family_id, round1_localreview_root


def _run(cmd: list[str]) -> int:
    print("[run_local_round1_family_review] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    src_path = str(SB_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    proc = subprocess.run(cmd, check=False, cwd=str(SB_ROOT), env=env)
    return int(proc.returncode)


def _run_locked(cmd: list[str], *, owner: str) -> int:
    print("[run_local_round1_family_review] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    src_path = str(SB_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    return run_with_local_gpu_lock(cmd, owner=owner, cwd=str(SB_ROOT), env=env)


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
    parser = argparse.ArgumentParser(description="Run generic local IntroStyle/DINO review for a round-1 family.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--eval-subdir", default="full_eval_fresh_localreview")
    parser.add_argument("--local-root", type=Path, default=None)
    parser.add_argument("--label-prefix", default="")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-introstyle", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--introstyle-device", default="")
    parser.add_argument("--introstyle-batch-size", type=int, default=1)
    parser.add_argument("--introstyle-bank-batch-size", type=int, default=4)
    parser.add_argument("--introstyle-ensemble-size", type=int, default=1)
    parser.add_argument("--introstyle-bank-cache-path", type=Path, default=None)
    parser.add_argument("--dino-device", default="")
    parser.add_argument("--dino-local-files-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config((WORKSPACE / Path(args.config)).resolve())
    run_name = str((cfg.get("ablation") or {}).get("name", Path(args.config).stem)).strip() or Path(args.config).stem
    family_id = infer_round1_family_id(run_name=run_name, config_stem=Path(args.config).stem)
    local_root = Path(args.local_root) if args.local_root is not None else round1_localreview_root(family_id=family_id, run_name=run_name)
    local_root.mkdir(parents=True, exist_ok=True)
    eval_subdir = str(args.eval_subdir).strip() or "full_eval_fresh_localreview"
    handoff_csv = local_root / f"{eval_subdir}_bestfew_handoff.csv"
    manifest_csv = local_root / f"{eval_subdir}_bestfew_introstyle_manifest.csv"
    intro_csv = local_root / f"{eval_subdir}_bestfew_introstyle.csv"
    intro_json = local_root / f"{eval_subdir}_bestfew_introstyle.json"
    dino_csv = local_root / f"{eval_subdir}_bestfew_dino.csv"
    merged_csv = local_root / f"{eval_subdir}_bestfew_introstyle_dino.csv"
    introstyle_bank_cache_path = (
        Path(args.introstyle_bank_cache_path).resolve()
        if args.introstyle_bank_cache_path is not None
        else INTROSTYLE_BANK_CACHE_ROOT / "distinct5_sd21_t25_u1_e1.pt"
    )

    if not bool(args.skip_pull):
        pull = [
            sys.executable,
            str(SCRIPT_DIR / "pull_remote_round1_family_localreview.py"),
            "--config",
            str(args.config),
            "--eval-subdir",
            eval_subdir,
            "--local-root",
            str(local_root),
        ]
        rc = _run(pull)
        if rc != 0:
            return rc

    label_prefix = str(args.label_prefix).strip() or run_name
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
        label_prefix,
        "--source-root",
        str(SOURCE_ROOT),
    ]
    rc = _run(manifest)
    if rc != 0:
        return rc

    if not _images_present(handoff_csv):
        print(
            "[run_local_round1_family_review] images are absent in the pulled eval dirs; "
            "skip local IntroStyle/DINO until the image-backed rerun is available.",
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
            str(max(1, int(args.introstyle_batch_size))),
            "--bank-batch-size",
            str(max(1, int(args.introstyle_bank_batch_size))),
            "--ensemble_size",
            str(max(1, int(args.introstyle_ensemble_size))),
            "--bank-cache-path",
            str(introstyle_bank_cache_path),
        ]
        rc = _run_locked(intro, owner=f"run_local_round1_family_review:introstyle:{run_name}")
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
        rc = _run_locked(dino, owner=f"run_local_round1_family_review:dino:{run_name}")
        if rc != 0:
            return rc

    if (not bool(args.skip_introstyle)) and (not bool(args.skip_dino)):
        merge = [
            sys.executable,
            str(SCRIPT_DIR / "merge_introstyle_dino_reviews.py"),
            "--introstyle-csv",
            str(intro_csv),
            "--dino-csv",
            str(dino_csv),
            "--output-csv",
            str(merged_csv),
        ]
        rc = _run(merge)
        if rc != 0:
            return rc

    if family_id:
        rc = _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "update_round1_family_status_docs.py"),
                "--family-id",
                str(family_id),
            ]
        )
        if rc != 0:
            return rc

    for path in (manifest_csv, intro_csv, dino_csv, merged_csv):
        if path.exists():
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
