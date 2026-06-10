from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config


def _run(cmd: list[str]) -> int:
    print("[pull_remote_round1_family_localreview] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull a round-1 family image-backed localreview bundle from the remote host.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--eval-subdir", default="full_eval_fresh_localreview")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--host", default="administrator@100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    args = parser.parse_args()

    config_rel = Path(args.config)
    cfg = load_config((WORKSPACE / config_rel).resolve())
    run_name = str((cfg.get("ablation") or {}).get("name", config_rel.stem)).strip() or config_rel.stem
    run_dir = str((cfg.get("checkpoint") or {}).get("save_dir", "")).strip()
    remote_run_dir = run_dir.replace("./", "/mnt/i/Github/Latent_Style/")

    eval_subdir = str(args.eval_subdir).strip() or "full_eval_fresh_localreview"
    local_root = Path(args.local_root)
    local_eval_root = local_root / eval_subdir
    local_root.mkdir(parents=True, exist_ok=True)
    local_eval_root.mkdir(parents=True, exist_ok=True)
    curve_csv = local_root / f"{eval_subdir}_clip_lpips_curve.csv"
    handoff_csv = local_root / f"{eval_subdir}_bestfew_handoff.csv"

    pull_curve = [
        sys.executable,
        str(SCRIPT_DIR / "pull_remote_eval_curve.py"),
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
        "--wsl-distro",
        str(args.wsl_distro),
        "--remote-run-dir",
        remote_run_dir,
        "--eval-subdir",
        eval_subdir,
        "--output-csv",
        str(curve_csv),
    ]
    rc = _run(pull_curve)
    if rc != 0:
        return rc

    build_handoff = [
        sys.executable,
        str(SCRIPT_DIR / "build_best_few_handoff.py"),
        "--curve-csv",
        str(curve_csv),
        "--run-name",
        run_name,
        "--eval-root",
        str(local_eval_root),
        "--output-csv",
        str(handoff_csv),
    ]
    rc = _run(build_handoff)
    if rc != 0:
        return rc

    rows = _read_rows(handoff_csv)
    for row in rows:
        epoch = str(row["epoch"]).strip()
        remote_dir = f"{remote_run_dir.rstrip('/')}/{eval_subdir}/{epoch}"
        local_dir = local_eval_root / epoch
        tar_name = f"{run_name}_{eval_subdir}_{epoch}.tar"
        pull_epoch = [
            sys.executable,
            str(SCRIPT_DIR / "pull_remote_eval_dir.py"),
            "--host",
            str(args.host),
            "--port",
            str(int(args.port)),
            "--wsl-distro",
            str(args.wsl_distro),
            "--remote-dir",
            remote_dir,
            "--local-dir",
            str(local_dir),
            "--tar-name",
            tar_name,
        ]
        rc = _run(pull_epoch)
        if rc != 0:
            return rc

    print(handoff_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
