from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[launch_remote_round1_tokenizer_reconstruction_pretrain] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and optionally launch a round-1 tokenizer reconstruction-pretrain packet.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--manifest-csv", type=Path, default=SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv")
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--freeze-mode", choices=["tokenizer_only", "style_branch"], default="style_branch")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-switch-smoke", action="store_true")
    args, passthrough = parser.parse_known_args()

    prepare = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_round1_tokenizer_reconstruction_pretrain_config.py"),
        "--family-id",
        str(args.family_id),
        "--manifest-csv",
        str(Path(args.manifest_csv).resolve()),
        "--num-epochs",
        str(int(args.num_epochs)),
        "--save-interval",
        str(int(args.save_interval)),
        "--freeze-mode",
        str(args.freeze_mode),
    ]

    prep_proc = subprocess.run(prepare, cwd=str(WORKSPACE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False)
    print(prep_proc.stdout, end="")
    if prep_proc.returncode != 0:
        return int(prep_proc.returncode)
    pretrain_config = None
    for line in prep_proc.stdout.splitlines():
        text = line.strip()
        if text.endswith(".json"):
            pretrain_config = text
            break
    if not pretrain_config:
        raise RuntimeError("prepare_round1_tokenizer_reconstruction_pretrain_config.py did not report a pretrain config path")
    if bool(args.prepare_only):
        return 0

    launch = [
        sys.executable,
        str(SCRIPT_DIR / "launch_remote_round1_family_train.py"),
        "--config",
        str(Path(pretrain_config).resolve().relative_to(WORKSPACE.resolve())),
    ]
    if bool(args.skip_switch_smoke):
        launch.append("--skip-switch-smoke")
    launch.extend(passthrough)
    return _run(launch)


if __name__ == "__main__":
    raise SystemExit(main())
