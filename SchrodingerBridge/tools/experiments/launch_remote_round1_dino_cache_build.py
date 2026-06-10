from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config
from dino_cache_utils import default_dino_cache_output, infer_image_root_for_latent_root


def _run(cmd: list[str]) -> int:
    print("[launch_remote_round1_dino_cache_build] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a remote DINO cache build for a round-1 tokenizer family config.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--image-root-override", default="")
    parser.add_argument("--output-override", default="")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=10000)
    args = parser.parse_args()

    config_rel = Path(args.config)
    cfg = load_config((WORKSPACE / config_rel).resolve())
    latent_root = Path(str((cfg.get("data") or {}).get("data_root", "")).strip())
    if not str(latent_root):
        raise ValueError("config.data.data_root is required")
    image_root = Path(str(args.image_root_override).strip()) if str(args.image_root_override).strip() else infer_image_root_for_latent_root(latent_root)
    output_path = Path(str(args.output_override).strip()) if str(args.output_override).strip() else default_dino_cache_output(latent_root, workspace_root=Path(args.remote_wsl_cwd))
    styles = [str(x).strip() for x in ((cfg.get("data") or {}).get("style_subdirs") or []) if str(x).strip()]
    run_name = str((cfg.get("ablation") or {}).get("name", config_rel.stem)).strip() or config_rel.stem
    image_root_arg = image_root.as_posix()
    latent_root_arg = latent_root.as_posix()
    output_arg = output_path.as_posix()

    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    build_script_rel = Path("SchrodingerBridge/tools/experiments/build_offline_dino_pairing_cache.py")
    utils_script_rel = Path("SchrodingerBridge/tools/experiments/dino_cache_utils.py")
    cmd = [
        sys.executable,
        str(launch),
        "--task-name",
        f"round1-{run_name}-dino-cache-build",
        "--remote-log-path",
        f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_dino_cache_build.log",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        "--sync-path",
        str(build_script_rel),
        "--sync-path",
        str(utils_script_rel),
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--no-health-check",
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            f"{args.remote_python} {build_script_rel.as_posix()} "
            f"--image-root {image_root_arg} "
            f"--latent-root {latent_root_arg} "
            f"--output {output_arg} "
            f"--styles {','.join(styles)} "
            f"--batch-size {int(args.batch_size)} "
            "--device cuda"
        ),
    ]
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
