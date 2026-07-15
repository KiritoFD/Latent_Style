from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
GENERIC_LAUNCHER = SCRIPT_DIR / "launch_remote_wsl_command.py"


def _run(cmd: list[str]) -> int:
    print("[launch_remote_wikiarts5_latent_prep] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch remote latent preparation for the new wikiarts-5 full-notest RGB train set.")
    parser.add_argument("--remote-image-root", default="/mnt/i/wikiarts_5_full_notest/train")
    parser.add_argument("--remote-latent-root", default="/mnt/i/wikiarts_5_full_notest_latents_ema/train")
    parser.add_argument("--styles", default="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e")
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=10000)
    parser.add_argument("--overwrite-latents", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    task_name = "wikiarts5-latent-prep"
    remote_log = f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{task_name}.log"
    cmd = [
        sys.executable,
        str(GENERIC_LAUNCHER),
        "--task-name",
        task_name,
        "--remote-log-path",
        remote_log,
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
        "--user",
        str(args.user),
        "--wsl-distro",
        str(args.wsl_distro),
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--no-health-check",
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/encode_image_folder_latents.py",
        "--sync-path",
        "SchrodingerBridge/tools/build_latent_packed_cache.py",
        "--sync-path",
        "SchrodingerBridge/tools/build_latent_prototype_pairing_cache.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/run_remote_wikiarts5_latent_prep.py",
        "--",
        str(args.remote_python),
        "SchrodingerBridge/tools/experiments/run_remote_wikiarts5_latent_prep.py",
        "--image-root",
        str(args.remote_image_root),
        "--latent-root",
        str(args.remote_latent_root),
        "--styles",
        str(args.styles),
        "--python-bin",
        str(args.remote_python),
        "--batch-size",
        str(int(args.batch_size)),
        "--vae-model",
        "ema",
        "--vae-cache-dir",
        f"{args.remote_wsl_cwd.rstrip('/')}/eval_cache/hf",
        "--seed",
        "20260610",
    ]
    if bool(args.overwrite_latents):
        cmd.append("--overwrite-latents")
    if bool(args.rebuild_cache):
        cmd.append("--rebuild-cache")
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
