from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = Path(
    r"G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samam_wikiarts5_patch8_segmented_20260610_094447"
)
DEFAULT_CONTROLLER_SCRIPT = SCRIPT_DIR / "run_samam_wikiarts5_segmented_eval_wsl.sh"
DEFAULT_STDOUT_LOG = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\samam_wikiarts5_segmented.stdout.log")
DEFAULT_STDERR_LOG = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\samam_wikiarts5_segmented.stderr.log")


def _to_wsl_mount(path: Path) -> str:
    text = str(path)
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        remainder = text[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{remainder}" if remainder else f"/mnt/{drive}"
    return text.replace("\\", "/")


def _json_flag(path: Path, *, key: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    value = payload
    for part in str(key).split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return bool(value)


def _wsl_has_process(*, distro: str, match_text: str) -> bool:
    proc = subprocess.run(
        [
            "wsl",
            "-d",
            str(distro),
            "bash",
            "-lc",
            f"ps -eo cmd | grep -F -- {shlex.quote(str(match_text))} | grep -v grep || true",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return bool(proc.stdout.strip())


def _launch_controller(
    *,
    distro: str,
    controller_script: Path,
    result_root: Path,
    stdout_log: Path,
    stderr_log: Path,
    baseline_python: str,
    max_steps: int,
    step_interval: int,
    patch_size: int,
    batch_size: int,
    gradient_checkpointing: int,
    identity_gradient_checkpointing: int,
    limit_val_batches: float,
    num_sanity_val_steps: int,
) -> None:
    controller_script_wsl = _to_wsl_mount(controller_script)
    result_root_wsl = _to_wsl_mount(result_root)
    stdout_log_wsl = _to_wsl_mount(stdout_log)
    stderr_log_wsl = _to_wsl_mount(stderr_log)
    exports = {
        "BASELINE_PYTHON": baseline_python,
        "OUT_ROOT": result_root_wsl,
        "MAX_STEPS": str(int(max_steps)),
        "STOP_AT_MAX_STEPS": "0",
        "STEP_INTERVAL": str(int(step_interval)),
        "PATCH_SIZE": str(int(patch_size)),
        "BATCH_SIZE": str(int(batch_size)),
        "GRADIENT_CHECKPOINTING": str(int(gradient_checkpointing)),
        "IDENTITY_GRADIENT_CHECKPOINTING": str(int(identity_gradient_checkpointing)),
        "LIMIT_VAL_BATCHES": str(float(limit_val_batches)),
        "NUM_SANITY_VAL_STEPS": str(int(num_sanity_val_steps)),
    }
    export_cmd = " ".join(f"export {key}={shlex.quote(value)};" for key, value in exports.items())
    launch_cmd = (
        "set -euo pipefail; "
        + export_cmd
        + f" {shlex.quote(controller_script_wsl)} >> {shlex.quote(stdout_log_wsl)} 2>> {shlex.quote(stderr_log_wsl)}"
    )
    cmd = ["wsl", "-d", str(distro), "bash", "-lc", launch_cmd]
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep the local wikiarts5 segmented SaMAM lane alive until convergence.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--controller-script", type=Path, default=DEFAULT_CONTROLLER_SCRIPT)
    parser.add_argument("--stdout-log", type=Path, default=DEFAULT_STDOUT_LOG)
    parser.add_argument("--stderr-log", type=Path, default=DEFAULT_STDERR_LOG)
    parser.add_argument("--wsl-distro", default="f")
    parser.add_argument("--baseline-python", default="/root/venvs/samam/bin/python")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--launch-cooldown-seconds", type=int, default=180)
    parser.add_argument("--convergence-json-key", default="converged")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--step-interval", type=int, default=250)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", type=int, default=1)
    parser.add_argument("--identity-gradient-checkpointing", type=int, default=1)
    parser.add_argument("--limit-val-batches", type=float, default=0.1)
    parser.add_argument("--num-sanity-val-steps", type=int, default=0)
    args = parser.parse_args()

    result_root = Path(args.result_root).expanduser()
    controller_script = Path(args.controller_script).expanduser()
    stdout_log = Path(args.stdout_log).expanduser()
    stderr_log = Path(args.stderr_log).expanduser()
    if not result_root.is_absolute():
        result_root = (Path.cwd() / result_root).resolve()
    if not controller_script.is_absolute():
        controller_script = (Path.cwd() / controller_script).resolve()
    if not stdout_log.is_absolute():
        stdout_log = (Path.cwd() / stdout_log).resolve()
    if not stderr_log.is_absolute():
        stderr_log = (Path.cwd() / stderr_log).resolve()
    convergence_json = result_root / "curve_convergence.json"
    result_root_wsl = _to_wsl_mount(result_root)
    controller_script_wsl = _to_wsl_mount(controller_script)
    last_launch_ts = 0.0

    while True:
        converged = _json_flag(convergence_json, key=str(args.convergence_json_key))
        controller_busy = _wsl_has_process(distro=str(args.wsl_distro), match_text=controller_script_wsl)
        train_busy = _wsl_has_process(distro=str(args.wsl_distro), match_text=result_root_wsl)
        payload = {
            "result_root": str(result_root),
            "converged": converged,
            "controller_busy": controller_busy,
            "train_busy": train_busy,
            "convergence_json": str(convergence_json),
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if converged and (not controller_busy) and (not train_busy):
            return 0
        if (not converged) and (not controller_busy) and (not train_busy):
            now = time.time()
            if now - last_launch_ts >= max(1, int(args.launch_cooldown_seconds)):
                _launch_controller(
                    distro=str(args.wsl_distro),
                    controller_script=controller_script,
                    result_root=result_root,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    baseline_python=str(args.baseline_python),
                    max_steps=int(args.max_steps),
                    step_interval=int(args.step_interval),
                    patch_size=int(args.patch_size),
                    batch_size=int(args.batch_size),
                    gradient_checkpointing=int(args.gradient_checkpointing),
                    identity_gradient_checkpointing=int(args.identity_gradient_checkpointing),
                    limit_val_batches=float(args.limit_val_batches),
                    num_sanity_val_steps=int(args.num_sanity_val_steps),
                )
                last_launch_ts = now
                print(
                    json.dumps(
                        {
                            "action": "launch_controller",
                            "controller_script": str(controller_script),
                            "result_root": str(result_root),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
