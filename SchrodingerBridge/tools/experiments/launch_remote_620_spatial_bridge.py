from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent

PYTHON_BIN = "/home/xy/venvs/samam312/bin/python"
REMOTE_ROOT = "/mnt/i/Github/Latent_Style"
REMOTE_SB = f"{REMOTE_ROOT}/SchrodingerBridge"
SYNC_PATHS = [
    "SchrodingerBridge/src/config_schema.py",
    "SchrodingerBridge/src/model.py",
    "SchrodingerBridge/src/model620.py",
    "SchrodingerBridge/src/blocks620.py",
    "SchrodingerBridge/src/style_encoder620.py",
    "SchrodingerBridge/src/losses620.py",
    "SchrodingerBridge/src/trainer.py",
    "SchrodingerBridge/src/run.py",
    "SchrodingerBridge/src/style_families.py",
    "SchrodingerBridge/src/utils/dataset.py",
    "SchrodingerBridge/src/utils/inference.py",
    "SchrodingerBridge/src/utils/training.py",
    "SchrodingerBridge/src/utils/run_evaluation.py",
    "SchrodingerBridge/tools/probe_620_path_liveness.py",
    "SchrodingerBridge/tools/probe_620_endpoint_decomposition.py",
    "SchrodingerBridge/tools/probe_620_fog_path.py",
    "SchrodingerBridge/tools/probe_620_endpoint_time_sweep.py",
    "SchrodingerBridge/tools/probe_620_hypothesis_metrics.py",
    "SchrodingerBridge/tools/probe_620_solver_trace.py",
    "SchrodingerBridge/tools/probe_config_effectiveness.py",
    "SchrodingerBridge/tools/experiments/dino_cache_utils.py",
    "SchrodingerBridge/tools/experiments/build_offline_dino_pairing_cache.py",
    "SchrodingerBridge/tools/experiments/build_offline_dino_pairing_plan.py",
    "SchrodingerBridge/tools/experiments/collect_round2_eval_curve.py",
    "SchrodingerBridge/tools/experiments/build_clip_lpips_curve_from_eval_root.py",
    "SchrodingerBridge/tools/experiments/backfill_eval_clip_schema.py",
    "SchrodingerBridge/tools/experiments/launch_remote_620_spatial_bridge.py",
    "SchrodingerBridge/tools/experiments/run_remote_620_spatial_bridge.sh",
    "SchrodingerBridge/configs/620_spatial_bridge_base.json",
    "SchrodingerBridge/configs/620_spatial_bridge_swd4.json",
    "SchrodingerBridge/configs/620_spatial_bridge_swd12.json",
    "SchrodingerBridge/configs/620_spatial_bridge_swd16.json",
    "SchrodingerBridge/configs/620_spatial_bridge_swd20.json",
    "SchrodingerBridge/configs/620_spatial_bridge_swd24.json",
    "SchrodingerBridge/configs/620_spatial_bridge_adapter.json",
    "SchrodingerBridge/configs/620_spatial_bridge_moe.json",
    "SchrodingerBridge/configs/620_spatial_bridge_gate12.json",
    "SchrodingerBridge/configs/620_spatial_bridge_lowmix.json",
    "SchrodingerBridge/configs/620_spatial_bridge_lowfreqfix.json",
    "SchrodingerBridge/configs/620_spatial_bridge_lowfreqfix_debug.json",
    "SchrodingerBridge/configs/620_spatial_bridge_targetlinear.json",
    "SchrodingerBridge/configs/620_spatial_bridge_targetlinear_endpointlowhigh.json",
    "SchrodingerBridge/configs/620_spatial_bridge_targetlinear_endpointstylehead.json",
    "SchrodingerBridge/configs/620_spatial_bridge_targetlinear_debug.json",
    "SchrodingerBridge/configs/620_spatial_bridge_film_smoke.json",
    "SchrodingerBridge/configs/620_spatial_bridge_film_formal.json",
    "SchrodingerBridge/configs/620_spatial_bridge_dim128.json",
    "SchrodingerBridge/configs/620_spatial_bridge_intrinsic.json",
    "SchrodingerBridge/configs/620_spatial_bridge_lowswd.json",
    "SchrodingerBridge/configs/620_spatial_bridge_contentkv.json",
    "SchrodingerBridge/exp/phase616_live_dashboard/sync_phase616_live_dashboard.py",
]

VERIFY_PYTHON_FILES = [
    "SchrodingerBridge/src/config_schema.py",
    "SchrodingerBridge/src/model.py",
    "SchrodingerBridge/src/model620.py",
    "SchrodingerBridge/src/blocks620.py",
    "SchrodingerBridge/src/style_encoder620.py",
    "SchrodingerBridge/src/losses620.py",
    "SchrodingerBridge/src/trainer.py",
    "SchrodingerBridge/src/run.py",
    "SchrodingerBridge/src/style_families.py",
    "SchrodingerBridge/src/utils/dataset.py",
    "SchrodingerBridge/src/utils/inference.py",
    "SchrodingerBridge/src/utils/training.py",
    "SchrodingerBridge/src/utils/run_evaluation.py",
    "SchrodingerBridge/tools/probe_620_path_liveness.py",
    "SchrodingerBridge/tools/probe_620_endpoint_decomposition.py",
    "SchrodingerBridge/tools/probe_620_fog_path.py",
    "SchrodingerBridge/tools/probe_620_endpoint_time_sweep.py",
    "SchrodingerBridge/tools/probe_620_hypothesis_metrics.py",
    "SchrodingerBridge/tools/probe_620_solver_trace.py",
    "SchrodingerBridge/tools/probe_config_effectiveness.py",
    "SchrodingerBridge/tools/experiments/build_offline_dino_pairing_cache.py",
    "SchrodingerBridge/tools/experiments/build_offline_dino_pairing_plan.py",
    "SchrodingerBridge/tools/experiments/collect_round2_eval_curve.py",
    "SchrodingerBridge/tools/experiments/build_clip_lpips_curve_from_eval_root.py",
    "SchrodingerBridge/tools/experiments/backfill_eval_clip_schema.py",
    "SchrodingerBridge/tools/experiments/launch_remote_620_spatial_bridge.py",
]

CONFIG_BY_VARIANT = {
    "base": "SchrodingerBridge/configs/620_spatial_bridge_base.json",
    "swd4": "SchrodingerBridge/configs/620_spatial_bridge_swd4.json",
    "swd12": "SchrodingerBridge/configs/620_spatial_bridge_swd12.json",
    "swd16": "SchrodingerBridge/configs/620_spatial_bridge_swd16.json",
    "swd20": "SchrodingerBridge/configs/620_spatial_bridge_swd20.json",
    "swd24": "SchrodingerBridge/configs/620_spatial_bridge_swd24.json",
    "adapter": "SchrodingerBridge/configs/620_spatial_bridge_adapter.json",
    "moe": "SchrodingerBridge/configs/620_spatial_bridge_moe.json",
    "gate12": "SchrodingerBridge/configs/620_spatial_bridge_gate12.json",
    "lowmix": "SchrodingerBridge/configs/620_spatial_bridge_lowmix.json",
    "lowfreqfix": "SchrodingerBridge/configs/620_spatial_bridge_lowfreqfix.json",
    "lowfreqfix_debug": "SchrodingerBridge/configs/620_spatial_bridge_lowfreqfix_debug.json",
    "targetlinear": "SchrodingerBridge/configs/620_spatial_bridge_targetlinear.json",
    "endpointlowhigh": "SchrodingerBridge/configs/620_spatial_bridge_targetlinear_endpointlowhigh.json",
    "endpointstylehead": "SchrodingerBridge/configs/620_spatial_bridge_targetlinear_endpointstylehead.json",
    "targetlinear_debug": "SchrodingerBridge/configs/620_spatial_bridge_targetlinear_debug.json",
    "film": "SchrodingerBridge/configs/620_spatial_bridge_film_smoke.json",
    "film_formal": "SchrodingerBridge/configs/620_spatial_bridge_film_formal.json",
    "dim128": "SchrodingerBridge/configs/620_spatial_bridge_dim128.json",
    "intrinsic": "SchrodingerBridge/configs/620_spatial_bridge_intrinsic.json",
    "lowswd": "SchrodingerBridge/configs/620_spatial_bridge_lowswd.json",
    "contentkv": "SchrodingerBridge/configs/620_spatial_bridge_contentkv.json",
}


def _run(cmd: list[str]) -> int:
    print("[launch_remote_620] " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    return int(subprocess.run(cmd, cwd=str(WORKSPACE), check=False).returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch 620 spatial bridge on the remote 3060 WSL lane.")
    parser.add_argument("--variant", choices=sorted(CONFIG_BY_VARIANT), default="base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--formal", action="store_true", help="Run formal 8-epoch training for the selected variant.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional training batch size override; must be divisible by 16.")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--run-name", default="", help="Optional remote run directory name. Defaults to *_smoke for smoke runs and the formal run name for formal runs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    epochs = 8 if args.formal else max(1, int(args.epochs))
    smoke = not bool(args.formal)
    variant = str(args.variant)
    run_name = {
        "base": "620_base_swd8_sigma002_nfe8_b80",
        "swd4": "620_swd4_sigma002_nfe8_b80",
        "swd12": "620_swd12_sigma002_nfe8_b80",
        "swd16": "620_swd16_sigma002_nfe8_b64",
        "swd20": "620_swd20_sigma002_nfe8_b64",
        "swd24": "620_swd24_sigma002_nfe8_b64",
        "adapter": "620_adapter_swd12_sigma002_nfe8_b64",
        "moe": "620_moe_swd12_sigma002_nfe8_b64",
        "gate12": "620_gate12_adapter_swd12_sigma002_nfe8_b64",
        "lowmix": "620_lowmix05_gate12_adapter_swd12_sigma002_nfe8_b64",
        "lowfreqfix": "620_lowfreqfix_swd8_sigma002_nfe8_b80",
        "lowfreqfix_debug": "620_lowfreqfix_debug_b16_gs2",
        "targetlinear": "620_targetlinear_swd8_sigma002_nfe8_b80",
        "endpointlowhigh": "620_targetlinear_endpointlowhigh_swd8_sigma002_nfe8_b80",
        "endpointstylehead": "620_targetlinear_endpointstylehead_swd8_sigma002_nfe8_b80",
        "targetlinear_debug": "620_targetlinear_debug_b16_gs2",
        "film": "620_film_smoke",
        "film_formal": "620_film_formal",
        "dim128": "620_dim128_formal",
        "intrinsic": "620_intrinsic_v2",
        "lowswd": "620_lowswd_formal",
        "contentkv": "620_contentkv_gate12_adapter_swd12_sigma002_nfe8_b64",
    }[variant]
    if args.batch_size is not None and int(args.batch_size) % 16 != 0:
        raise SystemExit(f"--batch-size must be divisible by 16, got {args.batch_size}")
    remote_run_name = str(args.run_name).strip() or (run_name + ("_smoke" if smoke else ""))
    task_name = str(args.task_name).strip() or (remote_run_name if smoke else run_name + "_formal")
    remote_log = f"{REMOTE_SB}/exp/620_spatial_bridge/{task_name}.remote.log"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "launch_remote_wsl_command.py"),
        "--task-name",
        task_name,
        "--remote-log-path",
        remote_log,
        "--remote-wsl-cwd",
        REMOTE_ROOT,
        "--remote-workspace-root",
        REMOTE_ROOT,
        "--python-bin",
        PYTHON_BIN,
        "--max-prelaunch-memory-mib",
        "1500",
        "--max-runtime-memory-mib",
        "12288",
        "--runtime-guard-max-memory-mib",
        "12288",
        "--runtime-guard-poll-seconds",
        "15",
    ]
    for path in SYNC_PATHS:
        cmd.extend(["--sync-path", path])
    for path in VERIFY_PYTHON_FILES:
        cmd.extend(["--verify-python-file", path])
    if args.dry_run:
        cmd.append("--dry-run")
    remote_cmd = [
        "bash",
        "SchrodingerBridge/tools/experiments/run_remote_620_spatial_bridge.sh",
        "--variant",
        variant,
        "--epochs",
        str(int(epochs)),
        "--run-name",
        remote_run_name,
    ]
    if args.batch_size is not None:
        remote_cmd.extend(["--batch-size", str(int(args.batch_size))])
    if not smoke:
        remote_cmd.append("--formal")
    cmd.extend(["--", *remote_cmd])
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
