from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent

PYTHON_BIN = "/home/xy/venvs/samam312/bin/python"
REMOTE_ROOT = "/mnt/i/Github/Latent_Style"
REMOTE_SB = f"{REMOTE_ROOT}/SchrodingerBridge"
LATENT_ROOT = "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train"
IMAGE_ROOT = "/mnt/i/wikiart_distinct5_samam_512_classview/train"
DINO_CACHE = f"{REMOTE_ROOT}/eval_cache/offline_pairing/dinov2_wikiart_distinct5_samam_512_train_cache.pt"
PAIRING_PLAN = f"{LATENT_ROOT}/.latent_cache/dino_pairing_top8.pt"
STYLES = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"


SYNC_PATHS = [
    "SchrodingerBridge/src/config_schema.py",
    "SchrodingerBridge/src/model.py",
    "SchrodingerBridge/src/model620.py",
    "SchrodingerBridge/src/blocks620.py",
    "SchrodingerBridge/src/style_encoder620.py",
    "SchrodingerBridge/src/losses620.py",
    "SchrodingerBridge/src/trainer.py",
    "SchrodingerBridge/src/run.py",
    "SchrodingerBridge/src/utils/dataset.py",
    "SchrodingerBridge/src/utils/training.py",
    "SchrodingerBridge/src/utils/run_evaluation.py",
    "SchrodingerBridge/tools/probe_620_path_liveness.py",
    "SchrodingerBridge/tools/probe_config_effectiveness.py",
    "SchrodingerBridge/tools/experiments/dino_cache_utils.py",
    "SchrodingerBridge/tools/experiments/build_offline_dino_pairing_cache.py",
    "SchrodingerBridge/tools/experiments/build_offline_dino_pairing_plan.py",
    "SchrodingerBridge/tools/experiments/launch_remote_620_spatial_bridge.py",
    "SchrodingerBridge/configs/620_spatial_bridge_base.json",
    "SchrodingerBridge/configs/620_spatial_bridge_swd4.json",
    "SchrodingerBridge/configs/620_spatial_bridge_swd12.json",
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
    "SchrodingerBridge/src/utils/dataset.py",
    "SchrodingerBridge/src/utils/training.py",
    "SchrodingerBridge/src/utils/run_evaluation.py",
    "SchrodingerBridge/tools/probe_620_path_liveness.py",
    "SchrodingerBridge/tools/probe_config_effectiveness.py",
    "SchrodingerBridge/tools/experiments/build_offline_dino_pairing_cache.py",
    "SchrodingerBridge/tools/experiments/build_offline_dino_pairing_plan.py",
    "SchrodingerBridge/tools/experiments/launch_remote_620_spatial_bridge.py",
]

CONFIG_BY_VARIANT = {
    "base": "SchrodingerBridge/configs/620_spatial_bridge_base.json",
    "swd4": "SchrodingerBridge/configs/620_spatial_bridge_swd4.json",
    "swd12": "SchrodingerBridge/configs/620_spatial_bridge_swd12.json",
}


def _run(cmd: list[str]) -> int:
    print("[launch_remote_620] " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    return int(subprocess.run(cmd, cwd=str(WORKSPACE), check=False).returncode)


def _remote_script(*, config_rel: str, run_name: str, epochs: int, smoke: bool) -> str:
    save_dir = f"./exp/620_spatial_bridge/{run_name}"
    if smoke:
        save_dir += "_smoke"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        cd {shlex.quote(REMOTE_ROOT)}
        export PYTHONPATH={shlex.quote(REMOTE_SB + '/src')}:{shlex.quote(REMOTE_SB + '/tools')}:{shlex.quote(REMOTE_SB + '/tools/experiments')}:$PYTHONPATH
        echo "[620] dataset root: {LATENT_ROOT}"
        test -d {shlex.quote(LATENT_ROOT)}
        test -d {shlex.quote(IMAGE_ROOT)}
        {shlex.quote(PYTHON_BIN)} - <<'PY'
        from pathlib import Path
        root = Path({LATENT_ROOT!r})
        styles = {STYLES.split(',')!r}
        counts = {{style: len([p for p in (root / style).iterdir() if p.is_file()]) for style in styles}}
        print("[620] balanced_counts", counts)
        if any(count != 1000 for count in counts.values()):
            raise SystemExit(f"expected exactly 1000 train latents per style, got {{counts}}")
        PY
        if [ ! -f {shlex.quote(DINO_CACHE)} ]; then
          echo "[620] building DINO cache: {DINO_CACHE}"
          mkdir -p {shlex.quote(str(PurePosixPath(DINO_CACHE).parent))}
          {shlex.quote(PYTHON_BIN)} SchrodingerBridge/tools/experiments/build_offline_dino_pairing_cache.py \\
            --latent-root {shlex.quote(LATENT_ROOT)} \\
            --image-root {shlex.quote(IMAGE_ROOT)} \\
            --output {shlex.quote(DINO_CACHE)}
        fi
        if [ ! -f {shlex.quote(PAIRING_PLAN)} ]; then
          echo "[620] building DINO pairing plan: {PAIRING_PLAN}"
          mkdir -p {shlex.quote(str(PurePosixPath(PAIRING_PLAN).parent))}
          {shlex.quote(PYTHON_BIN)} SchrodingerBridge/tools/experiments/build_offline_dino_pairing_plan.py \\
            --cache {shlex.quote(DINO_CACHE)} \\
            --output {shlex.quote(PAIRING_PLAN)} \\
            --topk 8 \\
            --styles {shlex.quote(STYLES)}
        fi
        {shlex.quote(PYTHON_BIN)} SchrodingerBridge/tools/probe_620_path_liveness.py --device cpu
        {shlex.quote(PYTHON_BIN)} - <<'PY'
        import json
        from pathlib import Path
        base = Path({(REMOTE_ROOT + '/' + config_rel)!r})
        payload = json.loads(base.read_text(encoding="utf-8"))
        payload.setdefault("training", {{}})["num_epochs"] = {int(epochs)}
        payload.setdefault("checkpoint", {{}})["save_dir"] = {save_dir!r}
        payload.setdefault("ablation", {{}})["name"] = {run_name!r}
        payload.setdefault("ablation", {{}})["stage"] = "smoke" if {bool(smoke)!r} else payload.get("ablation", {{}}).get("stage", "formal")
        out = Path({(REMOTE_SB + '/configs/_generated_620_launch.json')!r})
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        print("[620] launch_config", out)
        PY
        {shlex.quote(PYTHON_BIN)} SchrodingerBridge/src/run.py --config SchrodingerBridge/configs/_generated_620_launch.json
        """
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch 620 spatial bridge on the remote 3060 WSL lane.")
    parser.add_argument("--variant", choices=sorted(CONFIG_BY_VARIANT), default="base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--formal", action="store_true", help="Run formal 8-epoch training for the selected variant.")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    epochs = 8 if args.formal else max(1, int(args.epochs))
    smoke = not bool(args.formal)
    variant = str(args.variant)
    run_name = {
        "base": "620_base_swd8_sigma002_nfe8",
        "swd4": "620_swd4_sigma002_nfe8",
        "swd12": "620_swd12_sigma002_nfe8",
    }[variant]
    task_name = str(args.task_name).strip() or (run_name + ("_smoke" if smoke else "_formal"))
    remote_log = f"{REMOTE_SB}/exp/620_spatial_bridge/{task_name}.remote.log"
    remote_script = _remote_script(
        config_rel=CONFIG_BY_VARIANT[variant],
        run_name=run_name,
        epochs=epochs,
        smoke=smoke,
    )
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
        "--runtime-guard-max-memory-mib",
        "11264",
        "--runtime-guard-poll-seconds",
        "15",
    ]
    for path in SYNC_PATHS:
        cmd.extend(["--sync-path", path])
    for path in VERIFY_PYTHON_FILES:
        cmd.extend(["--verify-python-file", path])
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.extend(["--", "/bin/bash", "-lc", remote_script])
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
