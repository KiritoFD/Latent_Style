from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from csv_utils import manifest_fieldnames, read_csv_rows, write_csv_rows
from config_schema import load_experiment_config
from style_families import validate_dino_retired_runtime


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round2_pure_sde" / "round2_family_manifest.csv"


def _run(cmd: list[str]) -> int:
    print("[launch_remote_round2_family_train] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def _validate_config(config_path: Path, *, allow_dino: bool) -> None:
    cfg = load_experiment_config(config_path)
    validate_dino_retired_runtime(
        tokenizer_family=str(getattr(cfg.model, "tokenizer_family", "legacy_factorized")),
        semantic_supervision_family=str(getattr(cfg.bridge, "semantic_supervision_family", "legacy_terminal_swd")),
        allow_dino=allow_dino,
        context="round2 pure-sde family launch",
    )


def _load_manifest_row(*, manifest_csv: Path, family_id: str) -> dict[str, str]:
    rows = read_csv_rows(manifest_csv)
    row = next((x for x in rows if str(x.get("family_id", "")).strip() == str(family_id).strip()), None)
    if row is None:
        raise KeyError(f"family_id not found in manifest: {family_id}")
    return row


def _update_manifest_status(*, manifest_csv: Path, family_id: str, status: str) -> None:
    rows = read_csv_rows(manifest_csv)
    changed = False
    for row in rows:
        if str(row.get("family_id", "")).strip() != str(family_id).strip():
            continue
        row["decision_status"] = str(status).strip()
        changed = True
        break
    if changed:
        write_csv_rows(manifest_csv, rows, fieldnames=manifest_fieldnames(rows))


def _running_family_ids(*, manifest_csv: Path, exclude_family_id: str) -> list[str]:
    rows = read_csv_rows(manifest_csv)
    return [
        str(row.get("family_id", "")).strip()
        for row in rows
        if str(row.get("decision_status", "")).strip().lower() == "running"
        and str(row.get("family_id", "")).strip() != str(exclude_family_id).strip()
    ]


def _append_remote_run_note(
    *,
    family_id: str,
    manifest_row: dict[str, str],
    remote_wsl_cwd: str,
    remote_python: str,
    health_wait_seconds: int,
) -> None:
    doc = SB_ROOT / "docs" / "experiments" / "round2_pure_sde" / family_id / "remote_run.md"
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "",
        f"## Launch {now}",
        f"- Config: `{manifest_row.get('config_path', '')}`",
        f"- Run name: `{manifest_row.get('run_name', '')}`",
        f"- Run dir: `{manifest_row.get('run_dir', '')}`",
        f"- Remote cwd: `{remote_wsl_cwd}`",
        f"- Remote python: `{remote_python}`",
        f"- Health wait seconds: `{health_wait_seconds}`",
        "- Contract:",
        "  - formal band `9.0-10.8 GiB`",
        "  - hard stop `11.0 GiB`",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    with doc.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _arm_remote_fast_eval_followup(*, family_id: str, config_path: Path) -> None:
    helper = SCRIPT_DIR / "launch_remote_round2_family_fast_eval.py"
    if not helper.exists():
        return
    cmd = [
        sys.executable,
        str(helper),
        "--family-id",
        str(family_id),
        "--config",
        str(config_path),
    ]
    print("[launch_remote_round2_family_train] arm remote fast-eval followup -> " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(WORKSPACE), check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch one round-2 pure-latent/I2SB family on the remote 3060 host.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--allow-other-running", action="store_true")
    parser.add_argument("--arm-remote-fast-eval-followup", action="store_true")
    parser.add_argument("--allow-dino", action="store_true", help="Override the default round2 policy that archives DINO-conditioned configs.")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-device", default="cpu")
    parser.add_argument("--smoke-latent-size", type=int, default=32)
    parser.add_argument("--smoke-bank-tokens", type=int, default=8)
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=7000)
    parser.add_argument("--min-runtime-memory-mib", type=int, default=9216)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=10800)
    parser.add_argument("--min-runtime-slack-mib", type=int, default=128)
    parser.add_argument("--runtime-guard-max-memory-mib", type=int, default=11000)
    parser.add_argument("--runtime-guard-poll-seconds", type=int, default=10)
    parser.add_argument("--runtime-guard-min-memory-mib", type=int, default=9216)
    parser.add_argument("--runtime-guard-min-warmup-seconds", type=int, default=300)
    parser.add_argument("--runtime-guard-min-consecutive-polls", type=int, default=3)
    parser.add_argument("--runtime-guard-min-mode", choices=["ignore", "warn", "stop"], default="stop")
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    family_id = str(args.family_id).strip()
    row = _load_manifest_row(manifest_csv=manifest_csv, family_id=family_id)
    if not bool(args.allow_other_running):
        foreign_running = _running_family_ids(manifest_csv=manifest_csv, exclude_family_id=family_id)
        if foreign_running:
            raise RuntimeError(
                "Refusing round-2 launch because other running families exist: " + ", ".join(foreign_running)
            )

    launch_health_min_runtime_mib = 0
    raw_launch_floor = str(row.get("launch_health_min_runtime_memory_mib", "")).strip()
    if raw_launch_floor:
        try:
            launch_health_min_runtime_mib = max(0, int(raw_launch_floor))
        except ValueError:
            launch_health_min_runtime_mib = 0

    config_path = Path(str(row.get("config_path", "")).strip())
    if not config_path.is_absolute():
        config_path = (WORKSPACE / config_path).resolve()
    _validate_config(config_path, allow_dino=bool(args.allow_dino))
    launcher = SCRIPT_DIR / "launch_remote_experiment_train.py"
    cmd = [
        sys.executable,
        str(launcher),
        "--config",
        str(config_path),
        "--task-prefix",
        "round2",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--remote-python",
        str(args.remote_python),
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--min-runtime-memory-mib",
        str(int(launch_health_min_runtime_mib or args.min_runtime_memory_mib)),
        "--max-runtime-memory-mib",
        str(int(args.max_runtime_memory_mib)),
        "--min-runtime-slack-mib",
        str(int(args.min_runtime_slack_mib)),
        "--runtime-guard-max-memory-mib",
        str(int(args.runtime_guard_max_memory_mib)),
        "--runtime-guard-poll-seconds",
        str(int(args.runtime_guard_poll_seconds)),
        "--runtime-guard-min-memory-mib",
        str(int(args.runtime_guard_min_memory_mib)),
        "--runtime-guard-min-warmup-seconds",
        str(int(args.runtime_guard_min_warmup_seconds)),
        "--runtime-guard-min-consecutive-polls",
        str(int(args.runtime_guard_min_consecutive_polls)),
        "--runtime-guard-min-mode",
        str(args.runtime_guard_min_mode),
        "--health-wait-seconds",
        str(int(args.health_wait_seconds)),
    ]
    if bool(args.skip_smoke):
        cmd.append("--skip-smoke")
    else:
        cmd.extend(
            [
                "--smoke-device",
                str(args.smoke_device),
                "--smoke-latent-size",
                str(int(args.smoke_latent_size)),
                "--smoke-bank-tokens",
                str(int(args.smoke_bank_tokens)),
            ]
        )

    rc = _run(cmd)
    if rc == 0:
        _update_manifest_status(manifest_csv=manifest_csv, family_id=family_id, status="running")
        _append_remote_run_note(
            family_id=family_id,
            manifest_row=row,
            remote_wsl_cwd=str(args.remote_wsl_cwd),
            remote_python=str(args.remote_python),
            health_wait_seconds=int(args.health_wait_seconds),
        )
        if bool(args.arm_remote_fast_eval_followup):
            _arm_remote_fast_eval_followup(
                family_id=family_id,
                config_path=config_path,
            )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
