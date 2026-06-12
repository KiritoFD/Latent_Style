from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round2_pure_sde" / "round2_family_manifest.csv"
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_experiment_config
from style_families import validate_dino_retired_runtime


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("[launch_remote_round2_followon_train] " + " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _validate_config(config_path: Path, *, allow_dino: bool) -> None:
    cfg = load_experiment_config(config_path)
    validate_dino_retired_runtime(
        tokenizer_family=str(getattr(cfg.model, "tokenizer_family", "legacy_factorized")),
        semantic_supervision_family=str(getattr(cfg.bridge, "semantic_supervision_family", "legacy_terminal_swd")),
        allow_dino=allow_dino,
        context="round2 pure-sde follow-on launch",
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fieldnames(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            names.append(key)
    return names


def _write_rows(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _launch_metadata(config_path: Path) -> dict[str, str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    training = payload.get("training") or {}
    checkpoint = payload.get("checkpoint") or {}
    ablation = payload.get("ablation") or {}
    run_name = str(ablation.get("name") or training.get("remote_log_name") or config_path.stem).strip()
    run_dir = str(checkpoint.get("save_dir") or "").strip()
    batch_size = str(training.get("batch_size", "")).strip()
    resume_checkpoint = str(training.get("resume_checkpoint", "")).strip()
    return {
        "active_run_config_path": str(config_path),
        "active_run_name": run_name,
        "active_run_dir": run_dir,
        "active_run_batch_size": batch_size,
        "active_resume_checkpoint": resume_checkpoint,
    }


def _update_manifest_active_run(
    *,
    manifest_csv: Path,
    family_id: str,
    launch_config: Path,
) -> None:
    rows = _read_rows(manifest_csv)
    launch_fields = _launch_metadata(launch_config)
    updated = False
    active_keys = tuple(launch_fields.keys())
    for row in rows:
        row_family_id = str(row.get("family_id", "")).strip()
        if row_family_id == str(family_id).strip():
            row.update(launch_fields)
            row["decision_status"] = "calibration_running"
            updated = True
            continue
        else:
            had_foreign_active = any(str(row.get(key, "")).strip() for key in active_keys)
            if had_foreign_active:
                for key in active_keys:
                    row[key] = ""
                if str(row.get("decision_status", "")).strip() == "calibration_running":
                    row["decision_status"] = "reference_point_recorded"
            continue
    if not updated:
        raise KeyError(f"family_id not found in manifest: {family_id}")
    fieldnames = _fieldnames(rows)
    for key in launch_fields.keys():
        if key not in fieldnames:
            fieldnames.append(key)
    _write_rows(manifest_csv, rows, fieldnames=fieldnames)


def _prepare_followon(
    *,
    manifest_csv: Path,
    winner_family_id: str,
    winner_checkpoint: str,
    winner_checkpoint_mode: str,
    target_wave: str,
    target_family_id: str,
    remote_wsl_cwd: str,
) -> Path:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_round2_followon_configs.py"),
        "--manifest-csv",
        str(manifest_csv),
        "--winner-family-id",
        str(winner_family_id),
        "--winner-checkpoint-mode",
        str(winner_checkpoint_mode),
        "--target-wave",
        str(target_wave),
        "--target-family-id",
        str(target_family_id),
        "--remote-wsl-cwd",
        str(remote_wsl_cwd),
    ]
    if str(winner_checkpoint).strip():
        cmd.extend(["--winner-checkpoint", str(winner_checkpoint).strip()])
    proc = _run(cmd)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout or "prepare_round2_followon_configs.py failed")
    produced = [
        Path(line.strip())
        for line in proc.stdout.splitlines()
        if line.strip().lower().endswith(".json")
    ]
    if not produced:
        raise RuntimeError("No follow-on config path was reported by prepare_round2_followon_configs.py")
    return produced[0].resolve()


def _apply_launch_overrides(
    *,
    config_path: Path,
    batch_size_override: int | None,
    run_suffix: str,
) -> Path:
    if batch_size_override is None and not str(run_suffix).strip():
        return config_path
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if batch_size_override is not None:
        payload.setdefault("training", {})
        payload["training"]["batch_size"] = int(batch_size_override)
    suffix = str(run_suffix).strip()
    if suffix:
        suffix_token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in suffix)
        payload.setdefault("training", {})
        if str(payload["training"].get("remote_log_name", "")).strip():
            payload["training"]["remote_log_name"] = f"{payload['training']['remote_log_name']}_{suffix_token}"
        payload.setdefault("checkpoint", {})
        save_dir = str(payload["checkpoint"].get("save_dir", "")).strip()
        if save_dir:
            if save_dir.endswith("/"):
                save_dir = save_dir[:-1]
            payload["checkpoint"]["save_dir"] = f"{save_dir}_{suffix_token}"
        payload.setdefault("ablation", {})
        ablation_name = str(payload["ablation"].get("name", "")).strip()
        if ablation_name:
            payload["ablation"]["name"] = f"{ablation_name}_{suffix_token}"
    target = config_path.with_name(config_path.stem + (f"_{suffix}" if suffix else "") + ".launch.json")
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and optionally launch one round-2 follow-on config warm-started from a tokenizer winner checkpoint."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--winner-family-id", required=True)
    parser.add_argument("--winner-checkpoint", default="")
    parser.add_argument(
        "--winner-checkpoint-mode",
        choices=["latest", "best_transfer", "best_all_pairs"],
        default="latest",
    )
    parser.add_argument("--target-wave", default="wave2_sde_noise")
    parser.add_argument("--target-family-id", required=True)
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--allow-dino", action="store_true", help="Override the default round2 policy that archives DINO-conditioned configs.")
    parser.add_argument("--batch-size-override", type=int, default=None)
    parser.add_argument("--run-suffix", default="")
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
    parser.add_argument("--runtime-guard-min-mode", choices=["ignore", "warn", "stop"], default="warn")
    parser.add_argument("--health-wait-seconds", type=int, default=20)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()

    followon_config = _prepare_followon(
        manifest_csv=manifest_csv,
        winner_family_id=str(args.winner_family_id),
        winner_checkpoint=str(args.winner_checkpoint),
        winner_checkpoint_mode=str(args.winner_checkpoint_mode),
        target_wave=str(args.target_wave),
        target_family_id=str(args.target_family_id),
        remote_wsl_cwd=str(args.remote_wsl_cwd),
    )
    _validate_config(followon_config, allow_dino=bool(args.allow_dino))

    launch_config = _apply_launch_overrides(
        config_path=followon_config,
        batch_size_override=args.batch_size_override,
        run_suffix=str(args.run_suffix),
    )
    _validate_config(launch_config, allow_dino=bool(args.allow_dino))
    _update_manifest_active_run(
        manifest_csv=manifest_csv,
        family_id=str(args.target_family_id),
        launch_config=launch_config,
    )

    launch_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "launch_remote_experiment_train.py"),
        "--config",
        str(launch_config),
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--remote-python",
        str(args.remote_python),
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--min-runtime-memory-mib",
        str(int(args.min_runtime_memory_mib)),
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
        launch_cmd.append("--skip-smoke")
    else:
        launch_cmd.extend(
            [
                "--smoke-device",
                str(args.smoke_device),
                "--smoke-latent-size",
                str(int(args.smoke_latent_size)),
                "--smoke-bank-tokens",
                str(int(args.smoke_bank_tokens)),
            ]
        )
    proc = _run(launch_cmd)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
