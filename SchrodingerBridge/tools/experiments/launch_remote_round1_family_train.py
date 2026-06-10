from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from csv_utils import manifest_fieldnames, read_csv_rows, write_csv_rows
from dino_cache_utils import default_dino_cache_output, inspect_dino_cache
from round1_paths import infer_round1_family_id


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"


def _run(cmd: list[str]) -> int:
    print("[launch_remote_round1_family_train] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _run_round1_switch_smoke(
    *,
    family_id: str,
    device: str,
    latent_size: int,
    bank_tokens: int,
) -> int:
    smoke_script = SCRIPT_DIR / "smoke_round1_family_switches.py"
    output_path = SB_ROOT / "aaai2027" / f"round1_{family_id}_switch_smoke_latest.json"
    cmd = [
        sys.executable,
        str(smoke_script),
        "--family-id",
        str(family_id),
        "--device",
        str(device),
        "--latent-size",
        str(max(8, int(latent_size))),
        "--bank-tokens",
        str(max(1, int(bank_tokens))),
        "--output",
        str(output_path),
    ]
    print(f"[launch_remote_round1_family_train] prelaunch switch smoke -> {output_path}", flush=True)
    proc = subprocess.run(cmd, check=False, cwd=str(WORKSPACE))
    return int(proc.returncode)


def _read_json_optional(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _update_manifest_switch_smoke(
    *,
    manifest_csv: Path,
    family_id: str,
    artifact_path: Path,
    payload: dict | None,
    rc: int,
) -> None:
    if not manifest_csv.exists():
        return
    rows = read_csv_rows(manifest_csv)
    if not rows:
        return
    target = next((row for row in rows if str(row.get("family_id", "")).strip() == str(family_id).strip()), None)
    if target is None:
        return
    status = "failed" if int(rc) != 0 else "ok"
    row_count = ""
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            row_count = str(len(payload.get("results") or []))
            for result in payload.get("results") or []:
                if str((result or {}).get("family_id", "")).strip() != str(family_id).strip():
                    continue
                payload_status = str((result or {}).get("status", "")).strip().lower()
                if payload_status:
                    status = payload_status
                break
        elif payload.get("row_count") is not None:
            row_count = str(payload.get("row_count", "")).strip()
        payload_status = str(payload.get("status", "")).strip().lower()
        if payload_status:
            status = payload_status
    target["switch_smoke_status"] = status
    target["switch_smoke_artifact"] = str(artifact_path)
    target["switch_smoke_row_count"] = row_count
    write_csv_rows(manifest_csv, rows, fieldnames=manifest_fieldnames(rows))


def _update_manifest_decision_status(
    *,
    manifest_csv: Path,
    family_id: str,
    decision_status: str,
) -> None:
    if not manifest_csv.exists():
        return
    rows = read_csv_rows(manifest_csv)
    if not rows:
        return
    target = next((row for row in rows if str(row.get("family_id", "")).strip() == str(family_id).strip()), None)
    if target is None:
        return
    target["decision_status"] = str(decision_status).strip()
    write_csv_rows(manifest_csv, rows, fieldnames=manifest_fieldnames(rows))


def _refresh_round1_family_status_docs(*, family_id: str, manifest_csv: Path) -> None:
    updater = SCRIPT_DIR / "update_round1_family_status_docs.py"
    if not updater.exists():
        return
    cmd = [
        sys.executable,
        str(updater),
        "--family-id",
        str(family_id),
        "--manifest-csv",
        str(manifest_csv),
    ]
    print("[launch_remote_round1_family_train] refresh family status docs -> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False, cwd=str(WORKSPACE))


def _arm_runtime_watch_followup(*, config_rel: Path, manifest_csv: Path) -> None:
    helper = SCRIPT_DIR / "launch_round1_family_followups_detached.py"
    if not helper.exists():
        return
    cmd = [
        sys.executable,
        str(helper),
        "--config",
        str(config_rel),
        "--manifest-csv",
        str(manifest_csv),
        "--skip-fast-eval-deferred",
        "--skip-stageclose-deferred",
    ]
    print("[launch_remote_round1_family_train] arm runtime followup -> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False, cwd=str(WORKSPACE))


def _arm_remote_fast_eval_followup(*, config_rel: Path, manifest_csv: Path) -> None:
    helper = SCRIPT_DIR / "launch_remote_round1_family_fast_eval.py"
    if not helper.exists():
        return
    cmd = [
        sys.executable,
        str(helper),
        "--config",
        str(config_rel),
        "--manifest-csv",
        str(manifest_csv),
    ]
    print("[launch_remote_round1_family_train] arm remote fast-eval followup -> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False, cwd=str(WORKSPACE))


def _local_path_from_wsl_mount(text: str) -> Path:
    raw = str(text).strip()
    if raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5].upper()
        remainder = raw[7:].replace("/", "\\")
        return Path(f"{drive}:\\{remainder}") if remainder else Path(f"{drive}:\\")
    return Path(raw)


def _wsl_mount_from_local_path(path: Path) -> str:
    text = str(path)
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        remainder = text[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{remainder}" if remainder else f"/mnt/{drive}"
    return text.replace("\\", "/")


def _validate_dino_cache_for_config(*, cache_path: Path, payload: dict, workspace_root: Path) -> Path:
    data_cfg = payload.get("data") or {}
    style_subdirs = [str(x).strip() for x in data_cfg.get("style_subdirs", []) if str(x).strip()]
    latent_root = Path(str(data_cfg.get("data_root", "")).strip())
    if not cache_path.exists():
        suggested = default_dino_cache_output(latent_root, workspace_root=workspace_root)
        raise FileNotFoundError(
            f"DINO cache not found: {cache_path}. "
            f"Build a matching cache first, e.g. {suggested}"
        )
    meta = inspect_dino_cache(cache_path)
    cache_styles = [str(x).strip() for x in meta.get("styles", []) if str(x).strip()]
    if style_subdirs and cache_styles and sorted(cache_styles) != sorted(style_subdirs):
        suggested = default_dino_cache_output(latent_root, workspace_root=workspace_root)
        raise RuntimeError(
            "DINO cache style set mismatch. "
            f"config={style_subdirs} cache={cache_styles} cache_path={cache_path}. "
            f"Build and use a matching cache, e.g. {suggested}"
        )
    return cache_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a round-1 family training run on the remote 3060 WSL host.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=7000)
    parser.add_argument("--min-runtime-memory-mib", type=int, default=9216)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11059)
    parser.add_argument("--min-runtime-slack-mib", type=int, default=128)
    parser.add_argument("--runtime-guard-max-memory-mib", type=int, default=11571)
    parser.add_argument("--runtime-guard-poll-seconds", type=int, default=10)
    parser.add_argument("--runtime-guard-min-memory-mib", type=int, default=9216)
    parser.add_argument("--runtime-guard-min-warmup-seconds", type=int, default=300)
    parser.add_argument("--runtime-guard-min-consecutive-polls", type=int, default=3)
    parser.add_argument("--runtime-guard-min-mode", choices=["ignore", "warn", "stop"], default="stop")
    parser.add_argument("--dino-cache-override", default="")
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--allow-other-running", action="store_true")
    parser.add_argument("--skip-switch-smoke", action="store_true")
    parser.add_argument("--switch-smoke-device", default="cpu")
    parser.add_argument("--switch-smoke-latent-size", type=int, default=32)
    parser.add_argument("--switch-smoke-bank-tokens", type=int, default=8)
    parser.add_argument("--skip-remote-fast-eval-followup", action="store_true")
    args = parser.parse_args()

    config_rel = Path(args.config)
    config_abs = (WORKSPACE / config_rel).resolve()
    payload = json.loads(config_abs.read_text(encoding="utf-8"))
    run_name = str((payload.get("ablation") or {}).get("name", config_abs.stem)).strip() or config_abs.stem
    family_id = infer_round1_family_id(run_name=run_name, config_stem=config_abs.stem) or config_abs.stem
    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    if not bool(args.skip_switch_smoke):
        smoke_artifact = SB_ROOT / "aaai2027" / f"round1_{family_id}_switch_smoke_latest.json"
        smoke_rc = _run_round1_switch_smoke(
            family_id=family_id,
            device=str(args.switch_smoke_device),
            latent_size=int(args.switch_smoke_latent_size),
            bank_tokens=int(args.switch_smoke_bank_tokens),
        )
        smoke_payload = _read_json_optional(smoke_artifact)
        _update_manifest_switch_smoke(
            manifest_csv=manifest_csv,
            family_id=family_id,
            artifact_path=smoke_artifact,
            payload=smoke_payload,
            rc=smoke_rc,
        )
        _refresh_round1_family_status_docs(
            family_id=family_id,
            manifest_csv=manifest_csv,
        )
        if smoke_rc != 0:
            raise RuntimeError(
                f"Refusing remote launch because prelaunch switch smoke failed for family={family_id} "
                f"(rc={smoke_rc})."
            )
    if manifest_csv.is_file() and not bool(args.allow_other_running):
        rows = read_csv_rows(manifest_csv)
        foreign_running = [
            str(row.get("family_id", "")).strip()
            for row in rows
            if str(row.get("decision_status", "")).strip().lower() == "running"
            and str(row.get("family_id", "")).strip() != str(family_id).strip()
        ]
        if foreign_running:
            raise RuntimeError(
                "Refusing direct launch because other running round-1 families exist: "
                + ", ".join(foreign_running)
            )
    auto_dino_override = str(args.dino_cache_override).strip()
    data_cfg = payload.get("data") or {}
    if (not auto_dino_override) and bool(data_cfg.get("dino_cache_required", False)):
        current_dino = str(data_cfg.get("dino_cache_path", "")).strip()
        if current_dino:
            local_cache_path = _local_path_from_wsl_mount(current_dino)
            local_cache_path = _validate_dino_cache_for_config(cache_path=local_cache_path, payload=payload, workspace_root=WORKSPACE)
            if not current_dino.startswith("/mnt/"):
                auto_dino_override = _wsl_mount_from_local_path(local_cache_path)
    if auto_dino_override:
        payload.setdefault("data", {})
        payload["data"]["dino_cache_path"] = auto_dino_override
        payload["data"]["dino_cache_required"] = True
        rewritten_rel = config_rel.parent / f"{config_abs.stem}.remote.launch.json"
        rewritten_abs = (WORKSPACE / rewritten_rel).resolve()
        rewritten_abs.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        sync_config = rewritten_rel
        remote_config = f"{args.remote_wsl_cwd.rstrip('/')}/{rewritten_rel.as_posix()}"
    else:
        sync_config = config_rel
        remote_config = f"{args.remote_wsl_cwd.rstrip('/')}/{config_rel.as_posix()}"

    health_wait_seconds = int(args.health_wait_seconds)
    if bool((payload.get("data") or {}).get("dino_cache_required", False)):
        health_wait_seconds = max(health_wait_seconds, 90)
    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        f"round1-{run_name}-train",
        "--remote-log-path",
        f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_train.log",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/launch_remote_round1_family_train.py",
        "--sync-path",
        "SchrodingerBridge/docs/experiments/2026-06-10-round1-full-sweep-master.md",
        "--sync-path",
        "SchrodingerBridge/docs/experiments/round1_full_sweep",
        "--sync-path",
        str(sync_config),
        "--verify-python-file",
        "SchrodingerBridge/src/run.py",
        "--verify-python-file",
        "SchrodingerBridge/src/losses.py",
        "--verify-python-file",
        "SchrodingerBridge/src/model.py",
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--health-wait-seconds",
        str(int(health_wait_seconds)),
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
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            f"{args.remote_python} SchrodingerBridge/src/run.py --config {remote_config}"
        ),
    ]
    rc = _run(command)
    if rc == 0:
        _update_manifest_decision_status(
            manifest_csv=manifest_csv,
            family_id=family_id,
            decision_status="running",
        )
        _refresh_round1_family_status_docs(
            family_id=family_id,
            manifest_csv=manifest_csv,
        )
        _arm_runtime_watch_followup(
            config_rel=config_rel,
            manifest_csv=manifest_csv,
        )
        if not bool(args.skip_remote_fast_eval_followup):
            _arm_remote_fast_eval_followup(
                config_rel=config_rel,
                manifest_csv=manifest_csv,
            )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
