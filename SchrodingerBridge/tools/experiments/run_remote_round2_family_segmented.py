from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from csv_utils import read_csv_rows, write_csv_rows
from config_schema import load_experiment_config
from round2_registry import ROUND2_PURE_SDE_SPECS
from style_families import validate_dino_retired_runtime


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round2_pure_sde" / "round2_family_manifest.csv"


def _run(cmd: list[str], *, timeout_ms: int = 120000) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_ms / 1000,
        check=False,
    )


def _validate_config(config_path: Path, *, allow_dino: bool) -> None:
    cfg = load_experiment_config(config_path)
    validate_dino_retired_runtime(
        tokenizer_family=str(getattr(cfg.model, "tokenizer_family", "legacy_factorized")),
        semantic_supervision_family=str(getattr(cfg.bridge, "semantic_supervision_family", "legacy_terminal_swd")),
        allow_dino=allow_dino,
        context="round2 pure-sde segmented launch",
    )


def _load_manifest_row(manifest_csv: Path, *, family_id: str) -> dict[str, str]:
    rows = read_csv_rows(manifest_csv)
    for row in rows:
        if str(row.get("family_id", "")).strip() == str(family_id).strip():
            return row
    raise KeyError(f"family_id not found in manifest: {family_id}")


def _write_manifest_row_updates(manifest_csv: Path, *, family_id: str, updates: dict[str, str]) -> None:
    rows = read_csv_rows(manifest_csv)
    updated = False
    for row in rows:
        if str(row.get("family_id", "")).strip() != str(family_id).strip():
            continue
        row.update(updates)
        updated = True
        break
    if not updated:
        raise KeyError(f"family_id not found in manifest during write: {family_id}")
    write_csv_rows(manifest_csv, rows)


def _family_patience(family_id: str) -> int:
    for spec in ROUND2_PURE_SDE_SPECS:
        if str(spec.family_id).strip() == str(family_id).strip():
            return int(spec.patience)
    return 4


def _active_config_abs(row: dict[str, str]) -> Path:
    raw = str(row.get("active_run_config_path", "")).strip() or str(row.get("config_path", "")).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = (WORKSPACE / path).resolve()
    return path


def _active_run_dir(row: dict[str, str]) -> str:
    value = str(row.get("active_run_dir", "")).strip() or str(row.get("run_dir", "")).strip()
    if not value:
        raise ValueError("manifest row has no run_dir / active_run_dir")
    return value


def _active_run_name(row: dict[str, str], config_abs: Path) -> str:
    value = str(row.get("active_run_name", "")).strip()
    if value:
        return value
    payload = json.loads(config_abs.read_text(encoding="utf-8"))
    training = payload.get("training") or {}
    ablation = payload.get("ablation") or {}
    checkpoint = payload.get("checkpoint") or {}
    return (
        str(ablation.get("name") or training.get("remote_log_name") or Path(str(checkpoint.get("save_dir", ""))).name or config_abs.stem).strip()
        or config_abs.stem
    )


def _remote_run_dir(*, run_dir: str, remote_workspace_root: str) -> str:
    if run_dir.startswith("./"):
        return f"{remote_workspace_root.rstrip('/')}/{run_dir[2:]}"
    return run_dir


def _scan_remote_processes(*, run_name: str, port: int, wsl_distro: str) -> dict[str, list[dict[str, str]]]:
    scan_py = f"""
from pathlib import Path
import json

token = {run_name!r}
payload = {{"train": [], "eval": []}}
for pid in Path("/proc").iterdir():
    if not pid.is_dir() or not pid.name.isdigit():
        continue
    try:
        raw = (pid / "cmdline").read_bytes()
    except Exception:
        continue
    txt = raw.replace(b"\\x00", b" ").decode("utf-8", "replace").strip()
    if token not in txt:
        continue
    item = {{"pid": pid.name, "cmd": txt}}
    if (
        "SchrodingerBridge/src/run.py" in txt
        or "src/run.py --config" in txt
        or "/src/run.py --config" in txt
    ):
        payload["train"].append(item)
    elif (
        "rerun_full_eval_for_run.py" in txt
        or "run_evaluation.py" in txt
        or "watch_round2_eval_curve.py" in txt
    ):
        payload["eval"].append(item)
print(json.dumps(payload, ensure_ascii=False))
"""
    proc = subprocess.run(
        [
            "ssh",
            "-p",
            str(int(port)),
            "administrator@100.115.18.62",
            "wsl",
            "-d",
            str(wsl_distro),
            "python3",
            "-",
        ],
        input=scan_py,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout or "remote process scan failed")
    payload = json.loads(proc.stdout.strip() or "{}")
    return {
        "train": list(payload.get("train") or []),
        "eval": list(payload.get("eval") or []),
    }


def _scan_latest_epoch(*, remote_run_dir: str, port: int, wsl_distro: str) -> tuple[int, str]:
    scan_py = """
from pathlib import Path
import json
import sys

run_dir = Path(sys.argv[1])
pts = sorted(run_dir.glob("epoch_*.pt"))
latest = pts[-1].name if pts else ""
digits = "".join(ch for ch in latest if ch.isdigit())
print(json.dumps({"latest": latest, "epoch": int(digits) if digits else 0}, ensure_ascii=False))
"""
    proc = subprocess.run(
        [
            "ssh",
            "-p",
            str(int(port)),
            "administrator@100.115.18.62",
            "wsl",
            "-d",
            str(wsl_distro),
            "python3",
            "-",
            str(remote_run_dir),
        ],
        input=scan_py,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout or "remote checkpoint scan failed")
    payload = json.loads(proc.stdout.strip() or "{}")
    return int(payload.get("epoch") or 0), str(payload.get("latest") or "")


def _write_segment_config(
    *,
    config_abs: Path,
    payload: dict[str, Any],
    latest_epoch: int,
    latest_ckpt_remote: str,
    segment_epochs: int,
) -> Path:
    target_epoch = max(1, int(latest_epoch) + int(segment_epochs))
    launch_payload = json.loads(json.dumps(payload))
    launch_payload.setdefault("training", {})
    launch_payload["training"]["num_epochs"] = int(target_epoch)
    launch_payload["training"]["save_interval"] = 1
    launch_payload["training"]["full_eval_defer_until_training_end"] = True
    launch_payload["training"]["full_eval_each_epoch"] = False
    launch_payload["training"]["resume_optimizer"] = False
    launch_payload["training"]["resume_training_state"] = True
    if str(latest_ckpt_remote).strip():
        launch_payload["training"]["resume_checkpoint"] = str(latest_ckpt_remote).strip()
    segmented_path = config_abs.with_name(config_abs.stem + ".segmented.launch.json")
    segmented_path.write_text(json.dumps(launch_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return segmented_path


def _wait_until_no_process(
    *,
    run_name: str,
    kind: str,
    port: int,
    wsl_distro: str,
    poll_seconds: int,
    max_wait_seconds: int,
) -> None:
    deadline = time.time() + max_wait_seconds
    while True:
        proc_info = _scan_remote_processes(run_name=run_name, port=port, wsl_distro=wsl_distro)
        if not proc_info.get(kind):
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for remote {kind} to finish: {run_name}")
        time.sleep(max(1, int(poll_seconds)))


def _launch_segment_train(
    *,
    segmented_config: Path,
    remote_wsl_cwd: str,
    remote_python: str,
    health_wait_seconds: int,
    max_prelaunch_memory_mib: int,
    min_runtime_memory_mib: int,
    max_runtime_memory_mib: int,
    min_runtime_slack_mib: int,
    runtime_guard_max_memory_mib: int,
    runtime_guard_poll_seconds: int,
    runtime_guard_min_memory_mib: int,
    runtime_guard_min_warmup_seconds: int,
    runtime_guard_min_consecutive_polls: int,
    runtime_guard_min_mode: str,
    skip_smoke: bool,
    smoke_device: str,
    smoke_latent_size: int,
    smoke_bank_tokens: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "launch_remote_experiment_train.py"),
        "--config",
        str(segmented_config),
        "--remote-wsl-cwd",
        str(remote_wsl_cwd),
        "--remote-python",
        str(remote_python),
        "--max-prelaunch-memory-mib",
        str(int(max_prelaunch_memory_mib)),
        "--min-runtime-memory-mib",
        str(int(min_runtime_memory_mib)),
        "--max-runtime-memory-mib",
        str(int(max_runtime_memory_mib)),
        "--min-runtime-slack-mib",
        str(int(min_runtime_slack_mib)),
        "--runtime-guard-max-memory-mib",
        str(int(runtime_guard_max_memory_mib)),
        "--runtime-guard-poll-seconds",
        str(int(runtime_guard_poll_seconds)),
        "--runtime-guard-min-memory-mib",
        str(int(runtime_guard_min_memory_mib)),
        "--runtime-guard-min-warmup-seconds",
        str(int(runtime_guard_min_warmup_seconds)),
        "--runtime-guard-min-consecutive-polls",
        str(int(runtime_guard_min_consecutive_polls)),
        "--runtime-guard-min-mode",
        str(runtime_guard_min_mode),
        "--health-wait-seconds",
        str(int(health_wait_seconds)),
    ]
    if skip_smoke:
        cmd.append("--skip-smoke")
    else:
        cmd.extend(
            [
                "--smoke-device",
                str(smoke_device),
                "--smoke-latent-size",
                str(int(smoke_latent_size)),
                "--smoke-bank-tokens",
                str(int(smoke_bank_tokens)),
            ]
        )
    proc = _run(cmd, timeout_ms=240000)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    return proc


def _launch_segment_eval(
    *,
    family_id: str,
    run_name: str,
    remote_run_dir: str,
    expected_epoch: int,
    remote_wsl_cwd: str,
    remote_python: str,
    port: int,
    wsl_distro: str,
    test_dir: str,
    cache_dir: str,
    clip_hf_cache_dir: str,
    batch_size: int,
    vae_decode_batch_size: int,
    target_chunk_size: int,
) -> subprocess.CompletedProcess[str]:
    launch = SCRIPT_DIR / "launch_remote_wsl_command.py"
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        f"round2seg-{family_id}-eval",
        "--remote-log-path",
        f"{remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_segmented_eval.log",
        "--remote-wsl-cwd",
        str(remote_wsl_cwd),
        "--python-bin",
        str(remote_python),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py",
        "--verify-python-file",
        "SchrodingerBridge/src/utils/run_evaluation.py",
        "--no-health-check",
        "--max-prelaunch-memory-mib",
        "12000",
        "--runtime-guard-max-memory-mib",
        "11000",
        "--runtime-guard-min-mode",
        "ignore",
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            f"{remote_python} SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py "
            f"--run-dir {remote_run_dir} "
            f"--python-bin {remote_python} "
            f"--test-dir {test_dir} "
            f"--cache-dir {cache_dir} "
            f"--clip-hf-cache-dir {clip_hf_cache_dir} "
            f"--batch-size {int(batch_size)} "
            f"--vae-decode-batch-size {int(vae_decode_batch_size)} "
            f"--target-chunk-size {int(target_chunk_size)} "
            f"--epochs {int(expected_epoch)} "
            "--skip-existing "
            "--output-subdir full_eval "
            "--profile-timing"
        ),
    ]
    proc = _run(command, timeout_ms=240000)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    return proc


def _refresh_local_round2_summaries(*, family_id: str, run_dir: str, patience: int, manifest_csv: Path) -> None:
    local_run_dir = Path(run_dir)
    if not local_run_dir.is_absolute():
        local_run_dir = (WORKSPACE / run_dir[2:]).resolve() if run_dir.startswith("./") else (WORKSPACE / run_dir).resolve()
    collect = _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "collect_round2_eval_curve.py"),
            "--run-dir",
            str(local_run_dir),
            "--eval-subdir",
            "full_eval",
        ],
        timeout_ms=240000,
    )
    sys.stdout.write(collect.stdout)
    sys.stdout.flush()
    curve_csv = local_run_dir / "full_eval" / "clip_lpips_curve.csv"
    if curve_csv.is_file():
        conv = _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "report_round2_convergence.py"),
                "--curve-csv",
                str(curve_csv),
                "--patience",
                str(int(patience)),
            ],
            timeout_ms=240000,
        )
        sys.stdout.write(conv.stdout)
        sys.stdout.flush()
    upd = _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "update_round2_family_manifest.py"),
            "--manifest-csv",
            str(manifest_csv),
            "--family-id",
            str(family_id),
        ],
        timeout_ms=240000,
    )
    sys.stdout.write(upd.stdout)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a round-2 family in segmented remote train/eval mode.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--segment-epochs", type=int, default=1)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--allow-dino", action="store_true", help="Override the default round2 policy that archives DINO-conditioned configs.")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-train-wait-seconds", type=int, default=10800)
    parser.add_argument("--max-eval-wait-seconds", type=int, default=7200)
    parser.add_argument("--health-wait-seconds", type=int, default=30)
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
    parser.add_argument("--skip-fast-eval", action="store_true")
    parser.add_argument("--skip-smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-device", default="cpu")
    parser.add_argument("--smoke-latent-size", type=int, default=32)
    parser.add_argument("--smoke-bank-tokens", type=int, default=8)
    parser.add_argument("--test-dir", default="/mnt/i/wikiart_distinct5_samam_512_classview/test")
    parser.add_argument("--cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache")
    parser.add_argument("--clip-hf-cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache/hf")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-vae-decode-batch-size", type=int, default=4)
    parser.add_argument("--eval-target-chunk-size", type=int, default=1)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).resolve()
    row = _load_manifest_row(manifest_csv, family_id=str(args.family_id))
    config_abs = _active_config_abs(row)
    _validate_config(config_abs, allow_dino=bool(args.allow_dino))
    run_dir = _active_run_dir(row)
    run_name = _active_run_name(row, config_abs)
    remote_run_dir = _remote_run_dir(run_dir=run_dir, remote_workspace_root=str(args.remote_workspace_root))
    payload = json.loads(config_abs.read_text(encoding="utf-8"))

    latest_epoch, latest_name = _scan_latest_epoch(
        remote_run_dir=remote_run_dir,
        port=int(args.port),
        wsl_distro=str(args.wsl_distro),
    )
    latest_ckpt_remote = f"{remote_run_dir.rstrip('/')}/{latest_name}" if latest_name else ""
    print(
        f"[run_remote_round2_family_segmented] latest_epoch={latest_epoch} latest_ckpt={latest_name or 'none'}",
        flush=True,
    )

    segmented_config = _write_segment_config(
        config_abs=config_abs,
        payload=payload,
        latest_epoch=latest_epoch,
        latest_ckpt_remote=latest_ckpt_remote,
        segment_epochs=int(args.segment_epochs),
    )
    _validate_config(segmented_config, allow_dino=bool(args.allow_dino))
    expected_epoch = max(int(latest_epoch) + int(args.segment_epochs), 1)
    _write_manifest_row_updates(
        manifest_csv,
        family_id=str(args.family_id),
        updates={
            "decision_status": "calibration_running",
            "active_run_config_path": str(segmented_config),
            "active_run_name": run_name,
            "active_run_dir": run_dir,
            "active_run_batch_size": str((payload.get("training") or {}).get("batch_size", "")),
            "active_resume_checkpoint": latest_ckpt_remote,
        },
    )

    launch_train = _launch_segment_train(
        segmented_config=segmented_config,
        remote_wsl_cwd=str(args.remote_wsl_cwd),
        remote_python=str(args.remote_python),
        health_wait_seconds=int(args.health_wait_seconds),
        max_prelaunch_memory_mib=int(args.max_prelaunch_memory_mib),
        min_runtime_memory_mib=int(args.min_runtime_memory_mib),
        max_runtime_memory_mib=int(args.max_runtime_memory_mib),
        min_runtime_slack_mib=int(args.min_runtime_slack_mib),
        runtime_guard_max_memory_mib=int(args.runtime_guard_max_memory_mib),
        runtime_guard_poll_seconds=int(args.runtime_guard_poll_seconds),
        runtime_guard_min_memory_mib=int(args.runtime_guard_min_memory_mib),
        runtime_guard_min_warmup_seconds=int(args.runtime_guard_min_warmup_seconds),
        runtime_guard_min_consecutive_polls=int(args.runtime_guard_min_consecutive_polls),
        runtime_guard_min_mode=str(args.runtime_guard_min_mode),
        skip_smoke=bool(args.skip_smoke),
        smoke_device=str(args.smoke_device),
        smoke_latent_size=int(args.smoke_latent_size),
        smoke_bank_tokens=int(args.smoke_bank_tokens),
    )
    if launch_train.returncode != 0:
        return int(launch_train.returncode)

    print("[run_remote_round2_family_segmented] train launched; waiting for remote train to finish", flush=True)
    _wait_until_no_process(
        run_name=run_name,
        kind="train",
        port=int(args.port),
        wsl_distro=str(args.wsl_distro),
        poll_seconds=int(args.poll_seconds),
        max_wait_seconds=int(args.max_train_wait_seconds),
    )

    latest_epoch_after, latest_name_after = _scan_latest_epoch(
        remote_run_dir=remote_run_dir,
        port=int(args.port),
        wsl_distro=str(args.wsl_distro),
    )
    print(
        f"[run_remote_round2_family_segmented] latest_epoch_after_train={latest_epoch_after} latest_ckpt_after_train={latest_name_after or 'none'}",
        flush=True,
    )
    if int(latest_epoch_after) < int(expected_epoch):
        print(
            "[run_remote_round2_family_segmented] no new retained checkpoint landed during the bounded segment; "
            "skip remote eval launch for this cycle",
            flush=True,
        )
        return 26

    if not bool(args.skip_fast_eval):
        launch_eval = _launch_segment_eval(
            family_id=str(args.family_id),
            run_name=run_name,
            remote_run_dir=remote_run_dir,
            expected_epoch=int(expected_epoch),
            remote_wsl_cwd=str(args.remote_wsl_cwd),
            remote_python=str(args.remote_python),
            port=int(args.port),
            wsl_distro=str(args.wsl_distro),
            test_dir=str(args.test_dir),
            cache_dir=str(args.cache_dir),
            clip_hf_cache_dir=str(args.clip_hf_cache_dir),
            batch_size=int(args.eval_batch_size),
            vae_decode_batch_size=int(args.eval_vae_decode_batch_size),
            target_chunk_size=int(args.eval_target_chunk_size),
        )
        if launch_eval.returncode != 0:
            return int(launch_eval.returncode)
        print("[run_remote_round2_family_segmented] eval launched; waiting for remote eval to finish", flush=True)
        _wait_until_no_process(
            run_name=run_name,
            kind="eval",
            port=int(args.port),
            wsl_distro=str(args.wsl_distro),
            poll_seconds=int(args.poll_seconds),
            max_wait_seconds=int(args.max_eval_wait_seconds),
        )
        _refresh_local_round2_summaries(
            family_id=str(args.family_id),
            run_dir=run_dir,
            patience=_family_patience(str(args.family_id)),
            manifest_csv=manifest_csv,
        )
        print("[run_remote_round2_family_segmented] round2 eval summary refreshed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
