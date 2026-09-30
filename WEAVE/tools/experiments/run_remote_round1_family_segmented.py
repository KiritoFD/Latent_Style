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

from csv_utils import read_csv_rows
from round1_paths import infer_round1_family_id


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"


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


def _load_manifest_row(manifest_csv: Path, *, family_id: str) -> dict[str, str]:
    rows = read_csv_rows(manifest_csv)
    for row in rows:
        if str(row.get("family_id", "")).strip() == str(family_id).strip():
            return row
    raise KeyError(f"family_id not found in manifest: {family_id}")


def _write_manifest_status(manifest_csv: Path, *, family_id: str, decision_status: str) -> None:
    rows = read_csv_rows(manifest_csv)
    updated = False
    for row in rows:
        if str(row.get("family_id", "")).strip() != str(family_id).strip():
            continue
        row["decision_status"] = str(decision_status).strip()
        updated = True
        break
    if not updated:
        raise KeyError(f"family_id not found in manifest during write: {family_id}")
    if rows:
        fieldnames = list(rows[0].keys())
        with manifest_csv.open("w", encoding="utf-8", newline="") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _remote_run_dir(*, row: dict[str, str], remote_workspace_root: str) -> str:
    run_dir = str(row.get("run_dir", "")).strip()
    if run_dir.startswith("./"):
        return f"{remote_workspace_root.rstrip('/')}/{run_dir[2:]}"
    return run_dir


def _scan_remote_processes(*, run_name: str, host: str, port: int, wsl_distro: str) -> dict[str, list[dict[str, str]]]:
    scan_py = f"""
from pathlib import Path
import json

token = {run_name!r}
payload = {{"train": [], "fast_eval": []}}
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
        "watch_round1_family_fast_eval.py" in txt
        or "rerun_full_eval_for_run.py" in txt
        or "run_evaluation.py" in txt
        or "fast-eval.sh" in txt
        or "_fast_eval" in txt
        or "_fast-eval" in txt
    ):
        payload["fast_eval"].append(item)
    elif (
        "SchrodingerBridge/src/run.py" in txt
        or "src/run.py --config" in txt
        or "/src/run.py --config" in txt
    ):
        payload["train"].append(item)
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
        "fast_eval": list(payload.get("fast_eval") or []),
    }


def _stop_remote_fast_eval_processes(*, run_name: str, host: str, port: int, wsl_distro: str) -> list[str]:
    proc_info = _scan_remote_processes(run_name=run_name, host=host, port=port, wsl_distro=wsl_distro)
    fast_rows = list(proc_info.get("fast_eval") or [])
    killed: list[str] = []
    for row in fast_rows:
        pid = str(row.get("pid", "")).strip()
        if not pid.isdigit():
            continue
        stop_py = f"""
import os
try:
    os.kill({int(pid)}, 15)
    print({pid!r})
except Exception:
    pass
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
            input=stop_py,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        text = proc.stdout.strip()
        if text:
            killed.append(text)
    return killed


def _scan_latest_epoch(*, remote_run_dir: str, host: str, port: int, wsl_distro: str) -> tuple[int, str]:
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


def _write_segment_config(*, config_abs: Path, payload: dict[str, Any], latest_epoch: int, latest_ckpt_remote: str, segment_epochs: int) -> Path:
    target_epoch = max(1, int(latest_epoch) + int(segment_epochs))
    launch_payload = json.loads(json.dumps(payload))
    launch_payload.setdefault("training", {})
    launch_payload["training"]["num_epochs"] = int(target_epoch)
    launch_payload["training"]["save_interval"] = 1
    launch_payload["training"]["full_eval_defer_until_training_end"] = True
    launch_payload["training"]["full_eval_each_epoch"] = False
    if str(latest_ckpt_remote).strip():
        launch_payload["training"]["resume_checkpoint"] = str(latest_ckpt_remote).strip()
        launch_payload["training"]["resume_training_state"] = True
        # Segmented continuation may change effective optimizer groups after
        # freeze-mode filtering or other config evolution. Resume model/time
        # state, but rebuild optimizer state fresh unless the trainer can prove
        # the saved optimizer layout still matches.
        launch_payload["training"]["resume_optimizer"] = False
    segmented_path = config_abs.with_name(config_abs.stem + ".segmented.launch.json")
    segmented_path.write_text(json.dumps(launch_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return segmented_path


def _wait_until_no_train(*, run_name: str, host: str, port: int, wsl_distro: str, poll_seconds: int, max_wait_seconds: int) -> None:
    deadline = time.time() + max_wait_seconds
    while True:
        proc_info = _scan_remote_processes(run_name=run_name, host=host, port=port, wsl_distro=wsl_distro)
        if not proc_info["train"]:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for remote train to finish: {run_name}")
        time.sleep(max(1, int(poll_seconds)))


def _launch_train_with_retry(
    *,
    segmented_config: Path,
    poll_seconds: int,
    max_wait_seconds: int,
    health_wait_seconds: int,
    max_prelaunch_memory_mib: int,
    min_runtime_memory_mib: int,
    max_runtime_memory_mib: int,
    min_runtime_slack_mib: int,
    runtime_guard_min_mode: str,
) -> subprocess.CompletedProcess[str]:
    deadline = time.time() + max_wait_seconds
    while True:
        launch_train = _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "launch_remote_round1_family_train.py"),
                "--config",
                str(segmented_config.relative_to(WORKSPACE)),
                "--skip-remote-fast-eval-followup",
                "--health-wait-seconds",
                str(int(health_wait_seconds)),
                "--max-prelaunch-memory-mib",
                str(int(max_prelaunch_memory_mib)),
                "--min-runtime-memory-mib",
                str(int(min_runtime_memory_mib)),
                "--max-runtime-memory-mib",
                str(int(max_runtime_memory_mib)),
                "--min-runtime-slack-mib",
                str(int(min_runtime_slack_mib)),
                "--runtime-guard-min-mode",
                str(runtime_guard_min_mode),
            ],
            timeout_ms=240000,
        )
        sys.stdout.write(launch_train.stdout)
        sys.stdout.flush()
        if launch_train.returncode == 0:
            return launch_train
        if launch_train.returncode != 13:
            return launch_train
        if time.time() >= deadline:
            return launch_train
        print(
            f"[run_remote_round1_family_segmented] remote GPU not idle enough yet; retrying in {int(poll_seconds)}s",
            flush=True,
        )
        time.sleep(max(1, int(poll_seconds)))


def _wait_for_eval_summary(*, family_id: str, expected_epoch: int, poll_seconds: int, max_wait_seconds: int) -> None:
    deadline = time.time() + max_wait_seconds
    sync_script = SCRIPT_DIR / "sync_round1_remote_fast_eval_packet.py"
    summary_path = SB_ROOT / "aaai2027" / f"round1_{family_id}_remote_full_eval_pull" / "sync_summary.json"
    while True:
        proc = _run([sys.executable, str(sync_script), "--family-id", family_id], timeout_ms=180000)
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout or "fast-eval sync failed")
        if summary_path.is_file():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            latest = str(((payload.get("latest") or {}).get("epoch")) or "").strip()
            digits = "".join(ch for ch in latest if ch.isdigit())
            if digits and int(digits) >= int(expected_epoch):
                return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for remote fast-eval summary for epoch >= {expected_epoch}")
        time.sleep(max(1, int(poll_seconds)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a round-1 family in segmented remote train/eval mode.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--segment-epochs", type=int, default=1)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--host", default="administrator@100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-train-wait-seconds", type=int, default=10800)
    parser.add_argument("--max-eval-wait-seconds", type=int, default=7200)
    parser.add_argument("--health-wait-seconds", type=int, default=120)
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=7000)
    parser.add_argument("--min-runtime-memory-mib", type=int, default=9216)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11059)
    parser.add_argument("--min-runtime-slack-mib", type=int, default=128)
    parser.add_argument("--runtime-guard-min-mode", choices=["ignore", "warn", "stop"], default="stop")
    parser.add_argument(
        "--manifest-decision-status-on-start",
        default="running",
        help="Decision status to write into the round1 manifest when this segmented controller starts.",
    )
    parser.add_argument(
        "--manifest-decision-status-on-exit",
        default="recalibration_needed",
        help="Decision status to write back when this bounded segmented controller exits.",
    )
    parser.add_argument("--skip-fast-eval", action="store_true")
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).resolve()
    print(
        f"[run_remote_round1_family_segmented] start family_id={args.family_id} segment_epochs={int(args.segment_epochs)}",
        flush=True,
    )
    row = _load_manifest_row(manifest_csv, family_id=str(args.family_id))
    config_abs = Path(str(row["config_path"])).resolve()
    payload = json.loads(config_abs.read_text(encoding="utf-8"))
    run_name = str((payload.get("ablation") or {}).get("name", config_abs.stem)).strip() or config_abs.stem
    remote_run_dir = _remote_run_dir(row=row, remote_workspace_root=str(args.remote_workspace_root))

    latest_epoch, latest_name = _scan_latest_epoch(
        remote_run_dir=remote_run_dir,
        host=str(args.host),
        port=int(args.port),
        wsl_distro=str(args.wsl_distro),
    )
    latest_ckpt_remote = ""
    if latest_name:
        latest_ckpt_remote = f"{remote_run_dir.rstrip('/')}/{latest_name}"
    print(
        f"[run_remote_round1_family_segmented] latest_epoch={latest_epoch} latest_ckpt={latest_name or 'none'}",
        flush=True,
    )

    segmented_config = _write_segment_config(
        config_abs=config_abs,
        payload=payload,
        latest_epoch=latest_epoch,
        latest_ckpt_remote=latest_ckpt_remote,
        segment_epochs=int(args.segment_epochs),
    )
    expected_epoch = max(int(latest_epoch) + int(args.segment_epochs), 1)
    previous_status = str(row.get("decision_status", "")).strip() or "recalibration_needed"
    exit_status = str(args.manifest_decision_status_on_exit).strip() or previous_status
    _write_manifest_status(
        manifest_csv,
        family_id=str(args.family_id),
        decision_status=str(args.manifest_decision_status_on_start),
    )
    print(
        f"[run_remote_round1_family_segmented] expected_epoch>={expected_epoch}",
        flush=True,
    )

    killed_fast_eval = _stop_remote_fast_eval_processes(
        run_name=run_name,
        host=str(args.host),
        port=int(args.port),
        wsl_distro=str(args.wsl_distro),
    )
    if killed_fast_eval:
        print(
            "[run_remote_round1_family_segmented] stopped same-run fast-eval pids before next train segment: "
            + ", ".join(killed_fast_eval),
            flush=True,
        )

    launch_train = _launch_train_with_retry(
        segmented_config=segmented_config,
        poll_seconds=int(args.poll_seconds),
        max_wait_seconds=int(args.max_train_wait_seconds),
        health_wait_seconds=int(args.health_wait_seconds),
        max_prelaunch_memory_mib=int(args.max_prelaunch_memory_mib),
        min_runtime_memory_mib=int(args.min_runtime_memory_mib),
        max_runtime_memory_mib=int(args.max_runtime_memory_mib),
        min_runtime_slack_mib=int(args.min_runtime_slack_mib),
        runtime_guard_min_mode=str(args.runtime_guard_min_mode),
    )
    if launch_train.returncode != 0:
        _write_manifest_status(
            manifest_csv,
            family_id=str(args.family_id),
            decision_status=exit_status,
        )
        return int(launch_train.returncode)

    print("[run_remote_round1_family_segmented] train launched; waiting for remote train to finish", flush=True)
    _wait_until_no_train(
        run_name=run_name,
        host=str(args.host),
        port=int(args.port),
        wsl_distro=str(args.wsl_distro),
        poll_seconds=int(args.poll_seconds),
        max_wait_seconds=int(args.max_train_wait_seconds),
    )

    print("[run_remote_round1_family_segmented] remote train exited; syncing scalar packet", flush=True)
    sync_scalar = _run(
        [sys.executable, str(SCRIPT_DIR / "sync_round1_remote_scalar_packet.py"), "--family-id", str(args.family_id)],
        timeout_ms=240000,
    )
    sys.stdout.write(sync_scalar.stdout)
    sys.stdout.flush()
    if sync_scalar.returncode != 0:
        _write_manifest_status(
            manifest_csv,
            family_id=str(args.family_id),
            decision_status=exit_status,
        )
        return int(sync_scalar.returncode)

    latest_epoch_after, latest_name_after = _scan_latest_epoch(
        remote_run_dir=remote_run_dir,
        host=str(args.host),
        port=int(args.port),
        wsl_distro=str(args.wsl_distro),
    )
    print(
        f"[run_remote_round1_family_segmented] latest_epoch_after_train={latest_epoch_after} latest_ckpt_after_train={latest_name_after or 'none'}",
        flush=True,
    )
    if int(latest_epoch_after) < int(expected_epoch):
        print(
            "[run_remote_round1_family_segmented] no new retained checkpoint landed during the bounded segment; "
            "skip remote fast-eval launch for this cycle",
            flush=True,
        )
        _write_manifest_status(
            manifest_csv,
            family_id=str(args.family_id),
            decision_status=exit_status,
        )
        return 26

    if not bool(args.skip_fast_eval):
        print("[run_remote_round1_family_segmented] launching remote fast-eval after train exit", flush=True)
        launch_fast = _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "launch_remote_round1_family_fast_eval.py"),
                "--config",
                str(segmented_config.relative_to(WORKSPACE)),
                "--max-live-memory-mib-to-launch",
                "9800",
            ],
            timeout_ms=240000,
        )
        sys.stdout.write(launch_fast.stdout)
        sys.stdout.flush()
        if launch_fast.returncode != 0:
            _write_manifest_status(
                manifest_csv,
                family_id=str(args.family_id),
                decision_status=exit_status,
            )
            return int(launch_fast.returncode)
        _wait_for_eval_summary(
            family_id=str(args.family_id),
            expected_epoch=int(expected_epoch),
            poll_seconds=int(args.poll_seconds),
            max_wait_seconds=int(args.max_eval_wait_seconds),
        )
        print("[run_remote_round1_family_segmented] remote fast-eval summary reached expected epoch", flush=True)
    _write_manifest_status(
        manifest_csv,
        family_id=str(args.family_id),
        decision_status=exit_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
