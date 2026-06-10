from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config
from round1_paths import infer_round1_family_id, round1_fast_local_root, round1_localreview_root
from round1_registry import ROUND1_FAMILY_SPECS


DEFAULT_BASELINE_WAIT_JSON = WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samam_wikiarts5_patch8_segmented_20260610_094447" / "curve_convergence.json"
DEFAULT_WSL_PROCESS_MATCH = "/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447"
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"
DEFAULT_BASELINE_MANIFEST = WORKSPACE / "SchrodingerBridge" / "aaai2027" / "vlm_manifest_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260609.csv"


def _pascal_case(token: str) -> str:
    parts = [part for part in str(token).strip().split("_") if part]
    if not parts:
        return ""
    cooked = []
    for part in parts:
        if part.isupper():
            cooked.append(part)
        else:
            cooked.append(part[:1].upper() + part[1:])
    return "".join(cooked)


def _family_patience(family_id: str) -> int:
    for spec in ROUND1_FAMILY_SPECS:
        if str(spec.family_id).strip() == str(family_id).strip():
            return int(spec.patience)
    return 4


def _spawn_detached(*, args: list[str], stdout_log: Path, stderr_log: Path) -> None:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    with stdout_log.open("w", encoding="utf-8") as stdout_f, stderr_log.open("w", encoding="utf-8") as stderr_f:
        subprocess.Popen(
            [sys.executable, *args],
            cwd=str(WORKSPACE),
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


def _find_local_python_pids_for_tokens(*tokens: str) -> list[int]:
    clauses: list[str] = []
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        safe = text.replace("'", "''")
        clauses.append(f"($_.CommandLine -like '*{safe}*')")
    conditions = " -and ".join(clauses)
    if not conditions:
        return []
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and "
        + conditions
        + " } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            pids.append(int(text))
    return pids


def _stop_local_python_watchers(*tokens: str) -> None:
    pids = _find_local_python_pids_for_tokens(*tokens)
    if not pids:
        return
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Stop-Process -Id " + ",".join(str(pid) for pid in pids) + " -Force"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch detached local/runtime followup watchers for a round-1 family.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-wait-json", type=Path, default=DEFAULT_BASELINE_WAIT_JSON)
    parser.add_argument("--wsl-process-match", default=DEFAULT_WSL_PROCESS_MATCH)
    parser.add_argument("--wsl-distro", default="f")
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST)
    parser.add_argument("--baseline-runs", nargs="+", default=["Seedream_repaired750", "SaMAM_2250"])
    parser.add_argument("--family-label-prefix", default="")
    parser.add_argument("--family-method", default="")
    parser.add_argument("--skip-runtime-watch", action="store_true")
    parser.add_argument("--skip-remote-fast-eval-sync", action="store_true")
    parser.add_argument("--skip-fast-eval-deferred", action="store_true")
    parser.add_argument("--skip-stageclose-deferred", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (WORKSPACE / cfg_path).resolve()
    cfg = load_config(cfg_path)
    run_name = str((cfg.get("ablation") or {}).get("name", cfg_path.stem)).strip() or cfg_path.stem
    family_id = infer_round1_family_id(run_name=run_name, config_stem=cfg_path.stem) or cfg_path.stem
    fast_root = round1_fast_local_root(family_id=family_id, run_name=run_name)
    review_root = round1_localreview_root(family_id=family_id, run_name=run_name)
    family_label_prefix = str(args.family_label_prefix).strip() or _pascal_case(family_id)
    family_method = str(args.family_method).strip() or family_label_prefix
    family_patience = _family_patience(family_id)

    if not bool(args.skip_runtime_watch):
        _stop_local_python_watchers("watch_round1_family_runtime_status.py", family_id)
        _spawn_detached(
            args=[
                str(SCRIPT_DIR / "watch_round1_family_runtime_status.py"),
                "--family-id",
                family_id,
                "--poll-seconds",
                "180",
                "--manifest-csv",
                str(Path(args.manifest_csv).resolve()),
                "--allowed-status",
                "running",
            ],
            stdout_log=SB_ROOT / "aaai2027" / f"round1_{family_id}_runtime_watch_20260610.stdout.log",
            stderr_log=SB_ROOT / "aaai2027" / f"round1_{family_id}_runtime_watch_20260610.stderr.log",
        )

    if not bool(args.skip_remote_fast_eval_sync):
        _stop_local_python_watchers("watch_sync_round1_remote_fast_eval_packet.py", family_id)
        _spawn_detached(
            args=[
                str(SCRIPT_DIR / "watch_sync_round1_remote_fast_eval_packet.py"),
                "--family-id",
                family_id,
                "--poll-seconds",
                "180",
            ],
            stdout_log=SB_ROOT / "aaai2027" / f"round1_{family_id}_remote_fast_sync_20260610.stdout.log",
            stderr_log=SB_ROOT / "aaai2027" / f"round1_{family_id}_remote_fast_sync_20260610.stderr.log",
        )

    if not bool(args.skip_fast_eval_deferred):
        _stop_local_python_watchers("launch_local_round1_fast_eval_after_wsl_idle.py", family_id)
        _spawn_detached(
            args=[
                str(SCRIPT_DIR / "launch_local_round1_fast_eval_after_wsl_idle.py"),
                "--config",
                str(Path(args.config)),
                "--local-root",
                str(fast_root),
                "--stdout-log",
                str(fast_root / "local_fast_eval.stdout.log"),
                "--stderr-log",
                str(fast_root / "local_fast_eval.stderr.log"),
                "--wait-json",
                str(Path(args.baseline_wait_json).resolve()),
                "--wait-json-key",
                "converged",
                "--wsl-distro",
                str(args.wsl_distro),
                "--wsl-process-match",
                str(args.wsl_process_match),
                "--poll-seconds",
                "120",
                "--patience",
                str(int(family_patience)),
                "--manifest-csv",
                str(Path(args.manifest_csv).resolve()),
                "--family-id",
                family_id,
                "--allowed-status",
                "running",
            ],
            stdout_log=SB_ROOT / "aaai2027" / f"round1_{family_id}_fast_eval_deferred_20260610.stdout.log",
            stderr_log=SB_ROOT / "aaai2027" / f"round1_{family_id}_fast_eval_deferred_20260610.stderr.log",
        )

    if not bool(args.skip_stageclose_deferred):
        _stop_local_python_watchers("run_round1_family_stageclose_when_ready.py", family_id)
        _spawn_detached(
            args=[
                str(SCRIPT_DIR / "run_round1_family_stageclose_when_ready.py"),
                "--config",
                str(Path(args.config)),
                "--fast-local-root",
                str(fast_root),
                "--review-local-root",
                str(review_root),
                "--baseline-manifest",
                str(Path(args.baseline_manifest).resolve()),
                "--baseline-runs",
                *[str(x) for x in args.baseline_runs],
                "--family-label-prefix",
                family_label_prefix,
                "--family-method",
                family_method,
                "--vlm-output-dir",
                str(SB_ROOT / "aaai2027" / f"round1_{family_id}_external_vlm_stageclose"),
                "--manifest-csv",
                str(Path(args.manifest_csv).resolve()),
                "--allowed-status",
                "running",
            ],
            stdout_log=SB_ROOT / "aaai2027" / f"round1_{family_id}_stageclose_deferred_20260610.stdout.log",
            stderr_log=SB_ROOT / "aaai2027" / f"round1_{family_id}_stageclose_deferred_20260610.stderr.log",
        )

    print(family_id)
    print(fast_root)
    print(review_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
