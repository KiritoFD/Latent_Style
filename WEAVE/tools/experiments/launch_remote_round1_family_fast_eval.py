from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from csv_utils import read_csv_rows
from round1_paths import infer_round1_family_id
from round1_registry import ROUND1_FAMILY_SPECS

DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"


def _run(cmd: list[str]) -> int:
    print("[launch_remote_round1_family_fast_eval] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _remote_fast_eval_processes(*, run_name: str, host: str, port: int, wsl_distro: str) -> list[dict[str, str]]:
    scan_py = f"""
from pathlib import Path
import json

token = {run_name!r}
rows = []
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
    if (
        "watch_round1_family_fast_eval.py" in txt
        or "rerun_full_eval_for_run.py" in txt
        or "run_evaluation.py" in txt
        or "fast-eval.sh" in txt
        or "_fast_eval" in txt
        or "_fast-eval" in txt
    ):
        rows.append({{"pid": pid.name, "cmd": txt}})
print(json.dumps(rows, ensure_ascii=False))
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
        raise RuntimeError(proc.stdout or "remote fast-eval process scan failed")
    payload = json.loads(proc.stdout.strip() or "[]")
    return list(payload or [])


def _family_patience(family_id: str | None) -> int:
    for spec in ROUND1_FAMILY_SPECS:
        if str(spec.family_id).strip() == str(family_id or "").strip():
            return int(spec.patience)
    return 4


def _default_allowed_statuses(*, config_stem: str) -> list[str]:
    stem = str(config_stem).strip().lower()
    if "reconpretrain" in stem:
        return ["running", "recalibration_needed"]
    return ["running"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the remote fast-eval watcher for a round-1 family run.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--test-dir", default="/mnt/i/wikiart_distinct5_samam_512_classview/test")
    parser.add_argument("--cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache")
    parser.add_argument("--clip-hf-cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache/hf")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vae-decode-batch-size", type=int, default=4)
    parser.add_argument("--target-chunk-size", type=int, default=1)
    parser.add_argument("--max-live-memory-mib-to-launch", type=int, default=9800)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--allow-other-running", action="store_true")
    parser.add_argument("--allowed-status", action="append", default=[])
    parser.add_argument("--sync-src", action="store_true")
    parser.add_argument("--sync-remote-scripts", action="store_true")
    parser.add_argument("--sync-manifest", action="store_true")
    args = parser.parse_args()

    config_rel = Path(args.config)
    config_abs = (WORKSPACE / config_rel).resolve()
    payload = json.loads(config_abs.read_text(encoding="utf-8"))
    run_name = str((payload.get("ablation") or {}).get("name", config_abs.stem)).strip() or config_abs.stem
    family_id = infer_round1_family_id(run_name=run_name, config_stem=config_abs.stem)
    family_patience = _family_patience(family_id)
    allowed_statuses = [str(x).strip().lower() for x in list(args.allowed_status or []) if str(x).strip()]
    if not allowed_statuses:
        allowed_statuses = _default_allowed_statuses(config_stem=config_abs.stem)
    existing_fast_eval = _remote_fast_eval_processes(
        run_name=run_name,
        host=str(args.host),
        port=int(args.port),
        wsl_distro=str(args.wsl_distro),
    )
    if existing_fast_eval:
        print(
            "[launch_remote_round1_family_fast_eval] skip launch because same-run fast-eval processes already exist: "
            + ", ".join(str(row.get("pid", "")) for row in existing_fast_eval),
            flush=True,
        )
        return 0
    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    if manifest_csv.is_file() and not bool(args.allow_other_running):
        rows = read_csv_rows(manifest_csv)
        target_family = str(family_id or config_abs.stem).strip()
        foreign_running = [
            str(row.get("family_id", "")).strip()
            for row in rows
            if str(row.get("decision_status", "")).strip().lower() == "running"
            and str(row.get("family_id", "")).strip() != target_family
        ]
        if foreign_running:
            raise RuntimeError(
                "Refusing direct remote fast-eval launch because other running round-1 families exist: "
                + ", ".join(foreign_running)
            )
    run_dir = str((payload.get("checkpoint") or {}).get("save_dir", "")).strip()
    if run_dir.startswith("./"):
        run_dir = f"{args.remote_wsl_cwd.rstrip('/')}/{run_dir[2:]}"
    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    sync_pairs: list[str] = []
    if bool(args.sync_src):
        sync_pairs.extend(["--sync-path", "SchrodingerBridge/src"])
    if bool(args.sync_remote_scripts):
        sync_pairs.extend(
            [
                "--sync-path",
                "SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py",
                "--sync-path",
                "SchrodingerBridge/tools/experiments/report_round1_convergence.py",
                "--sync-path",
                "SchrodingerBridge/tools/experiments/watch_round1_family_fast_eval.py",
            ]
        )
    if bool(args.sync_manifest):
        sync_pairs.extend(
            [
            "--sync-path",
            str(manifest_csv.resolve().relative_to(WORKSPACE.resolve())),
            ]
        )
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        f"round1-{family_id}-fast-eval",
        "--remote-log-path",
        f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_fast_eval.log",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        *sync_pairs,
        "--verify-python-file",
        "SchrodingerBridge/src/utils/run_evaluation.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/watch_round1_family_fast_eval.py",
        "--max-prelaunch-memory-mib",
        "12000",
        "--no-health-check",
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            f"{args.remote_python} SchrodingerBridge/tools/experiments/watch_round1_family_fast_eval.py "
            f"--python-bin {args.remote_python} "
            f"--run-dir {run_dir} "
            f"--test-dir {args.test_dir} "
            f"--cache-dir {args.cache_dir} "
            f"--clip-hf-cache-dir {args.clip_hf_cache_dir} "
            "--output-subdir full_eval_fast_snapshot "
            f"--batch-size {int(args.batch_size)} "
            f"--vae-decode-batch-size {int(args.vae_decode_batch_size)} "
            f"--target-chunk-size {int(args.target_chunk_size)} "
            f"--max-live-memory-mib-to-launch {int(args.max_live_memory_mib_to_launch)} "
            f"--poll-seconds {int(args.poll_seconds)} "
            f"--patience {int(family_patience)} "
            "--manifest-csv SchrodingerBridge/docs/experiments/round1_full_sweep/round1_family_manifest.csv "
            f"--family-id {family_id} "
            + " ".join(f"--allowed-status {status}" for status in allowed_statuses)
        ),
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
