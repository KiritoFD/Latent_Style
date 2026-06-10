from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
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
from round1_paths import round1_family_doc_dir


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"

REMOTE_SCAN_PY = r"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "root": str(root),
    "exists": root.exists(),
    "config_json": str(root / "config.json"),
    "logs": [],
    "checkpoints": [],
    "eval_files": [],
}
if root.exists():
    for path in sorted((root / "logs").glob("*.csv")):
        payload["logs"].append(
            {
                "name": path.name,
                "path": str(path),
                "size": path.stat().st_size,
            }
        )
    for path in sorted(root.glob("epoch_*.pt")):
        payload["checkpoints"].append(
            {
                "name": path.name,
                "path": str(path),
                "size": path.stat().st_size,
            }
        )
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".csv") or path.name == "summary.json" or path.name == "round1_convergence.json":
            rel = path.relative_to(root)
            if rel.parts and rel.parts[0] == "logs":
                continue
            payload["eval_files"].append(str(rel).replace("\\", "/"))
print(json.dumps(payload, ensure_ascii=False))
"""


def _md_link(label: str, path: Path) -> str:
    return f"[{label}]({str(path).replace(chr(92), '/')})"


def _run(cmd: list[str], *, input_text: str | None = None, timeout: int = 120000) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout / 1000 if timeout > 1000 else timeout,
        check=False,
    )


def _load_manifest_row(manifest_csv: Path, *, family_id: str) -> dict[str, str]:
    rows = read_csv_rows(manifest_csv)
    for row in rows:
        if str(row.get("family_id", "")).strip() == str(family_id).strip():
            return row
    raise KeyError(f"family_id not found in manifest: {family_id}")


def _remote_run_dir(*, row: dict[str, str], remote_workspace_root: str) -> str:
    run_dir = str(row.get("run_dir", "")).strip()
    if run_dir.startswith("./"):
        return f"{remote_workspace_root.rstrip('/')}/{run_dir[2:]}"
    return run_dir


def _ssh_wsl_cmd(*, host: str, port: int, user: str, wsl_distro: str, extra: list[str]) -> list[str]:
    return [
        "ssh",
        "-p",
        str(int(port)),
        f"{user}@{host}",
        "wsl",
        "-d",
        str(wsl_distro),
        *extra,
    ]


def _scan_remote(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_run_dir: str,
) -> dict[str, Any]:
    cmd = _ssh_wsl_cmd(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        extra=["python3", "-", str(remote_run_dir)],
    )
    proc = _run(cmd, input_text=REMOTE_SCAN_PY, timeout=30000)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "remote scan failed")
    return json.loads(proc.stdout)


def _pull_text_file(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_file: str,
    local_file: Path,
) -> None:
    cmd = _ssh_wsl_cmd(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        extra=["--exec", "cat", str(remote_file)],
    )
    proc = _run(cmd, timeout=30000)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"pull failed: {remote_file}")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text(proc.stdout, encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _csv_has_data_rows(path: Path) -> bool:
    try:
        rows = _read_csv_rows(path)
    except Exception:
        return False
    return bool(rows)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _plot_if_exists(script_path: Path, *, input_path: Path, output_png: Path, extra_args: list[str] | None = None) -> None:
    if not input_path.is_file():
        return
    if script_path.name == "plot_round1_training_csv.py" and not _csv_has_data_rows(input_path):
        return
    cmd = [sys.executable, str(script_path), "--input-csv", str(input_path), "--output-png", str(output_png)]
    if script_path.name == "plot_round1_runtime_curve.py":
        cmd = [
            sys.executable,
            str(script_path),
            "--input-jsonl",
            str(input_path),
            "--output-png",
            str(output_png),
            "--output-csv",
            str(output_png.with_suffix(".csv")),
        ]
    if extra_args:
        cmd.extend(extra_args)
    proc = _run(cmd, timeout=120000)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"plot failed: {script_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull a round-1 family's remote scalar artifacts and build a local summary packet.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--local-root", type=Path, default=None)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    row = _load_manifest_row(manifest_csv, family_id=str(args.family_id))
    family_id = str(args.family_id).strip()
    run_name = str(row.get("run_name", "")).strip()
    local_root = Path(args.local_root).expanduser() if args.local_root is not None else (SB_ROOT / "aaai2027" / f"round1_{family_id}_remote_scalars")
    if not local_root.is_absolute():
        local_root = (WORKSPACE / local_root).resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    docs_dir = round1_family_doc_dir(family_id=family_id, run_name=run_name)
    docs_dir.mkdir(parents=True, exist_ok=True)

    remote_run_dir = _remote_run_dir(row=row, remote_workspace_root=str(args.remote_workspace_root))
    remote_scan = _scan_remote(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        wsl_distro=str(args.wsl_distro),
        remote_run_dir=remote_run_dir,
    )

    pulled_logs: list[Path] = []
    for item in remote_scan.get("logs", []):
        remote_file = str(item["path"])
        local_file = local_root / Path(str(item["name"]))
        _pull_text_file(
            host=str(args.host).split("@")[-1],
            port=int(args.port),
            user=str(args.user),
            wsl_distro=str(args.wsl_distro),
            remote_file=remote_file,
            local_file=local_file,
        )
        pulled_logs.append(local_file)

    latest_log = max(pulled_logs, key=lambda p: p.name) if pulled_logs else None
    training_curve_png = local_root / "training_curve.png"
    if latest_log is not None:
        _plot_if_exists(SCRIPT_DIR / "plot_round1_training_csv.py", input_path=latest_log, output_png=training_curve_png)

    runtime_jsonl = SB_ROOT / "aaai2027" / f"round1_{family_id}_runtime_samples.jsonl"
    runtime_curve_png = local_root / "runtime_curve.png"
    runtime_curve_csv = local_root / "runtime_curve.csv"
    if runtime_jsonl.is_file():
        proc = _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "plot_round1_runtime_curve.py"),
                "--input-jsonl",
                str(runtime_jsonl),
                "--output-png",
                str(runtime_curve_png),
                "--output-csv",
                str(runtime_curve_csv),
            ],
            timeout=120000,
        )
        if proc.returncode != 0:
            print(
                "[sync_round1_remote_scalar_packet] runtime plot warning: "
                + (proc.stderr or proc.stdout or "runtime plot failed"),
                file=sys.stderr,
            )

    latest_training_row: dict[str, str] | None = None
    if latest_log is not None:
        rows = _read_csv_rows(latest_log)
        if rows:
            latest_training_row = rows[-1]

    runtime_summary_path = SB_ROOT / "aaai2027" / f"round1_{family_id}_runtime_summary.json"
    runtime_summary = json.loads(runtime_summary_path.read_text(encoding="utf-8")) if runtime_summary_path.is_file() else {}

    payload = {
        "family_id": family_id,
        "run_name": run_name,
        "remote_run_dir": remote_run_dir,
        "pulled_logs": [str(path) for path in pulled_logs],
        "checkpoint_count": len(remote_scan.get("checkpoints", [])),
        "latest_checkpoint": "" if not remote_scan.get("checkpoints") else remote_scan["checkpoints"][-1]["name"],
        "eval_file_count": len(remote_scan.get("eval_files", [])),
        "eval_files": remote_scan.get("eval_files", []),
        "latest_training_row": latest_training_row,
        "runtime_summary": runtime_summary,
        "training_curve_png": str(training_curve_png) if training_curve_png.is_file() else "",
        "runtime_curve_png": str(runtime_curve_png) if runtime_curve_png.is_file() else "",
        "runtime_curve_csv": str(runtime_curve_csv) if runtime_curve_csv.is_file() else "",
        "updated_at": datetime.now().isoformat(),
    }
    summary_json = local_root / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines = [
        f"# {family_id} Remote Scalar Read",
        "",
        f"Updated: `{payload['updated_at']}`",
        "",
        f"- Run name: `{run_name}`",
        f"- Remote run dir: `{remote_run_dir}`",
        f"- Retained checkpoints: `{payload['checkpoint_count']}`",
        f"- Latest checkpoint: `{payload['latest_checkpoint']}`",
        f"- Remote eval files currently visible: `{payload['eval_file_count']}`",
    ]
    if int(payload["eval_file_count"]) == 0:
        md_lines.append("- Read: no remote `full_eval*` / summary / eval CSV artifacts exist yet under the run dir.")
    else:
        md_lines.append("- Eval files:")
        md_lines.extend([f"  - `{item}`" for item in payload["eval_files"][:20]])
    if pulled_logs:
        md_lines.append("- Pulled remote logs:")
        md_lines.extend([f"  - {_md_link(path.name, path)}" for path in pulled_logs])
    if latest_training_row:
        md_lines.extend(
            [
                "- Latest remote training CSV row:",
                f"  - `epoch={latest_training_row.get('epoch', '')}`",
                f"  - `loss={latest_training_row.get('loss', '')}`",
                f"  - `terminal_swd={latest_training_row.get('terminal_swd', '')}`",
                f"  - `samples_per_sec={latest_training_row.get('samples_per_sec', '')}`",
                f"  - `cuda_peak_allocated_gb={latest_training_row.get('cuda_peak_allocated_gb', '')}`",
                f"  - `cuda_peak_reserved_gb={latest_training_row.get('cuda_peak_reserved_gb', '')}`",
            ]
        )
    elif latest_log is not None:
        md_lines.extend(
            [
                "- Latest remote training CSV row:",
                "  - `waiting for first completed training row`",
            ]
        )
    latest_runtime = {}
    if isinstance(runtime_summary, dict):
        latest_runtime = dict(runtime_summary.get("latest_nonempty_sample", {}) or runtime_summary.get("latest_sample", {}) or {})
    if latest_runtime:
        md_lines.extend(
            [
                "- Latest runtime watcher sample:",
                f"  - `epoch={latest_runtime.get('remote_live_epoch', '')}/{latest_runtime.get('remote_live_epoch_total', '')}`",
                f"  - `step={latest_runtime.get('remote_live_step', '')}/{latest_runtime.get('remote_live_step_total', '')}`",
                f"  - `loss={latest_runtime.get('remote_live_loss', '')}`",
                f"  - `tswd={latest_runtime.get('remote_live_tswd', '')}`",
                f"  - `VRAM={latest_runtime.get('remote_live_memory_used_mib', '')}/{latest_runtime.get('remote_live_memory_total_mib', '')} MiB`",
                f"  - `band_status={latest_runtime.get('remote_live_band_status', '')}`",
            ]
        )
    if training_curve_png.is_file():
        md_lines.append(f"- Training curve: {_md_link(training_curve_png.name, training_curve_png)}")
    if runtime_curve_png.is_file():
        md_lines.append(f"- Runtime curve: {_md_link(runtime_curve_png.name, runtime_curve_png)}")
    if runtime_curve_csv.is_file():
        md_lines.append(f"- Runtime CSV: {_md_link(runtime_curve_csv.name, runtime_curve_csv)}")
    md_lines.append(f"- Summary JSON: {_md_link(summary_json.name, summary_json)}")

    summary_md = docs_dir / "remote_scalar_read.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(summary_md)
    print(summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
