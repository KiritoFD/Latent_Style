from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from csv_utils import read_csv_rows
from round1_manifest_utils import DEFAULT_MANIFEST, resolve_manifest_csv
from update_round1_family_status_docs import (
    DEFAULT_REMOTE_BAND_MAX_MIB,
    DEFAULT_REMOTE_BAND_MIN_MIB,
    DEFAULT_REMOTE_HARD_CAP_MIB,
    DEFAULT_REMOTE_HOST,
    DEFAULT_REMOTE_PORT,
    DEFAULT_REMOTE_USER,
    DEFAULT_REMOTE_WORKSPACE_ROOT,
    DEFAULT_REMOTE_WSL_DISTRO,
    _classify_remote_vram_band,
    _effective_fast_eval_paths,
    _read_json_optional,
    _remote_runtime_snapshot,
)


def _find_family_row(rows: list[dict[str, str]], *, family_id: str) -> dict[str, str]:
    wanted = str(family_id).strip()
    for row in rows:
        if str(row.get("family_id", "")).strip() == wanted:
            return row
    raise KeyError(f"Family id not found in manifest: {wanted}")


def _build_payload(
    *,
    row: dict[str, str],
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_workspace_root: str,
    remote_log_lines: int,
    band_min_mib: int,
    band_max_mib: int,
    hard_cap_mib: int,
) -> dict[str, object]:
    family_id = str(row.get("family_id", "")).strip()
    fast_root, curve_csv, convergence_json, sync_summary = _effective_fast_eval_paths(
        family_id=family_id,
        fast_root=Path(str(row.get("local_fast_root", "")).strip() or "."),
        fast_eval_subdir=str(row.get("fast_eval_subdir", "")).strip() or "full_eval_fast_snapshot",
    )
    convergence = _read_json_optional(convergence_json)
    runtime = _remote_runtime_snapshot(
        row=row,
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        remote_workspace_root=remote_workspace_root,
        remote_log_lines=remote_log_lines,
    ) or {}
    gpu_sample = runtime.get("gpu_sample") if isinstance(runtime.get("gpu_sample"), dict) else None
    tail = runtime.get("tail") if isinstance(runtime.get("tail"), dict) else {}
    used_mib = None
    band_status = "unknown"
    formal_status = "unknown"
    if gpu_sample is not None:
        try:
            used_mib = int(str(gpu_sample.get("memory_used_mib", "")).strip())
        except Exception:
            used_mib = None
        band_status = _classify_remote_vram_band(
            memory_used_mib=used_mib,
            band_min_mib=band_min_mib,
            band_max_mib=band_max_mib,
            hard_cap_mib=hard_cap_mib,
        )
        formal_status = "formal_in_band" if band_status == "in_band" else f"nonformal_{band_status}"
    return {
        "family_id": family_id,
        "run_name": str(row.get("run_name", "")).strip(),
        "manifest_status": str(row.get("decision_status", "")).strip(),
        "switch_smoke_status": str(row.get("switch_smoke_status", "")).strip(),
        "config_path": str(row.get("config_path", "")).strip(),
        "run_dir": str(row.get("run_dir", "")).strip(),
        "fast_eval_root": str(fast_root),
        "curve_csv": str(curve_csv),
        "convergence_json": str(convergence_json),
        "sync_summary_path": str(fast_root / "sync_summary.json"),
        "sync_summary": sync_summary,
        "convergence": convergence,
        "remote_runtime": {
            "train_alive": bool(runtime.get("train_alive")),
            "train_log": runtime.get("train_log"),
            "processes": runtime.get("processes"),
            "gpu_sample": gpu_sample,
            "band_status": band_status,
            "formal_status": formal_status,
            "tail": tail,
        },
    }


def _print_text(payload: dict[str, object]) -> None:
    runtime = payload.get("remote_runtime") if isinstance(payload.get("remote_runtime"), dict) else {}
    convergence = payload.get("convergence") if isinstance(payload.get("convergence"), dict) else {}
    gpu_sample = runtime.get("gpu_sample") if isinstance(runtime.get("gpu_sample"), dict) else {}
    tail = runtime.get("tail") if isinstance(runtime.get("tail"), dict) else {}
    processes = runtime.get("processes") if isinstance(runtime.get("processes"), dict) else {}
    print(f"family_id: {payload.get('family_id', '')}")
    print(f"run_name: {payload.get('run_name', '')}")
    print(f"manifest_status: {payload.get('manifest_status', '')}")
    print(f"switch_smoke_status: {payload.get('switch_smoke_status', '')}")
    print(f"config_path: {payload.get('config_path', '')}")
    print(f"run_dir: {payload.get('run_dir', '')}")
    print(f"fast_eval_root: {payload.get('fast_eval_root', '')}")
    print(f"curve_csv: {payload.get('curve_csv', '')}")
    print(f"remote_train_alive: {runtime.get('train_alive', False)}")
    if gpu_sample:
        print(
            "remote_gpu: "
            f"{gpu_sample.get('memory_used_mib', '?')} / {gpu_sample.get('memory_total_mib', '?')} MiB, "
            f"util={gpu_sample.get('utilization_gpu_pct', '?')}%, "
            f"band={runtime.get('band_status', 'unknown')}, "
            f"formal={runtime.get('formal_status', 'unknown')}"
        )
    if tail:
        epoch = tail.get("epoch")
        epoch_total = tail.get("epoch_total")
        step = tail.get("step")
        step_total = tail.get("step_total")
        loss = tail.get("loss")
        tswd = tail.get("tswd")
        print(
            "remote_progress: "
            f"epoch {epoch}/{epoch_total}, step {step}/{step_total}, loss={loss}, tswd={tswd}"
        )
    print(
        "process_counts: "
        f"train={len(list(processes.get('train') or []))}, "
        f"fast_eval={len(list(processes.get('fast_eval') or []))}, "
        f"posttrain_eval={len(list(processes.get('posttrain_eval') or []))}"
    )
    if convergence:
        print(
            "convergence: "
            f"row_count={convergence.get('row_count')}, "
            f"latest={convergence.get('newest_epoch')}, "
            f"last_pareto={convergence.get('last_pareto_epoch')}, "
            f"since_last_pareto={convergence.get('since_last_pareto')}, "
            f"tail_flat={convergence.get('tail_flat')}, "
            f"patience={convergence.get('patience')}, "
            f"converged={convergence.get('converged')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Report one round-1 family's live remote runtime plus local convergence snapshot.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--remote-user", default=DEFAULT_REMOTE_USER)
    parser.add_argument("--remote-wsl-distro", default=DEFAULT_REMOTE_WSL_DISTRO)
    parser.add_argument("--remote-workspace-root", default=DEFAULT_REMOTE_WORKSPACE_ROOT)
    parser.add_argument("--remote-log-lines", type=int, default=80)
    parser.add_argument("--remote-band-min-mib", type=int, default=DEFAULT_REMOTE_BAND_MIN_MIB)
    parser.add_argument("--remote-band-max-mib", type=int, default=DEFAULT_REMOTE_BAND_MAX_MIB)
    parser.add_argument("--remote-hard-cap-mib", type=int, default=DEFAULT_REMOTE_HARD_CAP_MIB)
    args = parser.parse_args()

    manifest_csv = resolve_manifest_csv(args.manifest_csv)
    rows = read_csv_rows(manifest_csv)
    row = _find_family_row(rows, family_id=str(args.family_id))
    payload = _build_payload(
        row=row,
        host=str(args.remote_host),
        port=int(args.remote_port),
        user=str(args.remote_user),
        wsl_distro=str(args.remote_wsl_distro),
        remote_workspace_root=str(args.remote_workspace_root),
        remote_log_lines=int(args.remote_log_lines),
        band_min_mib=int(args.remote_band_min_mib),
        band_max_mib=int(args.remote_band_max_mib),
        hard_cap_mib=int(args.remote_hard_cap_mib),
    )
    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
