from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_UPDATER = SCRIPT_DIR / "update_round1_family_status_docs.py"
DEFAULT_REFRESH_CWD = SCRIPT_DIR.parent.parent.parent
DEFAULT_MANIFEST = (
    DEFAULT_REFRESH_CWD / "SchrodingerBridge" / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"
)

from csv_utils import manifest_fieldnames, read_csv_rows, write_csv_rows


def _family_tag(family_id: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(family_id).strip()).strip("._-")
    return clean or "round1_family"


def _default_history_jsonl(*, refresh_cwd: Path, family_id: str) -> Path:
    return refresh_cwd / "SchrodingerBridge" / "aaai2027" / f"round1_{_family_tag(family_id)}_runtime_samples.jsonl"


def _default_summary_json(*, refresh_cwd: Path, family_id: str) -> Path:
    return refresh_cwd / "SchrodingerBridge" / "aaai2027" / f"round1_{_family_tag(family_id)}_runtime_summary.json"


def _default_convergence_json(*, refresh_cwd: Path, family_id: str) -> Path:
    return refresh_cwd / "SchrodingerBridge" / "aaai2027" / f"round1_{_family_tag(family_id)}_remote_full_eval_pull" / "round1_convergence.json"


def _run_update(command: list[str], *, cwd: Path | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        cwd=None if cwd is None else str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _read_manifest_row(path: Path, *, family_id: str) -> dict[str, str] | None:
    if not path.is_file():
        return None
    rows = read_csv_rows(path)
    for row in rows:
        if str(row.get("family_id", "")).strip() == str(family_id).strip():
            return row
    return None


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_last_jsonl(path: Path) -> dict | None:
    if not path.is_file():
        return None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            last = text
    if not last:
        return None
    try:
        payload = json.loads(last)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _snapshot_signature(payload: dict) -> tuple:
    return (
        str(payload.get("decision_status", "")).strip(),
        str(payload.get("remote_live_memory_used_mib", "")).strip(),
        str(payload.get("remote_live_memory_total_mib", "")).strip(),
        str(payload.get("remote_live_util_pct", "")).strip(),
        str(payload.get("remote_live_band_status", "")).strip(),
        str(payload.get("remote_live_formal_status", "")).strip(),
        str(payload.get("remote_live_epoch", "")).strip(),
        str(payload.get("remote_live_epoch_total", "")).strip(),
        str(payload.get("remote_live_step", "")).strip(),
        str(payload.get("remote_live_step_total", "")).strip(),
        str(payload.get("remote_live_loss", "")).strip(),
        str(payload.get("remote_live_tswd", "")).strip(),
    )


def _summarize_history(path: Path, *, tail_count: int) -> dict:
    if not path.is_file():
        return {"sample_count": 0, "recent_samples": []}
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    recent = rows[-max(1, int(tail_count)) :]
    latest = recent[-1] if recent else None
    latest_nonempty = None
    for item in reversed(recent):
        if any(str(item.get(key, "")).strip() for key in ["remote_live_memory_used_mib", "remote_live_epoch", "remote_live_loss"]):
            latest_nonempty = item
            break
    consecutive_under_band = 0
    for item in reversed(recent):
        if str(item.get("remote_live_band_status", "")).strip() == "under_band":
            consecutive_under_band += 1
        else:
            break
    return {
        "sample_count": len(rows),
        "recent_samples": recent,
        "latest_sample": latest,
        "latest_nonempty_sample": latest_nonempty,
        "recent_under_band_count": sum(
            1 for item in recent if str(item.get("remote_live_band_status", "")).strip() == "under_band"
        ),
        "consecutive_under_band": consecutive_under_band,
        "latest_formal_status": None if latest is None else latest.get("remote_live_formal_status"),
        "latest_nonempty_formal_status": None if latest_nonempty is None else latest_nonempty.get("remote_live_formal_status"),
    }


def _read_family_status(path: Path, *, family_id: str) -> str | None:
    if not path.is_file():
        return None
    rows = read_csv_rows(path)
    for row in rows:
        if str(row.get("family_id", "")).strip() == str(family_id).strip():
            return str(row.get("decision_status", "")).strip().lower()
    return None


def _read_converged(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("converged"))


def _write_family_status(path: Path, *, family_id: str, decision_status: str) -> bool:
    if not path.is_file():
        return False
    rows = read_csv_rows(path)
    changed = False
    for row in rows:
        if str(row.get("family_id", "")).strip() != str(family_id).strip():
            continue
        if str(row.get("decision_status", "")).strip() == str(decision_status).strip():
            return False
        row["decision_status"] = str(decision_status).strip()
        changed = True
        break
    if not changed:
        return False
    write_csv_rows(path, rows, fieldnames=manifest_fieldnames(rows))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Periodically refresh round-1 family docs with remote runtime status.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--refresh-cwd", type=Path, default=DEFAULT_REFRESH_CWD)
    parser.add_argument("--updater", type=Path, default=DEFAULT_UPDATER)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--history-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-tail-count", type=int, default=8)
    parser.add_argument("--allowed-status", action="append", default=[])
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()

    updater = Path(args.updater).expanduser()
    refresh_cwd = Path(args.refresh_cwd).expanduser()
    manifest_csv = Path(args.manifest_csv).expanduser()
    history_jsonl = (
        Path(args.history_jsonl).expanduser()
        if args.history_jsonl is not None
        else _default_history_jsonl(refresh_cwd=refresh_cwd, family_id=str(args.family_id))
    )
    summary_json = (
        Path(args.summary_json).expanduser()
        if args.summary_json is not None
        else _default_summary_json(refresh_cwd=refresh_cwd, family_id=str(args.family_id))
    )
    convergence_json = _default_convergence_json(refresh_cwd=refresh_cwd, family_id=str(args.family_id))
    allowed_statuses = {str(x).strip().lower() for x in list(args.allowed_status or []) if str(x).strip()}
    cycles = 0
    while True:
        family_status = None
        status_ok = True
        if allowed_statuses:
            family_status = _read_family_status(manifest_csv, family_id=str(args.family_id))
            status_ok = family_status in allowed_statuses
            if (family_status is not None) and (not status_ok):
                print(
                    json.dumps(
                        {
                            "family_id": str(args.family_id),
                            "family_status": family_status,
                            "status_ok": status_ok,
                            "allowed_status": sorted(allowed_statuses),
                            "action": "exit",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return 0
        result = _run_update(
            [
                sys.executable,
                str(updater),
                "--family-id",
                str(args.family_id),
                "--remote-live",
            ],
            cwd=refresh_cwd,
        )
        if result.stdout.strip():
            print(result.stdout.rstrip(), flush=True)
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr, flush=True)
        if result.returncode != 0:
            print(
                f"runtime_status_refresh_failed returncode={result.returncode} family_id={args.family_id}",
                file=sys.stderr,
                flush=True,
            )
        row = _read_manifest_row(manifest_csv, family_id=str(args.family_id))
        if row is not None:
            snapshot = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "family_id": str(args.family_id),
                "decision_status": row.get("decision_status", ""),
                "remote_live_memory_used_mib": row.get("remote_live_memory_used_mib", ""),
                "remote_live_memory_total_mib": row.get("remote_live_memory_total_mib", ""),
                "remote_live_util_pct": row.get("remote_live_util_pct", ""),
                "remote_live_band_status": row.get("remote_live_band_status", ""),
                "remote_live_formal_status": row.get("remote_live_formal_status", ""),
                "remote_live_epoch": row.get("remote_live_epoch", ""),
                "remote_live_epoch_total": row.get("remote_live_epoch_total", ""),
                "remote_live_step": row.get("remote_live_step", ""),
                "remote_live_step_total": row.get("remote_live_step_total", ""),
                "remote_live_loss": row.get("remote_live_loss", ""),
                "remote_live_tswd": row.get("remote_live_tswd", ""),
            }
            last_snapshot = _read_last_jsonl(history_jsonl)
            if _snapshot_signature(last_snapshot or {}) != _snapshot_signature(snapshot):
                _append_jsonl(history_jsonl, snapshot)
            summary = _summarize_history(history_jsonl, tail_count=int(args.summary_tail_count))
            summary_json.parent.mkdir(parents=True, exist_ok=True)
            summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            current_status = str(row.get("decision_status", "")).strip().lower()
            remote_epoch = str(row.get("remote_live_epoch", "")).strip()
            if current_status == "running" and (not remote_epoch) and _read_converged(convergence_json):
                changed = _write_family_status(
                    manifest_csv,
                    family_id=str(args.family_id),
                    decision_status="reviewing",
                )
                if changed:
                    print(
                        json.dumps(
                            {
                                "family_id": str(args.family_id),
                                "action": "auto_transition_to_reviewing",
                                "reason": "converged_and_remote_train_gone",
                                "convergence_json": str(convergence_json),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    result = _run_update(
                        [
                            sys.executable,
                            str(updater),
                            "--family-id",
                            str(args.family_id),
                            "--remote-live",
                        ],
                        cwd=refresh_cwd,
                    )
                    if result.stdout.strip():
                        print(result.stdout.rstrip(), flush=True)
                    if result.stderr.strip():
                        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
