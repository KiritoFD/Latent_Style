from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"

from csv_utils import read_csv_rows, write_csv_rows
from round1_paths import infer_round1_family_id, round1_fast_local_root
from round1_registry import ROUND1_FAMILY_SPECS


def _read_rows(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    write_csv_rows(path, rows)


def _run(cmd: list[str]) -> int:
    print("[run_round1_family_queue] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    proc = subprocess.run(cmd, check=False, cwd=str(WORKSPACE), env=env)
    return int(proc.returncode)


def _launch_followups_for_family(
    *,
    config_path: Path,
    launch_deferred_fast_eval: bool,
    launch_stageclose_deferred: bool,
) -> int:
    helper = SCRIPT_DIR / "launch_round1_family_followups_detached.py"
    cmd = [
        sys.executable,
        str(helper),
        "--config",
        str(config_path.relative_to(WORKSPACE)),
    ]
    if not bool(launch_deferred_fast_eval):
        cmd.append("--skip-fast-eval-deferred")
    if not bool(launch_stageclose_deferred):
        cmd.append("--skip-stageclose-deferred")
    return _run(cmd)


def _family_patience(family_id: str | None) -> int:
    for spec in ROUND1_FAMILY_SPECS:
        if str(spec.family_id).strip() == str(family_id or "").strip():
            return int(spec.patience)
    return 4


def _first_row(rows: list[dict[str, str]], *, status: str) -> dict[str, str] | None:
    for row in rows:
        if str(row.get("decision_status", "")).strip().lower() == status:
            return row
    return None


def _rows_by_status(rows: list[dict[str, str]], *, status: str) -> list[dict[str, str]]:
    wanted = str(status).strip().lower()
    return [row for row in rows if str(row.get("decision_status", "")).strip().lower() == wanted]


def _switch_smoke_status(row: dict[str, str]) -> str:
    return str(row.get("switch_smoke_status", "")).strip().lower()


def _is_dino_tail(row: dict[str, str]) -> bool:
    tokenizer_family = str(row.get("tokenizer_family", "")).strip().lower()
    semantic_supervision_family = str(row.get("semantic_supervision_family", "")).strip().lower()
    family_id = str(row.get("family_id", "")).strip().lower()
    if not tokenizer_family or not semantic_supervision_family:
        config_path = Path(str(row.get("config_path", "")).strip())
        if config_path:
            config_abs = config_path if config_path.is_absolute() else (WORKSPACE / config_path).resolve()
            if config_abs.exists():
                try:
                    payload = json.loads(config_abs.read_text(encoding="utf-8"))
                    model_cfg = payload.get("model") or {}
                    bridge_cfg = payload.get("bridge") or {}
                    tokenizer_family = tokenizer_family or str(model_cfg.get("tokenizer_family", "")).strip().lower()
                    semantic_supervision_family = semantic_supervision_family or str(bridge_cfg.get("semantic_supervision_family", "")).strip().lower()
                except Exception:
                    pass
    return (
        "dino" in tokenizer_family
        or "dino" in semantic_supervision_family
        or family_id in {"tok_a_dino_dict", "tok_b_cross_image"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the next planned round-1 family using the generic family launchers.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--launch-running-too", action="store_true")
    parser.add_argument("--use-remote-fast-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-fast-eval-launch", action="store_true", help="Launch only the remote training lane; defer fast-eval watcher launch.")
    parser.add_argument("--allow-switch-smoke-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest_csv).resolve()
    rows = _read_rows(manifest)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest}")

    running_rows = _rows_by_status(rows, status="running")
    if running_rows and not bool(args.launch_running_too):
        active = ", ".join(str(row.get("family_id", "")).strip() for row in running_rows)
        if bool(args.dry_run):
            print(f"active_running={active}")
        else:
            print(f"Refusing to launch a new family while running families exist: {active}")
            return 0

    planned_rows = _rows_by_status(rows, status="planned")
    smoke_failed_rows = [
        row for row in planned_rows if _switch_smoke_status(row) == "failed"
    ]
    if not bool(args.allow_switch_smoke_failed):
        planned_rows = [
            row for row in planned_rows if _switch_smoke_status(row) != "failed"
        ]

    non_dino_rows = [row for row in planned_rows if not _is_dino_tail(row)]
    dino_rows = [row for row in planned_rows if _is_dino_tail(row)]

    def _prefer_smoke_ok(seq: list[dict[str, str]]) -> list[dict[str, str]]:
        smoke_ok = [row for row in seq if _switch_smoke_status(row) == "ok"]
        smoke_unknown = [row for row in seq if _switch_smoke_status(row) != "ok"]
        return smoke_ok + smoke_unknown

    non_dino_rows = _prefer_smoke_ok(non_dino_rows)
    dino_rows = _prefer_smoke_ok(dino_rows)
    target = non_dino_rows[0] if non_dino_rows else (dino_rows[0] if dino_rows else None)
    if target is None and bool(args.launch_running_too):
        target = _first_row(rows, status="running")
    if target is None:
        if smoke_failed_rows and not bool(args.allow_switch_smoke_failed):
            failed = ", ".join(str(row.get("family_id", "")).strip() for row in smoke_failed_rows)
            print(f"No launchable round-1 family found because switch smoke failed for: {failed}")
            return 0
        print("No launchable round-1 family found.")
        return 0

    config_path = Path(str(target["config_path"]))
    if not config_path.is_absolute():
        config_path = (WORKSPACE / config_path).resolve()

    train = [
        sys.executable,
        str(SCRIPT_DIR / "launch_remote_round1_family_train.py"),
        "--config",
        str(config_path.relative_to(WORKSPACE)),
        "--skip-remote-fast-eval-followup",
    ]
    fast_eval = [
        sys.executable,
        str(SCRIPT_DIR / "launch_remote_round1_family_fast_eval.py"),
        "--config",
        str(config_path.relative_to(WORKSPACE)),
    ]
    family_id = infer_round1_family_id(run_name=str(target.get("run_name", "")), config_stem=config_path.stem)
    family_patience = _family_patience(family_id)
    local_fast_root = round1_fast_local_root(family_id=family_id, run_name=str(target.get("run_name", "")))
    local_fast = [
        sys.executable,
        str(SCRIPT_DIR / "launch_local_round1_family_fast_eval_detached.py"),
        "--config",
        str(config_path.relative_to(WORKSPACE)),
        "--local-root",
        str(local_fast_root),
        "--stdout-log",
        str(local_fast_root / "local_fast_eval.stdout.log"),
        "--stderr-log",
        str(local_fast_root / "local_fast_eval.stderr.log"),
        "--patience",
        str(int(family_patience)),
    ]

    if bool(args.dry_run):
        print(target["family_id"])
        if smoke_failed_rows and not bool(args.allow_switch_smoke_failed):
            skipped = ", ".join(str(row.get("family_id", "")).strip() for row in smoke_failed_rows)
            print(f"SKIPPED_SMOKE_FAILED={skipped}")
        print(" ".join(str(x) for x in train))
        if bool(args.skip_fast_eval_launch):
            print("FAST_EVAL_LAUNCH=SKIPPED")
        elif bool(args.use_remote_fast_eval):
            print(" ".join(str(x) for x in fast_eval))
        else:
            print(" ".join(str(x) for x in local_fast))
        return 0

    rc = _run(train)
    if rc != 0:
        return rc
    if not bool(args.skip_fast_eval_launch):
        rc = _run(fast_eval if bool(args.use_remote_fast_eval) else local_fast)
        if rc != 0:
            return rc

    for row in rows:
        if row.get("family_id") == target.get("family_id"):
            row["decision_status"] = "running"
            break
    _write_rows(manifest, rows)
    rc = _launch_followups_for_family(
        config_path=config_path,
        launch_deferred_fast_eval=bool(args.skip_fast_eval_launch),
        launch_stageclose_deferred=bool(args.skip_fast_eval_launch),
    )
    if rc != 0:
        return rc
    print(target["family_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
