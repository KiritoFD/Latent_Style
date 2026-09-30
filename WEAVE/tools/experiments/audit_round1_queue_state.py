from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from csv_utils import read_csv_rows
from round1_manifest_utils import (
    DEFAULT_MANIFEST,
    candidate_ids,
    is_dino_tail,
    relaunchable_non_dino,
    resolve_manifest_csv,
    rows_by_status,
    smoke_status_of,
)


def _fmt_row(row: dict[str, str]) -> str:
    family_id = str(row.get("family_id", "")).strip()
    axis = str(row.get("axis", "")).strip()
    smoke = smoke_status_of(row) or "unknown"
    batch = str(row.get("batch_size", "")).strip() or "?"
    note = "dino-tail" if is_dino_tail(row) else "non-dino"
    return f"{family_id} [{axis}] smoke={smoke} batch={batch} {note}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the current round1 manifest and report safe queue state.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest_csv = resolve_manifest_csv(args.manifest_csv)
    rows = read_csv_rows(manifest_csv)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_csv}")

    running = rows_by_status(rows, status="running")
    planned = rows_by_status(rows, status="planned")
    reviewing = rows_by_status(rows, status="reviewing")
    recal = rows_by_status(rows, status="recalibration_needed")

    planned_non_dino = [row for row in planned if not is_dino_tail(row)]
    planned_dino = [row for row in planned if is_dino_tail(row)]
    planned_smoke_ok_non_dino = [row for row in planned_non_dino if smoke_status_of(row) == "ok"]
    planned_smoke_ok_dino = [row for row in planned_dino if smoke_status_of(row) == "ok"]
    reviewing_non_dino = [row for row in reviewing if not is_dino_tail(row)]
    recal_non_dino = [row for row in recal if not is_dino_tail(row)]
    relaunchable = relaunchable_non_dino(rows)

    payload = {
        "manifest_csv": str(manifest_csv),
        "running": candidate_ids(running),
        "planned_non_dino": candidate_ids(planned_non_dino),
        "planned_dino": candidate_ids(planned_dino),
        "reviewing": candidate_ids(reviewing),
        "recalibration_needed": candidate_ids(recal),
        "reviewing_non_dino": candidate_ids(reviewing_non_dino),
        "recalibration_needed_non_dino": candidate_ids(recal_non_dino),
        "relaunchable_non_dino": candidate_ids(relaunchable),
        "next_queue_candidate_if_running_clears": (
            str(planned_smoke_ok_non_dino[0].get("family_id", "")).strip()
            if planned_smoke_ok_non_dino
            else (
                str(planned_smoke_ok_dino[0].get("family_id", "")).strip()
                if planned_smoke_ok_dino
                else ""
            )
        ),
        "dino_tail_block_would_trigger": bool((not planned_smoke_ok_non_dino) and planned_smoke_ok_dino),
    }

    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"manifest: {manifest_csv}")
    print("running:")
    if running:
        for row in running:
            print(f"  - {_fmt_row(row)}")
    else:
        print("  - none")

    print("planned non-dino:")
    if planned_non_dino:
        for row in planned_non_dino:
            print(f"  - {_fmt_row(row)}")
    else:
        print("  - none")

    print("planned dino-tail:")
    if planned_dino:
        for row in planned_dino:
            print(f"  - {_fmt_row(row)}")
    else:
        print("  - none")

    print("reviewing:")
    if reviewing:
        for row in reviewing:
            print(f"  - {_fmt_row(row)}")
    else:
        print("  - none")

    print("recalibration_needed:")
    if recal:
        for row in recal:
            print(f"  - {_fmt_row(row)}")
    else:
        print("  - none")

    print("relaunchable non-dino (if you want to keep queue non-dino-first):")
    if relaunchable:
        for row in relaunchable:
            print(f"  - {_fmt_row(row)}")
    else:
        print("  - none")

    next_candidate = payload["next_queue_candidate_if_running_clears"] or "none"
    print(f"next_queue_candidate_if_running_clears: {next_candidate}")
    print(f"dino_tail_block_would_trigger: {payload['dino_tail_block_would_trigger']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
