from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from csv_utils import manifest_fieldnames, read_csv_rows, write_csv_rows
from round1_manifest_utils import (
    DEFAULT_MANIFEST,
    is_dino_tail,
    resolve_manifest_csv,
    smoke_status_of,
    status_of,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote the first smoke-ok non-DINO round1 family from recalibration_needed to planned."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--from-status", default="recalibration_needed")
    parser.add_argument("--to-status", default="planned")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_csv = resolve_manifest_csv(args.manifest_csv)
    rows = read_csv_rows(manifest_csv)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_csv}")

    running = [row for row in rows if status_of(row) == "running"]
    if running:
        active = ", ".join(str(row.get("family_id", "")).strip() for row in running)
        print(f"REFUSE_RUNNING_ACTIVE={active}")
        return 0

    from_status = str(args.from_status).strip().lower()
    to_status = str(args.to_status).strip().lower()

    target = None
    for row in rows:
        if status_of(row) != from_status:
            continue
        if is_dino_tail(row):
            continue
        if smoke_status_of(row) != "ok":
            continue
        target = row
        break

    if target is None:
        print("No smoke-ok non-DINO candidate found for promotion.")
        return 0

    family_id = str(target.get("family_id", "")).strip()
    print(f"{family_id}: {from_status} -> {to_status}")
    if bool(args.dry_run):
        print("DRY_RUN=1")
        return 0

    target["decision_status"] = to_status
    write_csv_rows(manifest_csv, rows, fieldnames=manifest_fieldnames(rows))
    print(str(manifest_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
