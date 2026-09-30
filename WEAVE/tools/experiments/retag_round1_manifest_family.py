from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from csv_utils import manifest_fieldnames, read_csv_rows, write_csv_rows


VALID_STATUSES = {
    "planned",
    "running",
    "reviewing",
    "rejected",
    "recalibration_needed",
    "accepted",
}


def _normalize_statuses(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        item = str(value).strip().lower()
        if item:
            out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely retag one or more round-1 family decision_status rows in the manifest.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--family-id", action="append", required=True, help="One or more family ids to retag.")
    parser.add_argument("--decision-status", required=True, help="Target decision_status.")
    parser.add_argument(
        "--if-current-status",
        action="append",
        default=[],
        help="Optional current-status guard. Repeat to allow multiple source statuses.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    rows = read_csv_rows(manifest_csv)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_csv}")

    family_ids = [str(item).strip() for item in args.family_id if str(item).strip()]
    target_status = str(args.decision_status).strip().lower()
    if target_status not in VALID_STATUSES:
        raise ValueError(f"Unsupported decision_status: {target_status}")

    allowed_current = set(_normalize_statuses(args.if_current_status))
    missing = [family_id for family_id in family_ids if not any(str(row.get("family_id", "")).strip() == family_id for row in rows)]
    if missing:
        raise KeyError(f"Family ids not found in manifest: {', '.join(missing)}")

    changes: list[tuple[str, str, str]] = []
    for row in rows:
        family_id = str(row.get("family_id", "")).strip()
        if family_id not in family_ids:
            continue
        current = str(row.get("decision_status", "")).strip().lower()
        if allowed_current and current not in allowed_current:
            raise RuntimeError(
                f"Refusing to retag {family_id}: current status {current!r} not in guard set {sorted(allowed_current)!r}"
            )
        if current == target_status:
            continue
        changes.append((family_id, current, target_status))
        row["decision_status"] = target_status

    if not changes:
        print("No manifest rows changed.")
        return 0

    for family_id, current, new in changes:
        print(f"{family_id}: {current} -> {new}")

    if bool(args.dry_run):
        print("DRY_RUN=1")
        return 0

    write_csv_rows(manifest_csv, rows, fieldnames=manifest_fieldnames(rows))
    print(str(manifest_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
