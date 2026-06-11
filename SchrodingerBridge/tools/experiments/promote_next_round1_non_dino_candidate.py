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


def _status(row: dict[str, str]) -> str:
    return str(row.get("decision_status", "")).strip().lower()


def _smoke(row: dict[str, str]) -> str:
    return str(row.get("switch_smoke_status", "")).strip().lower()


def _is_dino_tail(row: dict[str, str]) -> bool:
    tokenizer_family = str(row.get("tokenizer_family", "")).strip().lower()
    semantic_supervision_family = str(row.get("semantic_supervision_family", "")).strip().lower()
    family_id = str(row.get("family_id", "")).strip().lower()
    return (
        "dino" in tokenizer_family
        or "dino" in semantic_supervision_family
        or family_id in {"tok_a_dino_dict", "tok_b_cross_image"}
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

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    rows = read_csv_rows(manifest_csv)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_csv}")

    running = [row for row in rows if _status(row) == "running"]
    if running:
        active = ", ".join(str(row.get("family_id", "")).strip() for row in running)
        print(f"REFUSE_RUNNING_ACTIVE={active}")
        return 0

    from_status = str(args.from_status).strip().lower()
    to_status = str(args.to_status).strip().lower()

    target = None
    for row in rows:
        if _status(row) != from_status:
            continue
        if _is_dino_tail(row):
            continue
        if _smoke(row) != "ok":
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
