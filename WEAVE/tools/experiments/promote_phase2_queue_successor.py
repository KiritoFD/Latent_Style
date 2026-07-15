from __future__ import annotations

import argparse
import json
from pathlib import Path

from csv_utils import manifest_fieldnames, read_csv_rows, write_csv_rows
from resolve_phase2_queue_packet import (
    DEFAULT_MANIFEST,
    DEFAULT_VALIDATION,
    resolve_packet,
    resolve_successor_packet,
)


def _normalize_status(value: object) -> str:
    return str(value or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote the next phase2 packet within one lane_class and flip preferred from the current packet to its successor."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--lane-class", required=True)
    parser.add_argument("--current-packet-id", default="")
    parser.add_argument("--next-packet-id", default="")
    parser.add_argument("--current-status", default="")
    parser.add_argument("--next-status", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    validation_json = Path(args.validation_json).expanduser().resolve()
    lane_class = str(args.lane_class).strip()

    if str(args.current_packet_id).strip():
        current = resolve_packet(
            manifest_csv=manifest_csv,
            lane_class=lane_class,
            preferred_only=False,
            validation_json=validation_json,
            require_valid=False,
        )
        if str(current.get("packet_id", "")).strip() != str(args.current_packet_id).strip():
            rows = read_csv_rows(manifest_csv)
            current_row = next(
                (
                    row for row in rows
                    if str(row.get("lane_class", "")).strip().lower() == lane_class.lower()
                    and str(row.get("packet_id", "")).strip() == str(args.current_packet_id).strip()
                ),
                None,
            )
            if current_row is None:
                raise ValueError(f"current packet_id {args.current_packet_id!r} not found in lane_class={lane_class!r}")
            current = {
                "packet_id": str(current_row.get("packet_id", "")).strip(),
            }
    else:
        current = resolve_packet(
            manifest_csv=manifest_csv,
            lane_class=lane_class,
            preferred_only=True,
            validation_json=validation_json,
            require_valid=False,
        )
    current_packet_id = str(current.get("packet_id", "")).strip()
    if not current_packet_id:
        raise ValueError(f"could not resolve current packet for lane_class={lane_class!r}")

    if str(args.next_packet_id).strip():
        next_packet_id = str(args.next_packet_id).strip()
    else:
        successor = resolve_successor_packet(
            manifest_csv=manifest_csv,
            lane_class=lane_class,
            current_packet_id=current_packet_id,
            validation_json=validation_json,
            require_valid=False,
        )
        next_packet_id = str(successor.get("packet_id", "")).strip()
    if not next_packet_id:
        raise ValueError(f"could not resolve successor packet for lane_class={lane_class!r}")

    rows = read_csv_rows(manifest_csv)
    fieldnames = manifest_fieldnames(rows)
    current_status = _normalize_status(args.current_status)
    next_status = _normalize_status(args.next_status)
    changed = False
    for row in rows:
        packet_id = str(row.get("packet_id", "")).strip()
        if str(row.get("lane_class", "")).strip().lower() != lane_class.lower():
            continue
        if packet_id == current_packet_id:
            if row.get("preferred", "").strip().lower() != "no":
                row["preferred"] = "no"
                changed = True
            if current_status and row.get("status", "") != current_status:
                row["status"] = current_status
                changed = True
        elif packet_id == next_packet_id:
            if row.get("preferred", "").strip().lower() != "yes":
                row["preferred"] = "yes"
                changed = True
            if next_status and row.get("status", "") != next_status:
                row["status"] = next_status
                changed = True
        else:
            if row.get("preferred", "").strip().lower() == "yes":
                row["preferred"] = "no"
                changed = True

    payload = {
        "lane_class": lane_class,
        "current_packet_id": current_packet_id,
        "next_packet_id": next_packet_id,
        "current_status": current_status or None,
        "next_status": next_status or None,
        "changed": changed,
        "dry_run": bool(args.dry_run),
        "manifest_csv": str(manifest_csv),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not bool(args.dry_run) and changed:
        write_csv_rows(manifest_csv, rows, fieldnames=fieldnames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
