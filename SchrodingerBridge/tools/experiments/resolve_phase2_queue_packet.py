from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = SB_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_schema import load_experiment_config
from csv_utils import read_csv_rows


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "phase2_queue_manifest.csv"
DEFAULT_VALIDATION = SB_ROOT / "docs" / "experiments" / "phase2_queue_manifest_validation.json"


def _priority(value: object) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return 10**9


def _is_closed_or_retired_status(value: object) -> bool:
    status = str(value or "").strip().lower()
    return status.startswith("closed") or status in {"superseded", "retired", "archived"}


def _load_validation_map(path: Path) -> tuple[bool, dict[str, dict[str, object]]]:
    if not path.is_file():
        return False, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, {}
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return bool(payload.get("ok")) if isinstance(payload, dict) else False, {}
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        packet_id = str(row.get("packet_id", "")).strip()
        if packet_id:
            by_id[packet_id] = row
    return bool(payload.get("ok")) if isinstance(payload, dict) else False, by_id


def _resolve_validation_row(
    *,
    packet_id: str,
    validation_json: Path | None,
    require_valid: bool,
) -> tuple[bool, dict[str, object] | None]:
    validation_ok = False
    validation_row: dict[str, object] | None = None
    if validation_json is not None:
        validation_ok, validation_map = _load_validation_map(validation_json)
        validation_row = validation_map.get(str(packet_id).strip())
        if require_valid:
            if not validation_ok:
                raise ValueError(f"validation snapshot is not ok: {validation_json}")
            if not validation_row or not bool(validation_row.get("ok")):
                raise ValueError(
                    f"selected packet failed validation: {packet_id} from {validation_json}"
                )
    return validation_ok, validation_row


def _packet_payload_from_row(
    *,
    selected: dict[str, str],
    lane: str,
    validation_json: Path | None,
    require_valid: bool,
) -> dict[str, object]:
    packet_id = str(selected.get("packet_id", "")).strip()
    validation_ok, validation_row = _resolve_validation_row(
        packet_id=packet_id,
        validation_json=validation_json,
        require_valid=require_valid,
    )
    config_path = Path(str(selected.get("config_path", "")).strip()).resolve()
    note_path = Path(str(selected.get("note_path", "")).strip()).resolve()
    cfg = load_experiment_config(config_path)
    save_dir = Path(str(cfg.checkpoint.save_dir)).name
    return {
        "packet_id": packet_id,
        "lane_class": lane,
        "priority_in_class": _priority(selected.get("priority_in_class")),
        "preferred": str(selected.get("preferred", "")).strip().lower() == "yes",
        "status": str(selected.get("status", "")).strip(),
        "formal_eligible": str(selected.get("formal_eligible", "")).strip().lower() == "yes",
        "tokenizer_profile": str(selected.get("tokenizer_profile", "")).strip(),
        "config_path": str(config_path),
        "note_path": str(note_path),
        "run_name": save_dir,
        "current_read": str(selected.get("current_read", "")).strip(),
        "watch_min_settled_epoch": str(selected.get("watch_min_settled_epoch", "")).strip(),
        "watch_min_allpairs_style_recovery": str(selected.get("watch_min_allpairs_style_recovery", "")).strip(),
        "watch_max_allpairs_lpips_for_recovery": str(selected.get("watch_max_allpairs_lpips_for_recovery", "")).strip(),
        "watch_min_transfer_style_recovery": str(selected.get("watch_min_transfer_style_recovery", "")).strip(),
        "watch_max_transfer_lpips_for_recovery": str(selected.get("watch_max_transfer_lpips_for_recovery", "")).strip(),
        "watch_handoff_mode": str(selected.get("watch_handoff_mode", "")).strip(),
        "validation_snapshot_ok": validation_ok,
        "validation_row_ok": bool(validation_row.get("ok")) if isinstance(validation_row, dict) else None,
    }


def resolve_packet(
    *,
    manifest_csv: Path,
    lane_class: str,
    preferred_only: bool,
    validation_json: Path | None,
    require_valid: bool,
) -> dict[str, object]:
    rows = read_csv_rows(manifest_csv)
    lane = str(lane_class).strip().lower()
    candidates = [
        row
        for row in rows
        if str(row.get("lane_class", "")).strip().lower() == lane
    ]
    if preferred_only:
        preferred = [
            row for row in candidates
            if str(row.get("preferred", "")).strip().lower() == "yes"
        ]
        if preferred:
            candidates = preferred
    if not candidates:
        raise ValueError(f"no packet found for lane_class={lane_class!r}")
    candidates.sort(key=lambda row: (_priority(row.get("priority_in_class")), str(row.get("packet_id", ""))))
    selected = candidates[0]
    return _packet_payload_from_row(
        selected=selected,
        lane=lane,
        validation_json=validation_json,
        require_valid=require_valid,
    )


def resolve_successor_packet(
    *,
    manifest_csv: Path,
    lane_class: str,
    current_packet_id: str,
    validation_json: Path | None,
    require_valid: bool,
) -> dict[str, object]:
    rows = read_csv_rows(manifest_csv)
    lane = str(lane_class).strip().lower()
    current_id = str(current_packet_id).strip()
    candidates = [
        row
        for row in rows
        if str(row.get("lane_class", "")).strip().lower() == lane
    ]
    if not candidates:
        raise ValueError(f"no packet found for lane_class={lane_class!r}")
    candidates.sort(key=lambda row: (_priority(row.get("priority_in_class")), str(row.get("packet_id", ""))))
    current_index = next(
        (idx for idx, row in enumerate(candidates) if str(row.get("packet_id", "")).strip() == current_id),
        None,
    )
    if current_index is None:
        raise ValueError(f"current packet_id {current_packet_id!r} not found for lane_class={lane_class!r}")
    successor = None
    for row in candidates[current_index + 1 :]:
        if _is_closed_or_retired_status(row.get("status")):
            continue
        successor = row
        break
    if successor is None:
        raise ValueError(
            f"no successor packet found after {current_packet_id!r} for lane_class={lane_class!r}"
        )
    return _packet_payload_from_row(
        selected=successor,
        lane=lane,
        validation_json=validation_json,
        require_valid=require_valid,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve the preferred packet for one phase2 queue lane.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lane-class", default="formal_lane")
    parser.add_argument("--allow-nonpreferred", action="store_true")
    parser.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--no-require-valid", action="store_true")
    args = parser.parse_args()

    payload = resolve_packet(
        manifest_csv=Path(args.manifest_csv).expanduser().resolve(),
        lane_class=str(args.lane_class),
        preferred_only=not bool(args.allow_nonpreferred),
        validation_json=Path(args.validation_json).expanduser().resolve() if str(args.validation_json).strip() else None,
        require_valid=not bool(args.no_require_valid),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
