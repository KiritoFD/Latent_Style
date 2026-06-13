from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from csv_utils import read_csv_rows, write_csv_rows, manifest_fieldnames


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "phase2_queue_manifest.csv"
DEFAULT_SNAPSHOT = SB_ROOT / "docs" / "experiments" / "phase2_queue_state_snapshot.json"
DEFAULT_STRUCTURE_PACKET = "vel_tok32_safe_semantic_topogate_k085"
DEFAULT_I2SB_PACKET = "i2sb_tok32_safe_semantic_topogate_sigma0p02_residual"


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _int_or_default(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _formal_row(snapshot: dict) -> tuple[dict[str, object], list[dict[str, object]]]:
    resolved = snapshot.get("resolved_packets")
    if not isinstance(resolved, dict):
        raise ValueError("snapshot missing resolved_packets")
    formal = resolved.get("formal_lane")
    if not isinstance(formal, dict):
        raise ValueError("snapshot missing formal lane")
    remote = snapshot.get("remote_formal_status")
    if not isinstance(remote, dict):
        raise ValueError("snapshot missing remote_formal_status")
    curve = remote.get("curve_summary")
    if not isinstance(curve, dict):
        raise ValueError("snapshot missing remote_formal_status.curve_summary")
    rows = curve.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("snapshot curve_summary has no rows")
    return formal, [row for row in rows if isinstance(row, dict)]


def _pick_parent_row(*, formal: dict[str, object], rows: list[dict[str, object]], mode: str) -> dict[str, object]:
    selected_mode = str(mode).strip().lower()
    transfer_lpips_gate = _float_or_none(formal.get("watch_max_transfer_lpips_for_recovery"))
    allpairs_lpips_gate = _float_or_none(formal.get("watch_max_allpairs_lpips_for_recovery"))
    in_band_rows = []
    for row in rows:
        transfer_lpips = _float_or_none(row.get("transfer_content_lpips"))
        allpairs_lpips = _float_or_none(row.get("all_pairs_content_lpips"))
        if transfer_lpips is None or allpairs_lpips is None:
            continue
        if transfer_lpips_gate is not None and transfer_lpips > transfer_lpips_gate:
            continue
        if allpairs_lpips_gate is not None and allpairs_lpips > allpairs_lpips_gate:
            continue
        in_band_rows.append(row)
    pool = in_band_rows or rows
    if selected_mode == "latest":
        return max(pool, key=lambda row: _int_or_default(row.get("epoch_int"), 0))
    if selected_mode == "best_transfer_style":
        return max(
            pool,
            key=lambda row: (
                _float_or_none(row.get("transfer_clip_style")) or float("-inf"),
                -(_float_or_none(row.get("transfer_content_lpips")) or float("inf")),
                _int_or_default(row.get("epoch_int"), 0),
            ),
        )
    if selected_mode == "best_allpairs_style":
        return max(
            pool,
            key=lambda row: (
                _float_or_none(row.get("all_pairs_clip_style")) or float("-inf"),
                -(_float_or_none(row.get("all_pairs_content_lpips")) or float("inf")),
                _int_or_default(row.get("epoch_int"), 0),
            ),
        )
    if selected_mode == "best_clean_allpairs":
        return min(
            pool,
            key=lambda row: (
                _float_or_none(row.get("all_pairs_content_lpips")) or float("inf"),
                -(_float_or_none(row.get("all_pairs_clip_style")) or float("-inf")),
                -(_float_or_none(row.get("transfer_clip_style")) or float("-inf")),
                -_int_or_default(row.get("epoch_int"), 0),
            ),
        )
    raise ValueError(f"Unsupported parent selection mode: {mode}")


def _find_manifest_row(rows: list[dict[str, str]], packet_id: str) -> dict[str, str]:
    target = str(packet_id).strip()
    for row in rows:
        if str(row.get("packet_id", "")).strip() == target:
            return row
    raise KeyError(f"packet_id not found in manifest: {packet_id}")


def _refresh_note(path: Path, *, formal_packet_id: str, parent_row: dict[str, object], mode: str) -> None:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    marker = "## Parent Refresh"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n\n"
    epoch = str(parent_row.get("epoch", "")).strip() or "unknown"
    checkpoint = str(parent_row.get("checkpoint", "")).strip() or "unknown"
    transfer_style = _float_or_none(parent_row.get("transfer_clip_style"))
    transfer_lpips = _float_or_none(parent_row.get("transfer_content_lpips"))
    allpairs_style = _float_or_none(parent_row.get("all_pairs_clip_style"))
    allpairs_lpips = _float_or_none(parent_row.get("all_pairs_content_lpips"))
    refresh = [
        marker,
        "",
        f"- Source formal packet: `{formal_packet_id}`",
        f"- Selection policy: `{mode}`",
        f"- Selected parent epoch: `{epoch}`",
        f"- Selected parent checkpoint: `{checkpoint}`",
        (
            f"- Selected parent metrics: transfer `{transfer_style:.6f} / {transfer_lpips:.6f}`, "
            f"all-pairs `{allpairs_style:.6f} / {allpairs_lpips:.6f}`"
            if None not in {transfer_style, transfer_lpips, allpairs_style, allpairs_lpips}
            else "- Selected parent metrics: n/a"
        ),
    ]
    path.write_text(text + "\n".join(refresh).rstrip() + "\n", encoding="utf-8")


def _refresh_config(path: Path, *, checkpoint: str) -> None:
    payload = deepcopy(_load_json(path))
    payload.setdefault("training", {})
    payload["training"]["resume_checkpoint"] = checkpoint
    payload["training"]["resume_model_strict"] = False
    payload["training"]["resume_optimizer"] = False
    payload["training"]["resume_training_state"] = False
    payload["training"]["resume_prefer_local_checkpoint"] = True
    _write_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh safe-parent phase2 successor packets from the current formal-lane settled curve.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--snapshot-json", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--structure-packet-id", default=DEFAULT_STRUCTURE_PACKET)
    parser.add_argument("--i2sb-packet-id", default=DEFAULT_I2SB_PACKET)
    parser.add_argument(
        "--parent-mode",
        choices=["best_clean_allpairs", "latest", "best_transfer_style", "best_allpairs_style"],
        default="best_clean_allpairs",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_csv).expanduser().resolve()
    snapshot_path = Path(args.snapshot_json).expanduser().resolve()
    rows = read_csv_rows(manifest_path)
    formal, curve_rows = _formal_row(_load_json(snapshot_path))
    parent_row = _pick_parent_row(formal=formal, rows=curve_rows, mode=str(args.parent_mode))
    checkpoint = str(parent_row.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("Selected parent row has no checkpoint path")
    parent_epoch = str(parent_row.get("epoch", "")).strip() or "unknown"
    formal_packet_id = str(formal.get("packet_id", "")).strip() or "formal_lane"

    target_ids = [str(args.structure_packet_id).strip(), str(args.i2sb_packet_id).strip()]
    for packet_id in target_ids:
        row = _find_manifest_row(rows, packet_id)
        row["parent_packet"] = formal_packet_id
        row["current_best_parent_ckpt"] = checkpoint
        row["current_read"] = (
            f"safe-parent successor refreshed from {formal_packet_id} {parent_epoch} "
            f"via {args.parent_mode}; parent checkpoint now {checkpoint}"
        )
        config_path = Path(str(row.get("config_path", "")).strip()).resolve()
        note_path = Path(str(row.get("note_path", "")).strip()).resolve()
        _refresh_config(config_path, checkpoint=checkpoint)
        _refresh_note(
            note_path,
            formal_packet_id=formal_packet_id,
            parent_row=parent_row,
            mode=str(args.parent_mode),
        )

    write_csv_rows(manifest_path, rows, fieldnames=manifest_fieldnames(rows))
    payload = {
        "formal_packet_id": formal_packet_id,
        "parent_mode": str(args.parent_mode),
        "selected_parent_epoch": parent_epoch,
        "selected_parent_checkpoint": checkpoint,
        "selected_parent_row": parent_row,
        "updated_packets": target_ids,
        "manifest_csv": str(manifest_path),
        "snapshot_json": str(snapshot_path),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
