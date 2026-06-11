from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from csv_utils import read_csv_rows


def _config_families(row: dict[str, str]) -> tuple[str, str]:
    tokenizer_family = str(row.get("tokenizer_family", "")).strip().lower()
    semantic_supervision_family = str(row.get("semantic_supervision_family", "")).strip().lower()
    if tokenizer_family or semantic_supervision_family:
        return tokenizer_family, semantic_supervision_family
    config_path = Path(str(row.get("config_path", "")).strip())
    if not config_path.is_absolute():
        config_path = (WORKSPACE / config_path).resolve()
    if not config_path.is_file():
        return tokenizer_family, semantic_supervision_family
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return tokenizer_family, semantic_supervision_family
    model_cfg = payload.get("model") or {}
    bridge_cfg = payload.get("bridge") or {}
    tokenizer_family = tokenizer_family or str(model_cfg.get("tokenizer_family", "")).strip().lower()
    semantic_supervision_family = semantic_supervision_family or str(bridge_cfg.get("semantic_supervision_family", "")).strip().lower()
    return tokenizer_family, semantic_supervision_family


def _is_dino_tail(row: dict[str, str]) -> bool:
    tokenizer_family, semantic_supervision_family = _config_families(row)
    family_id = str(row.get("family_id", "")).strip().lower()
    return (
        "dino" in tokenizer_family
        or "dino" in semantic_supervision_family
        or family_id in {"tok_a_dino_dict", "tok_b_cross_image"}
    )


def _status(row: dict[str, str]) -> str:
    return str(row.get("decision_status", "")).strip().lower()


def _smoke(row: dict[str, str]) -> str:
    return str(row.get("switch_smoke_status", "")).strip().lower()


def _fmt_row(row: dict[str, str]) -> str:
    family_id = str(row.get("family_id", "")).strip()
    axis = str(row.get("axis", "")).strip()
    smoke = _smoke(row) or "unknown"
    batch = str(row.get("batch_size", "")).strip() or "?"
    note = "dino-tail" if _is_dino_tail(row) else "non-dino"
    return f"{family_id} [{axis}] smoke={smoke} batch={batch} {note}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the current round1 manifest and report safe queue state.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    rows = read_csv_rows(manifest_csv)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_csv}")

    running = [row for row in rows if _status(row) == "running"]
    planned = [row for row in rows if _status(row) == "planned"]
    reviewing = [row for row in rows if _status(row) == "reviewing"]
    recal = [row for row in rows if _status(row) == "recalibration_needed"]

    planned_non_dino = [row for row in planned if not _is_dino_tail(row)]
    planned_dino = [row for row in planned if _is_dino_tail(row)]
    planned_smoke_ok_non_dino = [row for row in planned_non_dino if _smoke(row) == "ok"]
    planned_smoke_ok_dino = [row for row in planned_dino if _smoke(row) == "ok"]

    payload = {
        "manifest_csv": str(manifest_csv),
        "running": [str(row.get("family_id", "")).strip() for row in running],
        "planned_non_dino": [str(row.get("family_id", "")).strip() for row in planned_non_dino],
        "planned_dino": [str(row.get("family_id", "")).strip() for row in planned_dino],
        "reviewing": [str(row.get("family_id", "")).strip() for row in reviewing],
        "recalibration_needed": [str(row.get("family_id", "")).strip() for row in recal],
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

    next_candidate = payload["next_queue_candidate_if_running_clears"] or "none"
    print(f"next_queue_candidate_if_running_clears: {next_candidate}")
    print(f"dino_tail_block_would_trigger: {payload['dino_tail_block_would_trigger']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
