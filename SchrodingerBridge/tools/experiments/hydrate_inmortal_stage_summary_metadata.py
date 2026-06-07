from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _norm(value: object) -> str:
    return str(value or "").strip()


def _candidate_keys(name: str) -> list[str]:
    value = _norm(name)
    candidates = [value]
    if value.startswith("aaai2027_"):
        candidates.append(value[len("aaai2027_") :])
    else:
        candidates.append("aaai2027_" + value)
    return [item for idx, item in enumerate(candidates) if item and item not in candidates[:idx]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill stage-summary metadata columns from aaai2027_inmortal_results_master.csv."
    )
    parser.add_argument(
        "--stage-summary",
        type=Path,
        default=Path("docs/experiments/2026-06-07-inmortal-stage-summary.csv"),
    )
    parser.add_argument(
        "--results-master",
        type=Path,
        default=Path("docs/experiments/aaai2027_inmortal_results_master.csv"),
    )
    args = parser.parse_args()

    stage_rows = _read_csv(args.stage_summary.resolve())
    master_rows = _read_csv(args.results_master.resolve())
    master = {_norm(row.get("experiment")): row for row in master_rows if _norm(row.get("experiment"))}
    if not stage_rows:
        print(f"[hydrate_inmortal_stage_summary_metadata] no stage rows in {args.stage_summary.resolve()}")
        return 0

    fieldnames = list(stage_rows[0].keys())
    for row in stage_rows:
        key = _norm(row.get("run_name"))
        src = None
        for candidate in _candidate_keys(key):
            if candidate in master:
                src = master[candidate]
                break
        if not src:
            continue
        for dst_key, src_key in [
            ("family", "family"),
            ("train_batch", "train_batch"),
            ("train_epochs", "train_epochs"),
            ("note_path", "evidence_path"),
        ]:
            if not _norm(row.get(dst_key)):
                row[dst_key] = _norm(src.get(src_key))
        if not _norm(row.get("selection")):
            row["selection"] = _norm(src.get("selection"))

    _write_csv(args.stage_summary.resolve(), stage_rows, fieldnames)
    print(f"[hydrate_inmortal_stage_summary_metadata] updated {args.stage_summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
