from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _epoch_int(name: str) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else -1


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _metric(block: dict[str, Any], key: str) -> float | None:
    value = block.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _row_from_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    analysis = payload.get("analysis") or {}
    transfer = analysis.get("style_transfer_ability") or {}
    all_pairs = analysis.get("all_pairs_overview") or {}
    identity = analysis.get("identity_reconstruction") or {}
    timings = payload.get("timings_sec") or {}
    epoch = path.parent.name
    return {
        "epoch": epoch,
        "epoch_int": _epoch_int(epoch),
        "checkpoint": str(payload.get("checkpoint", "")),
        "timestamp": str(payload.get("timestamp", "")),
        "transfer_clip_style": _metric(transfer, "clip_style"),
        "transfer_content_lpips": _metric(transfer, "content_lpips"),
        "all_pairs_clip_style": _metric(all_pairs, "clip_style"),
        "all_pairs_content_lpips": _metric(all_pairs, "content_lpips"),
        "identity_clip_style": _metric(identity, "clip_style"),
        "identity_content_lpips": _metric(identity, "content_lpips"),
        "eval_wall_total_sec": _metric(timings, "wall_total"),
        "eval_total_sec": _metric(timings, "eval_total"),
        "generation_sec": _metric(timings, "lancet_generation"),
        "vae_decode_sec": _metric(timings, "vae_decode"),
        "summary_path": str(path),
    }


def _scan_rows(run_dir: Path, *, eval_subdir: str) -> list[dict[str, Any]]:
    eval_root = run_dir / eval_subdir
    rows: list[dict[str, Any]] = []
    if not eval_root.is_dir():
        return rows
    for summary_path in sorted(eval_root.glob("epoch_*/summary.json"), key=lambda p: _epoch_int(p.parent.name)):
        rows.append(_row_from_summary(summary_path))
    return rows


def _best_row(rows: list[dict[str, Any]], *, style_key: str, lpips_key: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(style_key) is not None and row.get(lpips_key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: (float(row[style_key]), -float(row[lpips_key]), -int(row["epoch_int"])))


def _latest_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: int(row["epoch_int"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "epoch_int",
        "checkpoint",
        "timestamp",
        "transfer_clip_style",
        "transfer_content_lpips",
        "all_pairs_clip_style",
        "all_pairs_content_lpips",
        "identity_clip_style",
        "identity_content_lpips",
        "eval_wall_total_sec",
        "eval_total_sec",
        "generation_sec",
        "vae_decode_sec",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect round-2 full_eval summaries into a compact curve CSV/JSON.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-subdir", default="full_eval")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    rows = _scan_rows(run_dir, eval_subdir=str(args.eval_subdir))
    csv_out = Path(args.csv_out).expanduser().resolve() if str(args.csv_out).strip() else (run_dir / str(args.eval_subdir) / "clip_lpips_curve.csv")
    json_out = Path(args.json_out).expanduser().resolve() if str(args.json_out).strip() else (run_dir / str(args.eval_subdir) / "curve_summary.json")
    _write_csv(csv_out, rows)
    payload = {
        "run_dir": str(run_dir),
        "eval_subdir": str(args.eval_subdir),
        "row_count": len(rows),
        "latest": _latest_row(rows),
        "best_transfer": _best_row(rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips"),
        "best_all_pairs": _best_row(rows, style_key="all_pairs_clip_style", lpips_key="all_pairs_content_lpips"),
        "rows": rows,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(csv_out)
    print(json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
