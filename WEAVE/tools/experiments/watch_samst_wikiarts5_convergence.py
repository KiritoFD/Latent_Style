from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(float(str(row.get("epoch_num", "-1")))))
    return rows


def _f(row: dict[str, Any], key: str) -> float:
    return float(str(row[key]))


def _i(row: dict[str, Any], key: str) -> int:
    return int(float(str(row[key])))


def _creates_new_pareto(rows: list[dict[str, str]], idx: int, *, style_key: str, lpips_key: str) -> bool:
    target_style = _f(rows[idx], style_key)
    target_lpips = _f(rows[idx], lpips_key)
    for prev in rows[:idx]:
        prev_style = _f(prev, style_key)
        prev_lpips = _f(prev, lpips_key)
        if prev_style >= target_style and prev_lpips <= target_lpips:
            if prev_style > target_style or prev_lpips < target_lpips:
                return False
    return True


def _build_payload(
    rows: list[dict[str, str]],
    *,
    patience: int,
    style_key: str,
    lpips_key: str,
) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "best_epoch": None,
            "best_in_newest_2": False,
            "pareto_epochs": [],
            "last_pareto_epoch": None,
            "since_last_pareto": None,
            "patience": int(patience),
            "converged": False,
            "style_key": str(style_key),
            "lpips_key": str(lpips_key),
        }

    best_idx = 0
    best_score = (_f(rows[0], style_key), -_f(rows[0], lpips_key))
    pareto_indices: list[int] = []
    for idx, row in enumerate(rows):
        score = (_f(row, style_key), -_f(row, lpips_key))
        if score > best_score:
            best_idx = idx
            best_score = score
        if _creates_new_pareto(rows, idx, style_key=style_key, lpips_key=lpips_key):
            pareto_indices.append(idx)

    newest_idx = len(rows) - 1
    last_pareto_idx = pareto_indices[-1]
    since_last_pareto = newest_idx - last_pareto_idx
    best_in_newest_2 = best_idx >= max(0, newest_idx - 1)
    converged = (not best_in_newest_2) and since_last_pareto >= int(patience)
    best_row = rows[best_idx]
    newest_row = rows[newest_idx]
    last_pareto_row = rows[last_pareto_idx]
    return {
        "row_count": len(rows),
        "best_epoch": best_row["epoch"],
        "best_epoch_num": _i(best_row, "epoch_num"),
        "best_transfer_clip_style": _f(best_row, style_key),
        "best_transfer_lpips": _f(best_row, lpips_key),
        "newest_epoch": newest_row["epoch"],
        "newest_epoch_num": _i(newest_row, "epoch_num"),
        "newest_transfer_clip_style": _f(newest_row, style_key),
        "newest_transfer_lpips": _f(newest_row, lpips_key),
        "best_in_newest_2": best_in_newest_2,
        "pareto_epochs": [rows[idx]["epoch"] for idx in pareto_indices],
        "last_pareto_epoch": last_pareto_row["epoch"],
        "last_pareto_epoch_num": _i(last_pareto_row, "epoch_num"),
        "since_last_pareto": since_last_pareto,
        "patience": int(patience),
        "converged": converged,
        "style_key": str(style_key),
        "lpips_key": str(lpips_key),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch the wikiarts5 SaMST CLIP-S/LPIPS curve and emit a convergence JSON.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--curve-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--patience", type=int, default=2, help="Stop after this many consecutive 5-epoch frontier points fail to create a new Pareto improvement.")
    parser.add_argument("--style-key", default="transfer_clip_style")
    parser.add_argument("--lpips-key", default="transfer_content_lpips")
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    curve_csv = (Path(args.curve_csv).expanduser().resolve() if args.curve_csv else run_root / "eval_bundle" / "clip_lpips_curve.csv")
    output_json = (Path(args.output_json).expanduser().resolve() if args.output_json else run_root / "eval_bundle" / "curve_convergence.json")
    cycles = 0
    while True:
        if curve_csv.is_file():
            rows = _read_rows(curve_csv)
            payload = _build_payload(
                rows,
                patience=max(1, int(args.patience)),
                style_key=str(args.style_key),
                lpips_key=str(args.lpips_key),
            )
            payload["curve_csv"] = str(curve_csv)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
