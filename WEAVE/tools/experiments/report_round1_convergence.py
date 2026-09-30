from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _f(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _metric(row: dict[str, str], primary: str, *aliases: str) -> float | None:
    value = _f(row.get(primary))
    if value is not None:
        return value
    for alias in aliases:
        value = _f(row.get(alias))
        if value is not None:
            return value
    return None


def _epoch_idx_map(rows: list[dict[str, str]]) -> dict[str, int]:
    return {str(row.get("epoch", "")).strip(): idx for idx, row in enumerate(rows)}


def _best_epoch(rows: list[dict[str, str]], *, maximize_key: str, minimize_key: str) -> str:
    best_row = None
    best_score = None
    for row in rows:
        style = _metric(row, maximize_key)
        lpips = _metric(row, minimize_key)
        if style is None or lpips is None:
            continue
        score = (style, -lpips)
        if best_score is None or score > best_score:
            best_row = row
            best_score = score
    if best_row is None:
        raise RuntimeError(f"No valid rows for keys {maximize_key} / {minimize_key}")
    return str(best_row["epoch"])


def _dominates(a: dict[str, str], b: dict[str, str]) -> bool:
    a_transfer_style = _metric(a, "transfer_clip_style")
    a_transfer_lpips = _metric(a, "transfer_content_lpips", "transfer_lpips")
    a_full_style = _metric(a, "full_clip_style", "allpairs_clip_style")
    a_full_lpips = _metric(a, "full_content_lpips", "allpairs_content_lpips", "allpairs_lpips")
    b_transfer_style = _metric(b, "transfer_clip_style")
    b_transfer_lpips = _metric(b, "transfer_content_lpips", "transfer_lpips")
    b_full_style = _metric(b, "full_clip_style", "allpairs_clip_style")
    b_full_lpips = _metric(b, "full_content_lpips", "allpairs_content_lpips", "allpairs_lpips")
    values = [
        a_transfer_style,
        a_transfer_lpips,
        a_full_style,
        a_full_lpips,
        b_transfer_style,
        b_transfer_lpips,
        b_full_style,
        b_full_lpips,
    ]
    if any(v is None for v in values):
        return False
    ge = (
        a_transfer_style >= b_transfer_style
        and a_transfer_lpips <= b_transfer_lpips
        and a_full_style >= b_full_style
        and a_full_lpips <= b_full_lpips
    )
    strict = (
        a_transfer_style > b_transfer_style
        or a_transfer_lpips < b_transfer_lpips
        or a_full_style > b_full_style
        or a_full_lpips < b_full_lpips
    )
    return ge and strict


def _pareto_indices(rows: list[dict[str, str]]) -> list[int]:
    indices: list[int] = []
    for idx, row in enumerate(rows):
        dominated = False
        for prev in rows[:idx]:
            if _dominates(prev, row):
                dominated = True
                break
        if not dominated:
            indices.append(idx)
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize round-1 fast-eval convergence from clip_lpips_curve.csv.")
    parser.add_argument("--curve-csv", required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--flat-tail-window", type=int, default=4)
    parser.add_argument("--flat-eps-style", type=float, default=0.005)
    parser.add_argument("--flat-eps-lpips", type=float, default=0.018)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    curve_path = Path(args.curve_csv).resolve()
    rows = list(csv.DictReader(curve_path.open("r", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"Empty curve csv: {curve_path}")

    epoch_index = _epoch_idx_map(rows)
    best_transfer_clip_epoch = _best_epoch(rows, maximize_key="transfer_clip_style", minimize_key="transfer_content_lpips")
    best_transfer_lpips_epoch = _best_epoch(rows, maximize_key="transfer_clip_style", minimize_key="transfer_content_lpips")
    best_allpairs_clip_epoch = _best_epoch(rows, maximize_key="full_clip_style", minimize_key="full_content_lpips")
    best_allpairs_lpips_epoch = _best_epoch(rows, maximize_key="full_clip_style", minimize_key="full_content_lpips")
    best_transfer_lpips_epoch = min(
        rows,
        key=lambda row: (
            _metric(row, "transfer_content_lpips", "transfer_lpips") if _metric(row, "transfer_content_lpips", "transfer_lpips") is not None else float("inf"),
            -(_metric(row, "transfer_clip_style") if _metric(row, "transfer_clip_style") is not None else float("-inf")),
        ),
    )["epoch"]
    best_allpairs_lpips_epoch = min(
        rows,
        key=lambda row: (
            _metric(row, "full_content_lpips", "allpairs_content_lpips", "allpairs_lpips") if _metric(row, "full_content_lpips", "allpairs_content_lpips", "allpairs_lpips") is not None else float("inf"),
            -(_metric(row, "full_clip_style", "allpairs_clip_style") if _metric(row, "full_clip_style", "allpairs_clip_style") is not None else float("-inf")),
        ),
    )["epoch"]
    best_epochs = {
        str(best_transfer_clip_epoch),
        str(best_transfer_lpips_epoch),
        str(best_allpairs_clip_epoch),
        str(best_allpairs_lpips_epoch),
    }
    newest_idx = len(rows) - 1
    newest_epochs = {str(rows[idx]["epoch"]) for idx in range(max(0, newest_idx - 1), newest_idx + 1)}
    best_in_newest_2 = bool(best_epochs & newest_epochs)
    pareto_idx = _pareto_indices(rows)
    last_pareto_idx = pareto_idx[-1]
    since_last_pareto = newest_idx - last_pareto_idx
    tail_window = max(2, int(args.flat_tail_window))
    tail = rows[max(0, newest_idx - (tail_window - 1)) : newest_idx + 1]
    tail_transfer_style = [_metric(r, "transfer_clip_style") for r in tail]
    tail_transfer_lpips = [_metric(r, "transfer_content_lpips", "transfer_lpips") for r in tail]
    tail_full_style = [_metric(r, "full_clip_style", "allpairs_clip_style") for r in tail]
    tail_full_lpips = [_metric(r, "full_content_lpips", "allpairs_content_lpips", "allpairs_lpips") for r in tail]
    tail_flat = False
    if len(tail_transfer_style) >= 2 and all(
        v is not None for v in tail_transfer_style + tail_transfer_lpips + tail_full_style + tail_full_lpips
    ):
        tail_flat = (
            max(tail_transfer_style) - min(tail_transfer_style) <= float(args.flat_eps_style)
            and max(tail_transfer_lpips) - min(tail_transfer_lpips) <= float(args.flat_eps_lpips)
            and max(tail_full_style) - min(tail_full_style) <= float(args.flat_eps_style)
            and max(tail_full_lpips) - min(tail_full_lpips) <= float(args.flat_eps_lpips)
        )

    converged = (not best_in_newest_2) and since_last_pareto >= int(args.patience) and tail_flat
    best_epoch = str(best_transfer_clip_epoch)
    best_index = int(epoch_index.get(best_epoch, 0))
    payload = {
        "curve_csv": str(curve_path),
        "row_count": len(rows),
        "best_epoch": best_epoch,
        "best_index": best_index,
        "best_transfer_clip_epoch": str(best_transfer_clip_epoch),
        "best_transfer_lpips_epoch": str(best_transfer_lpips_epoch),
        "best_allpairs_clip_epoch": str(best_allpairs_clip_epoch),
        "best_allpairs_lpips_epoch": str(best_allpairs_lpips_epoch),
        "pareto_epochs": [str(rows[idx]["epoch"]) for idx in pareto_idx],
        "last_pareto_epoch": str(rows[last_pareto_idx]["epoch"]),
        "newest_epoch": rows[newest_idx]["epoch"],
        "since_best": newest_idx - best_index,
        "since_last_pareto": since_last_pareto,
        "best_in_newest_2": best_in_newest_2,
        "tail_flat": tail_flat,
        "tail_window": tail_window,
        "flat_eps_style": float(args.flat_eps_style),
        "flat_eps_lpips": float(args.flat_eps_lpips),
        "patience": int(args.patience),
        "criterion": "joint_transfer_allpairs_pareto",
        "converged": converged,
    }
    output_json = Path(args.output_json).resolve() if str(args.output_json).strip() else curve_path.with_name("round1_convergence.json")
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
