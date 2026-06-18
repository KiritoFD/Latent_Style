from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

OBJECTIVE_STYLE_TARGET = 0.74
OBJECTIVE_LPIPS_TARGET = 0.30


def _f(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _metric(row: dict[str, str], key: str) -> float | None:
    value = _f(row.get(key))
    if value is not None:
        return value
    # Transfer-only curves intentionally leave all-pairs/identity columns empty.
    # Use the transfer surface as the convergence authority in that mode rather
    # than failing metadata refresh after every checkpoint.
    fallback = {
        "all_pairs_clip_style": "transfer_clip_style",
        "all_pairs_content_lpips": "transfer_content_lpips",
    }.get(key)
    if fallback is not None:
        return _f(row.get(fallback))
    return None


def _epoch_idx_map(rows: list[dict[str, str]]) -> dict[str, int]:
    return {str(row.get("epoch", "")).strip(): idx for idx, row in enumerate(rows)}


def _objective_gap(style: float | None, lpips: float | None, *, style_target: float, lpips_target: float) -> float:
    if style is None or lpips is None:
        return 1e9
    return max(0.0, float(style_target) - float(style)) + max(0.0, float(lpips) - float(lpips_target))


def _best_epoch(rows: list[dict[str, str]], *, style_key: str, lpips_key: str) -> str:
    best_row = None
    best_score = None
    for row in rows:
        style = _metric(row, style_key)
        lpips = _metric(row, lpips_key)
        if style is None or lpips is None:
            continue
        score = (style, -lpips)
        if best_score is None or score > best_score:
            best_row = row
            best_score = score
    if best_row is None:
        raise RuntimeError(f"No valid rows for keys {style_key} / {lpips_key}")
    return str(best_row["epoch"])


def _dominates(a: dict[str, str], b: dict[str, str]) -> bool:
    a_transfer_style = _metric(a, "transfer_clip_style")
    a_transfer_lpips = _metric(a, "transfer_content_lpips")
    a_all_style = _metric(a, "all_pairs_clip_style")
    a_all_lpips = _metric(a, "all_pairs_content_lpips")
    b_transfer_style = _metric(b, "transfer_clip_style")
    b_transfer_lpips = _metric(b, "transfer_content_lpips")
    b_all_style = _metric(b, "all_pairs_clip_style")
    b_all_lpips = _metric(b, "all_pairs_content_lpips")
    values = [
        a_transfer_style,
        a_transfer_lpips,
        a_all_style,
        a_all_lpips,
        b_transfer_style,
        b_transfer_lpips,
        b_all_style,
        b_all_lpips,
    ]
    if any(v is None for v in values):
        return False
    ge = (
        a_transfer_style >= b_transfer_style
        and a_transfer_lpips <= b_transfer_lpips
        and a_all_style >= b_all_style
        and a_all_lpips <= b_all_lpips
    )
    strict = (
        a_transfer_style > b_transfer_style
        or a_transfer_lpips < b_transfer_lpips
        or a_all_style > b_all_style
        or a_all_lpips < b_all_lpips
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


def build_convergence_payload(
    rows: list[dict[str, str]],
    *,
    curve_path: Path,
    patience: int,
    min_epochs: int,
    flat_tail_window: int,
    flat_eps_style: float,
    flat_eps_lpips: float,
    objective_style_target: float,
    objective_lpips_target: float,
) -> dict[str, object]:
    if not rows:
        raise RuntimeError(f"Empty curve csv: {curve_path}")

    epoch_index = _epoch_idx_map(rows)
    best_transfer_epoch = _best_epoch(rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips")
    best_all_pairs_epoch = _best_epoch(rows, style_key="all_pairs_clip_style", lpips_key="all_pairs_content_lpips")
    best_epochs = {str(best_transfer_epoch), str(best_all_pairs_epoch)}
    newest_idx = len(rows) - 1
    newest_epochs = {str(rows[idx]["epoch"]) for idx in range(max(0, newest_idx - 1), newest_idx + 1)}
    best_in_newest_2 = bool(best_epochs & newest_epochs)
    pareto_idx = _pareto_indices(rows)
    last_pareto_idx = pareto_idx[-1]
    since_last_pareto = newest_idx - last_pareto_idx
    tail_window = max(2, int(flat_tail_window))
    tail = rows[max(0, newest_idx - (tail_window - 1)) : newest_idx + 1]
    tail_transfer_style = [_metric(r, "transfer_clip_style") for r in tail]
    tail_transfer_lpips = [_metric(r, "transfer_content_lpips") for r in tail]
    tail_all_style = [_metric(r, "all_pairs_clip_style") for r in tail]
    tail_all_lpips = [_metric(r, "all_pairs_content_lpips") for r in tail]
    tail_flat = False
    if len(tail_transfer_style) >= 2 and all(v is not None for v in tail_transfer_style + tail_transfer_lpips + tail_all_style + tail_all_lpips):
        tail_flat = (
            max(tail_transfer_style) - min(tail_transfer_style) <= float(flat_eps_style)
            and max(tail_transfer_lpips) - min(tail_transfer_lpips) <= float(flat_eps_lpips)
            and max(tail_all_style) - min(tail_all_style) <= float(flat_eps_style)
            and max(tail_all_lpips) - min(tail_all_lpips) <= float(flat_eps_lpips)
        )
    pareto_converged = (not best_in_newest_2) and since_last_pareto >= int(patience) and tail_flat

    objective_best_idx = min(
        range(len(rows)),
        key=lambda idx: (
            _objective_gap(
                _metric(rows[idx], "transfer_clip_style"),
                _metric(rows[idx], "transfer_content_lpips"),
                style_target=objective_style_target,
                lpips_target=objective_lpips_target,
            ),
            -float(_metric(rows[idx], "transfer_clip_style") or 0.0),
            float(_metric(rows[idx], "transfer_content_lpips") or 1.0),
            idx,
        ),
    )
    objective_best_epoch = str(rows[objective_best_idx]["epoch"])
    objective_best_gap = _objective_gap(
        _metric(rows[objective_best_idx], "transfer_clip_style"),
        _metric(rows[objective_best_idx], "transfer_content_lpips"),
        style_target=objective_style_target,
        lpips_target=objective_lpips_target,
    )
    objective_epochs_since_best = newest_idx - objective_best_idx
    objective_patience_converged = (
        len(rows) >= int(min_epochs)
        and objective_best_idx >= 0
        and objective_epochs_since_best >= int(patience)
    )
    stop_ready = bool(pareto_converged or objective_patience_converged)
    if pareto_converged:
        stop_reason = "joint_transfer_allpairs_pareto"
    elif objective_patience_converged:
        stop_reason = "objective_gap_patience"
    else:
        stop_reason = ""

    best_epoch = str(best_transfer_epoch)
    best_index = int(epoch_index.get(best_epoch, 0))
    return {
        "curve_csv": str(curve_path),
        "row_count": len(rows),
        "best_epoch": best_epoch,
        "best_index": best_index,
        "best_transfer_epoch": str(best_transfer_epoch),
        "best_all_pairs_epoch": str(best_all_pairs_epoch),
        "pareto_epochs": [str(rows[idx]["epoch"]) for idx in pareto_idx],
        "last_pareto_epoch": str(rows[last_pareto_idx]["epoch"]),
        "newest_epoch": rows[newest_idx]["epoch"],
        "since_best": newest_idx - best_index,
        "since_last_pareto": since_last_pareto,
        "best_in_newest_2": best_in_newest_2,
        "tail_flat": tail_flat,
        "tail_window": tail_window,
        "flat_eps_style": float(flat_eps_style),
        "flat_eps_lpips": float(flat_eps_lpips),
        "patience": int(patience),
        "min_epochs": int(min_epochs),
        "criterion": "joint_transfer_allpairs_pareto",
        "converged": pareto_converged,
        "objective_style_target": float(objective_style_target),
        "objective_lpips_target": float(objective_lpips_target),
        "objective_best_epoch": objective_best_epoch,
        "objective_best_index": int(objective_best_idx),
        "objective_best_gap": float(objective_best_gap),
        "objective_epochs_since_best": int(objective_epochs_since_best),
        "objective_patience_converged": bool(objective_patience_converged),
        "stop_ready": bool(stop_ready),
        "stop_reason": stop_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize round-2 CLIP/LPIPS convergence from clip_lpips_curve.csv.")
    parser.add_argument("--curve-csv", required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--flat-tail-window", type=int, default=4)
    parser.add_argument("--flat-eps-style", type=float, default=0.005)
    parser.add_argument("--flat-eps-lpips", type=float, default=0.018)
    parser.add_argument("--objective-style-target", type=float, default=OBJECTIVE_STYLE_TARGET)
    parser.add_argument("--objective-lpips-target", type=float, default=OBJECTIVE_LPIPS_TARGET)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    curve_path = Path(args.curve_csv).resolve()
    rows = list(csv.DictReader(curve_path.open("r", encoding="utf-8")))
    payload = build_convergence_payload(
        rows,
        curve_path=curve_path,
        patience=int(args.patience),
        min_epochs=int(args.min_epochs),
        flat_tail_window=int(args.flat_tail_window),
        flat_eps_style=float(args.flat_eps_style),
        flat_eps_lpips=float(args.flat_eps_lpips),
        objective_style_target=float(args.objective_style_target),
        objective_lpips_target=float(args.objective_lpips_target),
    )
    output_json = Path(args.output_json).resolve() if str(args.output_json).strip() else curve_path.with_name("round2_convergence.json")
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
