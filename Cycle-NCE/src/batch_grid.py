from __future__ import annotations

import argparse
import csv
from pathlib import Path

from grid import (
    GridCell,
    _parse_grid_sources,
    _read_metrics,
    _resolve,
    _save_grid,
    _write_metrics_csv,
)


STYLE_ORDER = ["photo", "Hayao", "monet", "vangogh", "cezanne"]


def _find_reuse_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for p in root.rglob("metrics.csv"):
        parent = p.parent
        if "full_eval" not in {x.name for x in parent.parents} and parent.name != "full_eval_lpips_clip_style":
            continue
        if parent.name.endswith("_grid5"):
            continue
        dirs.append(parent.resolve())
    seen = {}
    for d in dirs:
        seen[str(d)] = d
    return sorted(seen.values(), key=lambda x: str(x).lower())


def _summarize_cells(cells: dict[tuple[str, str, str], GridCell]) -> dict[str, object]:
    clip_vals = [float(c.clip_style) for c in cells.values() if c.clip_style is not None]
    lpips_vals = [float(c.content_lpips) for c in cells.values() if c.content_lpips is not None]
    reused_count = sum(1 for c in cells.values() if c.reused)
    total = len(cells)
    avg_clip = sum(clip_vals) / len(clip_vals) if clip_vals else None
    avg_lpips = sum(lpips_vals) / len(lpips_vals) if lpips_vals else None
    return {
        "cells_total": total,
        "cells_with_clip": len(clip_vals),
        "cells_with_lpips": len(lpips_vals),
        "cells_complete": min(len(clip_vals), len(lpips_vals)),
        "reused_count": reused_count,
        "avg_clip_style": avg_clip,
        "avg_content_lpips": avg_lpips,
        "clip_minus_lpips": (avg_clip - avg_lpips) if (avg_clip is not None and avg_lpips is not None) else None,
        "is_complete_25": bool(len(clip_vals) == total and len(lpips_vals) == total),
    }


def _process_one(
    *,
    reuse_dir: Path,
    grid_dir: Path,
    root: Path,
) -> dict[str, object]:
    rows = _parse_grid_sources(grid_dir, STYLE_ORDER)
    metrics_map = _read_metrics(reuse_dir / "metrics.csv", reuse_dir)
    out_dir = reuse_dir.parent / f"{reuse_dir.name}_grid5"

    cells: dict[tuple[str, str, str], GridCell] = {}
    for src in rows:
        for tgt_style in STYLE_ORDER:
            key = (src.src_style, src.src_image_name, tgt_style)
            cell = metrics_map.get(key)
            if cell is None:
                cell = GridCell(
                    src_style=src.src_style,
                    src_image=src.src_image_name,
                    tgt_style=tgt_style,
                    image_path=None,
                    clip_style=None,
                    content_lpips=None,
                    reused=False,
                )
            cells[key] = cell

    _write_metrics_csv(rows=rows, style_order=STYLE_ORDER, cells=cells, output_csv=out_dir / "grid_metrics.csv")
    _save_grid(rows=rows, style_order=STYLE_ORDER, cells=cells, output_path=out_dir / "grid.png")

    summary = _summarize_cells(cells)
    summary["reuse_dir"] = str(reuse_dir)
    summary["grid_output_dir"] = str(out_dir)
    summary["reuse_dir_rel"] = str(reuse_dir.relative_to(root))
    summary["grid_output_rel"] = str(out_dir.relative_to(root))
    return summary


def _write_summary(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reuse_dir_rel",
        "grid_output_rel",
        "cells_total",
        "cells_with_clip",
        "cells_with_lpips",
        "cells_complete",
        "reused_count",
        "avg_clip_style",
        "avg_content_lpips",
        "clip_minus_lpips",
        "is_complete_25",
        "reuse_dir",
        "grid_output_dir",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate grid5 outputs for every reusable full_eval directory.")
    parser.add_argument("--root", type=str, default="..\\..")
    parser.add_argument("--grid_dir", type=str, default="..\\..\\style_data\\grid5")
    parser.add_argument("--report_dir", type=str, default="..\\grid5_batch_reports")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    root = _resolve(args.root, src_dir)
    grid_dir = _resolve(args.grid_dir, src_dir)
    report_dir = _resolve(args.report_dir, src_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    reuse_dirs = _find_reuse_dirs(root)
    print(f"[scan] found {len(reuse_dirs)} candidate full_eval directories")

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for idx, reuse_dir in enumerate(reuse_dirs, start=1):
        print(f"[{idx}/{len(reuse_dirs)}] {reuse_dir}")
        try:
            summaries.append(_process_one(reuse_dir=reuse_dir, grid_dir=grid_dir, root=root))
        except Exception as exc:
            failures.append({"reuse_dir": str(reuse_dir), "error": str(exc)})
            print(f"  [warn] failed: {exc}")

    summary_csv = report_dir / "grid5_batch_summary.csv"
    _write_summary(summaries, summary_csv)

    failures_csv = report_dir / "grid5_batch_failures.csv"
    with open(failures_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reuse_dir", "error"])
        writer.writeheader()
        for row in failures:
            writer.writerow(row)

    complete = [r for r in summaries if bool(r.get("is_complete_25"))]
    complete_sorted = sorted(
        complete,
        key=lambda r: (
            float(r["clip_minus_lpips"]) if r.get("clip_minus_lpips") is not None else -1e9,
            float(r["avg_clip_style"]) if r.get("avg_clip_style") is not None else -1e9,
            -(float(r["avg_content_lpips"]) if r.get("avg_content_lpips") is not None else 1e9),
        ),
        reverse=True,
    )

    top_k = max(1, int(args.top_k))
    top_csv = report_dir / "grid5_batch_top_by_clip_minus_lpips.csv"
    _write_summary(complete_sorted[:top_k], top_csv)

    print(f"[done] summary: {summary_csv}")
    print(f"[done] failures: {failures_csv}")
    print(f"[done] top-{top_k}: {top_csv}")
    print(f"[done] complete={len(complete)} failed={len(failures)} total={len(reuse_dirs)}")


if __name__ == "__main__":
    main()
