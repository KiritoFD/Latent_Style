from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze_brightness_color_alignment import (
    _aggregate,
    _build_reference_index,
    _load_cached_style_distributions,
    analyze_one,
)


def _find_image_dirs(exp_root: Path) -> list[Path]:
    return sorted([p for p in exp_root.rglob("images") if p.is_dir()])


def _parse_run_info(images_dir: Path, exp_root: Path) -> dict[str, str | int]:
    rel = images_dir.relative_to(exp_root)
    parts = rel.parts
    experiment = parts[0] if parts else images_dir.parent.name
    epoch = ""
    tokenized = 0
    for part in parts:
        if part.startswith("epoch_"):
            epoch = part
        if "tokenized" in part.lower():
            tokenized = 1
    return {
        "experiment": experiment,
        "run_path": str(images_dir.parent),
        "images_dir": str(images_dir),
        "epoch": epoch,
        "is_tokenized": tokenized,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rank_rows(rows: list[dict[str, object]], key: str, limit: int = 10) -> list[dict[str, object]]:
    valid = [row for row in rows if key in row]
    return sorted(valid, key=lambda r: float(r[key]))[:limit]


def _to_md_table(rows: list[dict[str, object]], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in cols:
            v = row.get(col, "")
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch brightness/color report for Cycle-NCE exp runs against training-data stats.")
    parser.add_argument(
        "--exp_root",
        type=Path,
        default=Path(r"G:\GitHub\Latent_Style\Cycle-NCE\exp"),
        help="Root directory containing experiment folders.",
    )
    parser.add_argument(
        "--stats_dir",
        type=Path,
        default=Path(r"G:\GitHub\Latent_Style\Cycle-NCE\src\eval_cache\train_color_stats"),
        help="Cached training-data color stats directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path(r"G:\GitHub\Latent_Style\Cycle-NCE\exp\brightness_color_vs_train_report"),
        help="Output directory for batch report files.",
    )
    parser.add_argument("--glob", type=str, default="*.jpg", help="Image glob under each images dir.")
    args = parser.parse_args()

    image_dirs = _find_image_dirs(args.exp_root)
    if not image_dirs:
        raise FileNotFoundError(f"No images directories found under {args.exp_root}")

    style_dir_map, style_distributions = _load_cached_style_distributions(args.stats_dir)
    reference_indices = {key: _build_reference_index(path) for key, path in style_dir_map.items()}

    run_rows: list[dict[str, object]] = []
    style_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for images_dir in image_dirs:
        run_info = _parse_run_info(images_dir, args.exp_root)
        generated_paths = sorted(images_dir.glob(args.glob))
        if not generated_paths:
            failures.append({"images_dir": str(images_dir), "error": f"no files matching {args.glob}"})
            continue

        records = []
        for path in generated_paths:
            try:
                records.append(
                    analyze_one(
                        path,
                        style_dir_map,
                        reference_indices,
                        style_distributions,
                        max_samples=4096,
                        seed=42,
                    )
                )
            except Exception as exc:
                failures.append({"images_dir": str(images_dir), "error": f"{path.name}: {exc}"})

        if not records:
            continue

        by_target = _aggregate(records, "target_style")
        overall = {
            **run_info,
            "image_count": len(records),
            "target_style_count": len(by_target),
        }

        numeric_keys = sorted(
            {
                key
                for row in by_target
                for key, value in row.items()
                if isinstance(value, (int, float)) and key not in {"count"}
            }
        )
        for key in numeric_keys:
            vals = [float(row[key]) for row in by_target if key in row]
            if vals:
                overall[key] = sum(vals) / len(vals)
        run_rows.append(overall)

        for row in by_target:
            style_rows.append({**run_info, **row})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = args.output_dir / "runs_vs_train.csv"
    styles_csv = args.output_dir / "runs_by_target_style_vs_train.csv"
    summary_json = args.output_dir / "summary.json"
    report_md = args.output_dir / "report.md"

    _write_csv(runs_csv, run_rows)
    _write_csv(styles_csv, style_rows)

    best_brightness = _rank_rows(run_rows, "style_brightness_mean_abs", limit=10)
    best_brightness_std = _rank_rows(run_rows, "style_brightness_std_abs", limit=10)
    best_rgb = _rank_rows(run_rows, "style_rgb_mean_l2", limit=10)

    report = []
    report.append("# Exp Brightness/Color vs Train Report")
    report.append("")
    report.append(f"- exp_root: `{args.exp_root}`")
    report.append(f"- stats_dir: `{args.stats_dir}`")
    report.append(f"- runs analyzed: {len(run_rows)}")
    report.append(f"- failures: {len(failures)}")
    report.append("")
    report.append("## Best Brightness Mean Alignment")
    report.append(_to_md_table(best_brightness, ["experiment", "epoch", "is_tokenized", "image_count", "style_brightness_mean_abs", "style_brightness_std_abs", "style_rgb_mean_l2"]))
    report.append("")
    report.append("## Best Brightness Std Alignment")
    report.append(_to_md_table(best_brightness_std, ["experiment", "epoch", "is_tokenized", "image_count", "style_brightness_std_abs", "style_brightness_mean_abs", "style_rgb_std_l2"]))
    report.append("")
    report.append("## Best RGB Mean Alignment")
    report.append(_to_md_table(best_rgb, ["experiment", "epoch", "is_tokenized", "image_count", "style_rgb_mean_l2", "style_brightness_mean_abs", "style_chroma_mean_l2"]))
    report.append("")

    report_md.write_text("\n".join(report) + "\n", encoding="utf-8")
    summary_json.write_text(
        json.dumps(
            {
                "exp_root": str(args.exp_root),
                "stats_dir": str(args.stats_dir),
                "run_count": len(run_rows),
                "failure_count": len(failures),
                "failures": failures,
                "best_brightness_mean_alignment": best_brightness,
                "best_brightness_std_alignment": best_brightness_std,
                "best_rgb_mean_alignment": best_rgb,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[DONE] runs={len(run_rows)} failures={len(failures)}")
    print(f"[OUT] {runs_csv}")
    print(f"[OUT] {styles_csv}")
    print(f"[OUT] {summary_json}")
    print(f"[OUT] {report_md}")


if __name__ == "__main__":
    main()
