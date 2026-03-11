#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


ALLOWED_EPOCH_EXTS = {".png", ".csv", ".json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect ablation full_eval artifacts and summarize summary_history metrics."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Cycle-NCE root directory (default: script parent).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/ablation-result).",
    )
    parser.add_argument(
        "--epoch-dir",
        type=str,
        default="epoch_0060",
        help="Epoch directory name under full_eval to collect files from.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="summary_history_metrics.csv",
        help="Output CSV filename under output-dir.",
    )
    parser.add_argument(
        "--copy-full-eval",
        action="store_true",
        help="Copy each experiment full_eval tree into output-dir, excluding full_eval/images.",
    )
    return parser.parse_args()


def parse_experiment_conditions(experiment: str) -> dict[str, str]:
    out: dict[str, str] = {
        "condition_A": "",
        "condition_p": "",
        "condition_id": "",
        "condition_tv": "",
        "condition_extra": "",
    }
    if not experiment.startswith("ablate_"):
        return out

    tail = experiment[len("ablate_") :]
    parts = [p for p in tail.split("_") if p]
    extras: list[str] = []
    for p in parts:
        if re.fullmatch(r"A.+", p):
            if not out["condition_A"]:
                out["condition_A"] = p
                continue
        if re.fullmatch(r"p\d+", p):
            out["condition_p"] = p
            continue
        if re.fullmatch(r"id\d+", p):
            out["condition_id"] = p
            continue
        if re.fullmatch(r"tv\d+", p):
            out["condition_tv"] = p
            continue
        extras.append(p)
    out["condition_extra"] = "_".join(extras)
    return out


def _flatten_latest_or_mean(prefix: str, section: dict[str, Any], row: dict[str, Any]) -> None:
    for k, v in section.items():
        if prefix == "latest" and k == "summary_path":
            continue
        row[f"{prefix}_{k}"] = v


def _flatten_best(section: dict[str, Any], row: dict[str, Any]) -> None:
    for best_key, payload in section.items():
        if not isinstance(payload, dict):
            continue
        metric = best_key.removeprefix("best_")
        row[f"best_{metric}"] = payload.get(metric)
        row[f"best_{metric}_epoch"] = payload.get("epoch")


def _iter_experiment_dirs(root: Path) -> list[Path]:
    exps: list[Path] = []
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        if (p / "full_eval").is_dir():
            exps.append(p)
    return exps


def _copy_full_eval_without_images(src_full_eval: Path, dst_full_eval: Path) -> None:
    def _ignore(dir_path: str, names: list[str]):
        if Path(dir_path).name.lower() == "full_eval" and "images" in names:
            return {"images"}
        return set()

    shutil.copytree(src_full_eval, dst_full_eval, dirs_exist_ok=True, ignore=_ignore)


def collect_artifacts(root: Path, output_dir: Path, epoch_dir: str, copy_full_eval: bool = False) -> list[str]:
    copied: list[str] = []
    for exp_dir in _iter_experiment_dirs(root):
        exp_name = exp_dir.name
        full_eval = exp_dir / "full_eval"
        history = full_eval / "summary_history.json"
        if history.is_file():
            dst = output_dir / f"{exp_name}_summary_history.json"
            shutil.copy2(history, dst)
            copied.append(dst.name)

        if copy_full_eval:
            dst_full_eval = output_dir / exp_name / "full_eval"
            _copy_full_eval_without_images(full_eval, dst_full_eval)
            copied.append(str((Path(exp_name) / "full_eval").as_posix()))

        epoch_path = full_eval / epoch_dir
        if not epoch_path.is_dir():
            continue
        for f in sorted(epoch_path.iterdir()):
            if not f.is_file() or f.suffix.lower() not in ALLOWED_EPOCH_EXTS:
                continue
            dst = output_dir / f"{exp_name}_{f.name}"
            shutil.copy2(f, dst)
            copied.append(dst.name)
    return copied


def build_metrics_csv(output_dir: Path, csv_name: str) -> Path:
    rows: list[dict[str, Any]] = []
    json_files = sorted(output_dir.glob("*_summary_history.json"))

    for jf in json_files:
        experiment = jf.stem.removesuffix("_summary_history")
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)

        row: dict[str, Any] = {"experiment": experiment}
        row.update(parse_experiment_conditions(experiment))
        row["num_rounds"] = data.get("num_rounds")
        row["updated_at"] = data.get("updated_at")

        latest = data.get("latest")
        if isinstance(latest, dict):
            _flatten_latest_or_mean("latest", latest, row)

        mean = data.get("mean")
        if isinstance(mean, dict):
            _flatten_latest_or_mean("mean", mean, row)

        best = data.get("best")
        if isinstance(best, dict):
            _flatten_best(best, row)

        rows.append(row)

    base_cols = [
        "experiment",
        "condition_A",
        "condition_p",
        "condition_id",
        "condition_tv",
        "condition_extra",
        "num_rounds",
        "updated_at",
    ]
    dynamic_cols = sorted({k for r in rows for k in r.keys() if k not in set(base_cols)})
    fieldnames = base_cols + dynamic_cols

    out_csv = output_dir / csv_name
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out_csv


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or (root / "ablation-result")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = collect_artifacts(
        root=root,
        output_dir=output_dir,
        epoch_dir=args.epoch_dir,
        copy_full_eval=bool(args.copy_full_eval),
    )
    out_csv = build_metrics_csv(output_dir=output_dir, csv_name=args.summary_csv)

    print(f"Root: {root}")
    print(f"Output: {output_dir}")
    print(f"Copied files: {len(copied)}")
    print(f"Summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
