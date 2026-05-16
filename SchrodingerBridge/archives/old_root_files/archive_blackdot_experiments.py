from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_summary(eval_dir: Path) -> tuple[str, Path] | None:
    direct_summary = eval_dir / "summary.json"
    if direct_summary.exists():
        return "single", direct_summary

    batch_summary = eval_dir / "batch_summary.csv"
    if batch_summary.exists():
        try:
            with batch_summary.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                row = rows[-1]
                epoch = str(row.get("epoch", "")).strip()
                output_dir = str(row.get("output_dir", "")).strip()
                if epoch and output_dir:
                    summary_path = Path(output_dir) / "summary.json"
                    if summary_path.exists():
                        return epoch, summary_path
        except Exception:
            pass

    candidates = sorted(
        [p for p in eval_dir.glob("epoch_*") if p.is_dir() and (p / "summary.json").exists()],
        key=lambda p: p.name.lower(),
    )
    if candidates:
        latest = candidates[-1]
        return latest.name, latest / "summary.json"
    return None


def _extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    analysis = summary.get("analysis", {}) or {}
    all_pairs = analysis.get("all_pairs_overview", {}) or {}
    transfer = analysis.get("style_transfer_ability", {}) or {}
    photo = analysis.get("photo_to_art_performance", {}) or {}
    identity = analysis.get("identity_reconstruction", {}) or {}
    return {
        "all_clip_style": all_pairs.get("clip_style", ""),
        "all_clip_content": all_pairs.get("clip_content", ""),
        "all_content_lpips": all_pairs.get("content_lpips", ""),
        "transfer_clip_style": transfer.get("clip_style", ""),
        "transfer_clip_content": transfer.get("clip_content", ""),
        "transfer_content_lpips": transfer.get("content_lpips", ""),
        "photo_to_art_clip_style": photo.get("clip_style", ""),
        "photo_to_art_clip_content": photo.get("clip_content", ""),
        "photo_to_art_content_lpips": photo.get("content_lpips", ""),
        "identity_clip_style": identity.get("clip_style", ""),
        "identity_clip_content": identity.get("clip_content", ""),
        "identity_content_lpips": identity.get("content_lpips", ""),
    }


def _build_row(train_dir: Path, eval_dir: Path) -> dict[str, Any]:
    config = _read_json(train_dir / "config.json")
    ablation = config.get("ablation", {}) or {}
    model = config.get("model", {}) or {}
    bridge = config.get("bridge", {}) or {}
    training = config.get("training", {}) or {}

    latest = _find_latest_summary(eval_dir)
    if latest is None:
        raise FileNotFoundError(f"No summary.json found under {eval_dir}")
    latest_epoch, summary_path = latest
    summary = _read_json(summary_path)

    row = {
        "experiment_id": train_dir.name,
        "eval_dir": eval_dir.name,
        "latest_eval_epoch": latest_epoch,
        "train_dir": str(train_dir.resolve()),
        "eval_root": str(eval_dir.resolve()),
        "ablation_name": ablation.get("name", ""),
        "axis": ablation.get("axis", ""),
        "notes": ablation.get("notes", ""),
        "num_epochs": training.get("num_epochs", ""),
        "save_interval": training.get("save_interval", ""),
        "batch_size": training.get("batch_size", ""),
        "learning_rate": training.get("learning_rate", ""),
        "semantic_attn_temperature": model.get("semantic_attn_temperature", ""),
        "terminal_swd_weight": bridge.get("terminal_swd_weight", ""),
        "w_kinetic": bridge.get("w_kinetic", ""),
        "w_cycle": bridge.get("w_cycle", ""),
        "w_repulsive": bridge.get("w_repulsive", ""),
        "w_low_freq": bridge.get("w_low_freq", ""),
        "low_freq_kernel_size": bridge.get("low_freq_kernel_size", ""),
        "semantic_swd_num_projections": bridge.get("semantic_swd_num_projections", ""),
        "swd_use_high_freq": bridge.get("swd_use_high_freq", ""),
    }
    row.update(_extract_metrics(summary))
    return row


def _is_eval_root(path: Path) -> bool:
    return (path / "summary.json").exists() or (path / "batch_summary.csv").exists() or any(
        p.is_dir() and p.name.startswith("epoch_") and (p / "summary.json").exists() for p in path.glob("epoch_*")
    )


def _discover_pairs(root: Path) -> list[tuple[Path, Path]]:
    root = root.resolve()
    children = {child.name: child for child in root.iterdir() if child.is_dir()}
    pairs: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()

    for child in sorted(children.values(), key=lambda p: p.name.lower()):
        if not (child / "config.json").exists():
            continue

        candidates: list[Path] = []
        inplace_eval = child / "full_eval"
        if _is_eval_root(inplace_eval):
            candidates.append(inplace_eval)

        sibling_names = [child.name, child.name.replace("_", "")]
        for sibling_name in sibling_names:
            sibling = children.get(sibling_name)
            if sibling is not None and sibling != child and _is_eval_root(sibling):
                candidates.append(sibling)

        for candidate in candidates:
            pair = (child, candidate)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                break
    return pairs


def _merge_rows(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing:
        merged[(str(row.get("experiment_id", "")), str(row.get("eval_dir", "")))] = row
    for row in new_rows:
        merged[(str(row.get("experiment_id", "")), str(row.get("eval_dir", "")))] = row
    return sorted(merged.values(), key=lambda r: (str(r.get("experiment_id", "")), str(r.get("eval_dir", ""))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive black-dot experiment config+metric rows into a CSV.")
    parser.add_argument("--csv", required=True, help="Target CSV path.")
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("TRAIN_DIR", "EVAL_DIR"),
        help="Train directory and matching eval/full_eval directory. Can be repeated.",
    )
    parser.add_argument(
        "--discover-root",
        action="append",
        default=[],
        help="Auto-discover train/eval pairs under a directory. Supports in-place full_eval and compact eval dirs.",
    )
    args = parser.parse_args()

    explicit_pairs = list(args.pair or [])
    discovered_pairs: list[tuple[str, str]] = []
    for item in args.discover_root:
        for train_dir, eval_dir in _discover_pairs(Path(item)):
            discovered_pairs.append((str(train_dir), str(eval_dir)))

    all_pairs = [*explicit_pairs, *discovered_pairs]
    if not all_pairs:
        raise SystemExit("Provide at least one --pair TRAIN_DIR EVAL_DIR or --discover-root ROOT_DIR.")

    csv_path = Path(args.csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))

    new_rows = [_build_row(Path(train), Path(eval_root)) for train, eval_root in all_pairs]
    rows = _merge_rows(existing, new_rows)

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(csv_path)


if __name__ == "__main__":
    main()
