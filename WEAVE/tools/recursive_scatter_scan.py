from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "next_round_80_scatter.csv"
DEFAULT_OUTPUT = ROOT / "next_round_80_scatter_unified.csv"

CONFIG_COLUMNS = [
    "skip_routing_mode",
    "skip_fusion_mode",
    "semantic_attn_temperature",
    "style_attn_temperature",
    "base_dim",
    "num_res_blocks",
    "w_kinetic",
    "w_cycle",
    "terminal_swd_weight",
    "w_color",
    "w_low_freq",
    "w_repulsive",
    "w_nce",
    "low_freq_kernel_size",
    "swd_use_high_freq",
    "swd_patch_sizes",
    "swd_num_projections",
    "semantic_swd_num_projections",
    "batch_size",
    "learning_rate",
    "num_epochs",
    "save_interval",
    "virtual_length_multiplier",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _to_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _template_header(template: Path) -> list[str]:
    with template.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def _empty_row(header: list[str]) -> dict[str, str]:
    return {key: "" for key in header}


def _flatten_config(config: dict[str, Any], header: list[str]) -> dict[str, str]:
    row = _empty_row(header)
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    bridge = config.get("bridge", {}) if isinstance(config.get("bridge"), dict) else {}
    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    data = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    ablation = config.get("ablation", {}) if isinstance(config.get("ablation"), dict) else {}
    checkpoint = config.get("checkpoint", {}) if isinstance(config.get("checkpoint"), dict) else {}

    values = {
        "skip_routing_mode": model.get("skip_routing_mode"),
        "skip_fusion_mode": model.get("skip_fusion_mode"),
        "semantic_attn_temperature": model.get("semantic_attn_temperature"),
        "style_attn_temperature": model.get("style_attn_temperature"),
        "base_dim": model.get("base_dim"),
        "num_res_blocks": model.get("num_res_blocks"),
        "w_kinetic": bridge.get("w_kinetic"),
        "w_cycle": bridge.get("w_cycle"),
        "terminal_swd_weight": bridge.get("terminal_swd_weight"),
        "w_color": bridge.get("w_color"),
        "w_low_freq": bridge.get("w_low_freq"),
        "w_repulsive": bridge.get("w_repulsive"),
        "w_nce": bridge.get("w_nce"),
        "low_freq_kernel_size": bridge.get("low_freq_kernel_size"),
        "swd_use_high_freq": bridge.get("swd_use_high_freq"),
        "swd_patch_sizes": bridge.get("swd_patch_sizes"),
        "swd_num_projections": bridge.get("swd_num_projections"),
        "semantic_swd_num_projections": bridge.get("semantic_swd_num_projections"),
        "batch_size": training.get("batch_size"),
        "learning_rate": training.get("learning_rate"),
        "num_epochs": training.get("num_epochs"),
        "save_interval": training.get("save_interval"),
        "virtual_length_multiplier": data.get("virtual_length_multiplier"),
        "experiment_id": _first_present(ablation.get("name"), Path(str(checkpoint.get("save_dir", ""))).name or None),
    }
    for key, value in values.items():
        if key in row:
            row[key] = _to_cell(value)
    return row


def _find_nearest_config(start: Path, root: Path) -> Path | None:
    for parent in [start, *start.parents]:
        if root not in [parent, *parent.parents] and parent != root:
            break
        # The project root config is often just a scratch/default config. For
        # recursive result consolidation, it is safer to leave a summary
        # unconfigured than to attach this root config to an unrelated run.
        if parent == root:
            break
        candidate = parent / "config.json"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _find_named_config(summary_path: Path, root: Path) -> Path | None:
    parts = summary_path.relative_to(root).parts
    if "full_eval" in parts:
        idx = parts.index("full_eval")
        if idx + 1 < len(parts):
            eval_name = parts[idx + 1]
            eval_base = root.joinpath(*parts[:idx])
            sibling = eval_base / eval_name / "config.json"
            if sibling.exists():
                return sibling
    names = list(parts)
    for name in reversed(names):
        if name.startswith("epoch_") or name in {"full_eval", "images", "logs"}:
            continue
        for base in [
            root / "ablation_destructive_7epoch" / "configs",
            root / "next_round_80",
            root / "screening_grid_3epoch_108",
            root / "pareto_probe_4",
        ]:
            candidate = base / f"{name}.json"
            if candidate.exists():
                return candidate
    return None


def _epoch_from_path(path: Path) -> tuple[str, str]:
    match = re.search(r"epoch_(\d+)", str(path))
    if not match:
        return "", ""
    epoch_num = int(match.group(1))
    return f"epoch_{epoch_num:04d}", str(epoch_num)


def _experiment_from_summary_path(summary_path: Path, root: Path) -> str:
    rel_parts = summary_path.relative_to(root).parts
    if "full_eval" in rel_parts:
        idx = rel_parts.index("full_eval")
        if idx > 0:
            return rel_parts[idx - 1]
    for marker in ("step_size_sweep_epoch7", "residual_scale_sweep_epoch7"):
        if marker in rel_parts:
            idx = rel_parts.index(marker)
            if idx + 1 < len(rel_parts):
                return f"{rel_parts[idx - 1]}__{rel_parts[idx + 1]}"
    if len(rel_parts) > 1:
        return rel_parts[-2]
    return summary_path.stem


def _summary_metric_block(summary: dict[str, Any], key: str) -> dict[str, Any]:
    analysis = summary.get("analysis", {})
    if not isinstance(analysis, dict):
        return {}
    block = analysis.get(key, {})
    return block if isinstance(block, dict) else {}


def _fill_metrics(row: dict[str, str], summary: dict[str, Any]) -> None:
    all_pairs = _summary_metric_block(summary, "all_pairs_overview")
    transfer = _summary_metric_block(summary, "style_transfer_ability")
    photo = _summary_metric_block(summary, "photo_to_art_performance")

    metric_map = {
        "clip_style": all_pairs.get("clip_style"),
        "clip_content": all_pairs.get("clip_content"),
        "content_lpips": all_pairs.get("content_lpips"),
        "all_clip_style": all_pairs.get("clip_style"),
        "all_clip_content": all_pairs.get("clip_content"),
        "all_content_lpips": all_pairs.get("content_lpips"),
        "transfer_clip_style": transfer.get("clip_style"),
        "transfer_clip_content": transfer.get("clip_content"),
        "transfer_content_lpips": transfer.get("content_lpips"),
        "clip_style_photo_to_art": photo.get("clip_style"),
        "clip_content_photo_to_art": photo.get("clip_content"),
        "content_lpips_photo_to_art": photo.get("content_lpips"),
        "cmmd_all": all_pairs.get("cmmd"),
        "dino_structure_all": all_pairs.get("dino_structure"),
        "gram_micro_all": all_pairs.get("gram_micro"),
        "gram_macro_all": all_pairs.get("gram_macro"),
        "cmmd_transfer": transfer.get("cmmd"),
        "dino_structure_transfer": transfer.get("dino_structure"),
        "gram_micro_transfer": transfer.get("gram_micro"),
        "gram_macro_transfer": transfer.get("gram_macro"),
        "cmmd_photo_to_art": photo.get("cmmd"),
        "dino_structure_photo_to_art": photo.get("dino_structure"),
        "gram_micro_photo_to_art": photo.get("gram_micro"),
        "gram_macro_photo_to_art": photo.get("gram_macro"),
    }
    for key, value in metric_map.items():
        if key in row:
            row[key] = _to_cell(value)

    for lpips_col, inv_col in [
        ("content_lpips", "one_minus_content_lpips"),
        ("all_content_lpips", "one_minus_all_content_lpips"),
        ("transfer_content_lpips", "one_minus_transfer_content_lpips"),
    ]:
        if lpips_col in row and inv_col in row and row.get(lpips_col):
            try:
                row[inv_col] = str(1.0 - float(row[lpips_col]))
            except ValueError:
                pass


def _rows_from_scatter_csv(path: Path, header: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return rows
            if not {"experiment_id", "epoch"}.issubset(set(reader.fieldnames)):
                return rows
            overlap = set(reader.fieldnames).intersection(header)
            if len(overlap) < 10:
                return rows
            for src in reader:
                row = _empty_row(header)
                for key in overlap:
                    row[key] = src.get(key, "") or ""
                if any(row.values()):
                    rows.append(row)
    except Exception:
        return []
    return rows


def _rows_from_summary(summary_path: Path, root: Path, header: list[str]) -> list[dict[str, str]]:
    summary = _read_json(summary_path)
    if not summary or "analysis" not in summary:
        return []

    config_path = _find_nearest_config(summary_path.parent, root) or _find_named_config(summary_path, root)
    config = _read_json(config_path) if config_path else None
    row = _flatten_config(config or {}, header)

    experiment_id = row.get("experiment_id") or _experiment_from_summary_path(summary_path, root)
    epoch, epoch_num = _epoch_from_path(summary_path)
    row["experiment_id"] = experiment_id
    row["epoch"] = epoch
    row["epoch_num"] = epoch_num
    _fill_metrics(row, summary)
    if "status" in row:
        row["status"] = "ok"
    if "returncode" in row:
        row["returncode"] = "0"
    if "summary_exists" in row:
        row["summary_exists"] = "True"
    return [row]


def _rows_from_config(config_path: Path, root: Path, header: list[str]) -> list[dict[str, str]]:
    config = _read_json(config_path)
    if not config or not any(k in config for k in ("model", "bridge", "training")):
        return []
    row = _flatten_config(config, header)
    if not row.get("experiment_id"):
        row["experiment_id"] = config_path.stem if config_path.name != "config.json" else config_path.parent.name
    if "status" in row:
        row["status"] = "config_only"
    if "summary_exists" in row:
        row["summary_exists"] = "False"
    return [row]


def _merge_score(row: dict[str, str]) -> int:
    score = sum(1 for value in row.values() if value not in {"", None})
    if row.get("summary_exists") == "True":
        score += 100
    if row.get("status") == "ok":
        score += 20
    return score


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    if row.get("experiment_id") or row.get("epoch"):
        return (row.get("experiment_id", ""), row.get("epoch", ""))
    return tuple(row.get(col, "") for col in CONFIG_COLUMNS if col in row)


def _dedupe_rows(rows: list[dict[str, str]], header: list[str]) -> list[dict[str, str]]:
    merged: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        normalized = _empty_row(header)
        for key in header:
            normalized[key] = row.get(key, "") or ""
        key = _row_key(normalized)
        if key not in merged:
            merged[key] = normalized
            continue
        current = merged[key]
        if _merge_score(normalized) > _merge_score(current):
            better, worse = normalized, current
        else:
            better, worse = current, normalized
        for col in header:
            if not better.get(col) and worse.get(col):
                better[col] = worse[col]
        merged[key] = better

    by_exp: dict[str, dict[int, dict[str, str]]] = {}
    for row in merged.values():
        try:
            epoch_num = int(row.get("epoch_num") or "")
        except ValueError:
            continue
        by_exp.setdefault(row.get("experiment_id", ""), {})[epoch_num] = row
    for exp_rows in by_exp.values():
        if 1 not in exp_rows or 3 not in exp_rows:
            continue
        try:
            delta = float(exp_rows[3].get("content_lpips", "")) - float(exp_rows[1].get("content_lpips", ""))
        except ValueError:
            continue
        for row in exp_rows.values():
            if "delta_lpips_1_to_3" in row and not row.get("delta_lpips_1_to_3"):
                row["delta_lpips_1_to_3"] = str(delta)

    def sort_key(row: dict[str, str]) -> tuple[str, int, str]:
        try:
            epoch_num = int(row.get("epoch_num") or "0")
        except ValueError:
            epoch_num = 0
        return (row.get("experiment_id", ""), epoch_num, row.get("epoch", ""))

    return sorted(merged.values(), key=sort_key)


def scan(
    root: Path,
    template: Path,
    include_config_only: bool,
    exclude_paths: set[Path] | None = None,
) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
    header = _template_header(template)
    rows: list[dict[str, str]] = []
    counts = {"csv_rows": 0, "summary_rows": 0, "config_rows": 0}
    excludes = {p.resolve() for p in (exclude_paths or set())}

    for csv_path in sorted(root.rglob("*.csv")):
        if csv_path.resolve() in excludes:
            continue
        if csv_path.resolve() == template.resolve():
            # Keep the original too; it is useful as an input source.
            pass
        csv_rows = _rows_from_scatter_csv(csv_path, header)
        rows.extend(csv_rows)
        counts["csv_rows"] += len(csv_rows)

    for summary_path in sorted(root.rglob("summary.json")):
        summary_rows = _rows_from_summary(summary_path, root, header)
        rows.extend(summary_rows)
        counts["summary_rows"] += len(summary_rows)

    if include_config_only:
        for config_path in sorted(root.rglob("*.json")):
            if config_path.name == "summary.json":
                continue
            config_rows = _rows_from_config(config_path, root, header)
            rows.extend(config_rows)
            counts["config_rows"] += len(config_rows)

    deduped = _dedupe_rows(rows, header)
    counts["deduped_rows"] = len(deduped)
    counts["raw_rows"] = len(rows)
    return header, deduped, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursively consolidate SB JSON/CSV configs and metrics into scatter CSV format.")
    parser.add_argument("--root", type=Path, default=ROOT, help="SchrodingerBridge root to scan recursively.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="CSV whose header defines the output format.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--no-config-only", action="store_true", help="Do not emit rows for configs without summaries.")
    args = parser.parse_args()

    root = args.root.resolve()
    template = args.template.resolve()
    output = args.output.resolve()
    header, rows, counts = scan(
        root,
        template,
        include_config_only=not args.no_config_only,
        exclude_paths={output},
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"output": str(output), **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
