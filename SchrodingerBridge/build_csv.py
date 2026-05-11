from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXP_RE = re.compile(
    r"^S-(?P<skip>[^_]+)_K-(?P<kin>[^_]+)_C-(?P<cyc>[^_]+)_W-(?P<swd>[^_]+)_Col-(?P<col>[^_]+)$",
    re.IGNORECASE,
)


def _as_float(text: str | None) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _epoch_num(epoch_name: str) -> int | None:
    m = re.search(r"(\d+)$", str(epoch_name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _parse_exp_name(name: str) -> dict[str, str]:
    m = EXP_RE.match(name.strip())
    if not m:
        return {"skip_mode": "", "w_kinetic_cfg": "", "w_cycle_cfg": "", "terminal_swd_weight_cfg": "", "w_color_cfg": ""}
    return {
        "skip_mode": m.group("skip"),
        "w_kinetic_cfg": m.group("kin"),
        "w_cycle_cfg": m.group("cyc"),
        "terminal_swd_weight_cfg": m.group("swd"),
        "w_color_cfg": m.group("col"),
    }


def _load_batch_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _config_summary(config: dict[str, object]) -> dict[str, object]:
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    bridge = config.get("bridge", {}) if isinstance(config.get("bridge"), dict) else {}
    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    patch_sizes = bridge.get("swd_patch_sizes")
    if isinstance(patch_sizes, (list, tuple)):
        patch_sizes_value = "|".join(str(v) for v in patch_sizes)
    elif patch_sizes is None:
        patch_sizes_value = ""
    else:
        patch_sizes_value = str(patch_sizes).replace(", ", "|").replace(",", "|")

    summary: dict[str, object] = {
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
        "swd_patch_sizes": patch_sizes_value,
        "swd_num_projections": bridge.get("swd_num_projections"),
        "semantic_swd_num_projections": bridge.get("semantic_swd_num_projections"),
        "batch_size": training.get("batch_size"),
        "learning_rate": training.get("learning_rate"),
        "num_epochs": training.get("num_epochs"),
        "save_interval": training.get("save_interval"),
        "virtual_length_multiplier": (config.get("data", {}) if isinstance(config.get("data"), dict) else {}).get("virtual_length_multiplier"),
    }
    return summary


def _collect_rows(grid_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted([p for p in grid_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        batch_csv = run_dir / "full_eval" / "batch_summary.csv"
        if not batch_csv.is_file():
            continue
        config_path = run_dir / "config.json"
        exp_rows = _load_batch_summary(batch_csv)
        by_epoch: dict[int, dict[str, str]] = {}
        for row in exp_rows:
            ep = _epoch_num(row.get("epoch", ""))
            if ep is not None:
                by_epoch[ep] = row

        lpips_ep1 = _as_float(by_epoch.get(1, {}).get("content_lpips"))
        lpips_ep3 = _as_float(by_epoch.get(3, {}).get("content_lpips"))
        delta_lpips_1_to_3 = None
        if lpips_ep1 is not None and lpips_ep3 is not None:
            delta_lpips_1_to_3 = lpips_ep3 - lpips_ep1

        params = _parse_exp_name(run_dir.name)
        config_summary: dict[str, object] = {}
        if config_path.is_file():
            try:
                config_summary = _config_summary(_load_config(config_path))
            except Exception as exc:
                config_summary = {
                    "config_load_error": f"{type(exc).__name__}: {exc}",
                }

        for row in exp_rows:
            ep = _epoch_num(row.get("epoch", ""))
            merged: dict[str, object] = dict(row)
            merged["epoch_num"] = ep
            merged["delta_lpips_1_to_3"] = delta_lpips_1_to_3
            merged["one_minus_content_lpips"] = (
                1.0 - _as_float(row.get("content_lpips")) if _as_float(row.get("content_lpips")) is not None else None
            )
            merged["one_minus_all_content_lpips"] = (
                1.0 - _as_float(row.get("all_content_lpips")) if _as_float(row.get("all_content_lpips")) is not None else None
            )
            merged["one_minus_transfer_content_lpips"] = (
                1.0 - _as_float(row.get("transfer_content_lpips")) if _as_float(row.get("transfer_content_lpips")) is not None else None
            )
            merged.update(params)
            merged.update(config_summary)
            rows.append(merged)
    return rows


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    return [
        "experiment_id",
        "epoch",
        "epoch_num",
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
        "clip_style",
        "clip_content",
        "content_lpips",
        "one_minus_content_lpips",
        "all_clip_style",
        "all_clip_content",
        "all_content_lpips",
        "one_minus_all_content_lpips",
        "transfer_clip_style",
        "transfer_clip_content",
        "transfer_content_lpips",
        "one_minus_transfer_content_lpips",
        "clip_style_photo_to_art",
        "clip_content_photo_to_art",
        "content_lpips_photo_to_art",
        "delta_lpips_1_to_3",
        "cmmd_all",
        "dino_structure_all",
        "gram_micro_all",
        "gram_macro_all",
        "cmmd_transfer",
        "dino_structure_transfer",
        "gram_micro_transfer",
        "gram_macro_transfer",
        "cmmd_photo_to_art",
        "dino_structure_photo_to_art",
        "gram_micro_photo_to_art",
        "gram_macro_photo_to_art",
        "status",
        "returncode",
        "summary_exists",
    ]


def build_csv(grid_root: Path, output: Path) -> Path:
    rows = _collect_rows(grid_root)
    if not rows:
        raise SystemExit(f"No full_eval batch_summary.csv files found under: {grid_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"Aggregated {len(rows)} rows from {grid_root} -> {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one aggregated CSV from a grid-search run directory and write it to the SchrodingerBridge root."
    )
    parser.add_argument(
        "grid_root",
        help="Grid-search experiment directory, for example: grid_search_3epoch",
    )
    args = parser.parse_args()

    grid_root = Path(args.grid_root)
    if not grid_root.is_absolute():
        grid_root = (ROOT / grid_root).resolve()
    else:
        grid_root = grid_root.resolve()

    output = ROOT / f"{grid_root.name}_scatter.csv"
    build_csv(grid_root=grid_root, output=output)
    print(f"Root CSV ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
