from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ABLATE_SERIES = "ablate_8plus6"

# Run ID, patch-level, skip-level, depth-level
STAGE1_RUNS: list[tuple[str, str, str, str]] = [
    ("arch_1_pM_sC_dH", "pM", "sC", "dH"),
    ("arch_2_pM_sA_dL", "pM", "sA", "dL"),
    ("arch_3_pM_sC_dL", "pM", "sC", "dL"),
    ("arch_4_pM_sA_dH", "pM", "sA", "dH"),
    ("arch_5_pMW_sA_dH", "pMW", "sA", "dH"),
    ("arch_6_pMW_sC_dL", "pMW", "sC", "dL"),
    ("arch_7_pMW_sA_dL", "pMW", "sA", "dL"),
    ("arch_8_pMW_sC_dH", "pMW", "sC", "dH"),
]

PATCH_LEVELS = {
    "pM": [7, 11, 15, 19, 25],
    "pMW": [1, 5, 7, 11, 15],
}

SKIP_LEVELS = {
    "sC": "concat_conv",
    "sA": "add_proj",
}

DEPTH_LEVELS = {
    "dL": {"num_hires_blocks": 2, "num_res_blocks": 1, "num_decoder_blocks": 2, "batch_size": 256},
    "dH": {"num_hires_blocks": 2, "num_res_blocks": 2, "num_decoder_blocks": 3, "batch_size": 256},
}

WEIGHT_SWEEPS = [
    ("weight_0_base", {}),
    ("weight_1_swd_low", {"w_swd": 100.0}),
    ("weight_2_swd_high", {"w_swd": 200.0}),
    ("weight_3_color_low", {"w_color": 20.0}),
    ("weight_4_color_high", {"w_color": 80.0}),
    ("weight_5_id_loose", {"w_identity": 15.0}),
    ("weight_6_id_tight", {"w_identity": 45.0}),
]


def _load_base_config(src_dir: Path) -> dict:
    base_path = src_dir / "config.json"
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_run_script(src_dir: Path, script_name: str, run_ids: list[str]) -> None:
    script_path = src_dir / script_name
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write("setlocal\n")
        f.write("cd /d %~dp0\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        for run_id in run_ids:
            f.write("echo ==================================================\n")
            f.write(f"echo Starting {run_id}...\n")
            f.write("echo ==================================================\n")
            f.write(f"uv run run.py --config config_{run_id}.json\n")
            f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")


def _apply_stage_center_defaults(cfg: dict) -> None:
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("training", {})
    cfg.setdefault("checkpoint", {})

    # Fixed center for stage-1 orthogonal architecture search.
    cfg["training"]["learning_rate"] = 5e-4
    cfg["training"]["num_epochs"] = 60
    cfg["training"]["save_interval"] = 30
    cfg["training"]["full_eval_interval"] = 30
    cfg["training"]["full_eval_on_last_epoch"] = True
    cfg["training"]["scheduler"] = "cosine"
    cfg["training"]["warmup_ratio"] = 0.15
    cfg["training"]["use_gradient_checkpointing"] = True
    cfg["training"]["full_eval_enable_art_fid"] = False
    cfg["training"]["full_eval_enable_kid"] = False

    cfg["loss"]["w_swd"] = 150.0
    cfg["loss"]["w_color"] = 50.0
    cfg["loss"]["w_identity"] = 30.0

    # Keep this ablation family in C-G-W regime.
    cfg["model"]["base_dim"] = int(cfg["model"].get("base_dim", 96))
    cfg["model"]["hires_block_type"] = "conv"
    cfg["model"]["body_block_type"] = "global_attn"
    cfg["model"]["decoder_block_type"] = "window_attn"
    cfg["model"]["style_modulator_type"] = "cross_attn"
    cfg["model"]["style_attn_num_tokens"] = int(cfg["model"].get("style_attn_num_tokens", 64))
    cfg["model"]["style_attn_num_heads"] = int(cfg["model"].get("style_attn_num_heads", 4))
    cfg["model"]["style_attn_sharpen_scale"] = float(cfg["model"].get("style_attn_sharpen_scale", 2.5))
    cfg["model"]["window_attn_window_size"] = int(cfg["model"].get("window_attn_window_size", 8))


def _build_stage1_cfg(base: dict, run_id: str, patch_level: str, skip_level: str, depth_level: str) -> dict:
    cfg = copy.deepcopy(base)
    _apply_stage_center_defaults(cfg)
    cfg["loss"]["swd_patch_sizes"] = list(PATCH_LEVELS[patch_level])
    cfg["model"]["skip_fusion_mode"] = SKIP_LEVELS[skip_level]
    cfg["model"]["num_hires_blocks"] = int(DEPTH_LEVELS[depth_level]["num_hires_blocks"])
    cfg["model"]["num_res_blocks"] = int(DEPTH_LEVELS[depth_level]["num_res_blocks"])
    cfg["model"]["num_decoder_blocks"] = int(DEPTH_LEVELS[depth_level]["num_decoder_blocks"])
    cfg["training"]["batch_size"] = int(DEPTH_LEVELS[depth_level]["batch_size"])
    cfg["checkpoint"]["save_dir"] = f"../{run_id}"
    return cfg


def _build_stage2_cfg(winner_cfg: dict, run_id: str, loss_patch: dict) -> dict:
    cfg = copy.deepcopy(winner_cfg)
    _apply_stage_center_defaults(cfg)
    cfg["loss"].update(loss_patch)
    cfg["checkpoint"]["save_dir"] = f"../{run_id}"
    return cfg


def generate_stage1(src_dir: Path) -> list[str]:
    base = _load_base_config(src_dir)
    run_ids: list[str] = []
    for run_id, patch_level, skip_level, depth_level in STAGE1_RUNS:
        cfg = _build_stage1_cfg(base, run_id, patch_level, skip_level, depth_level)
        _write_json(src_dir / f"config_{run_id}.json", cfg)
        run_ids.append(run_id)
        print(f"generated: config_{run_id}.json")
    return run_ids


def generate_stage2(src_dir: Path, winner: str) -> list[str]:
    # Build winner cfg from stage-1 definition to avoid coupling with stale files.
    by_id = {r[0]: r for r in STAGE1_RUNS}
    if winner not in by_id:
        valid = ", ".join(by_id.keys())
        raise ValueError(f"--stage2_winner must be one of: {valid}")
    _, patch_level, skip_level, depth_level = by_id[winner]

    base = _load_base_config(src_dir)
    winner_cfg = _build_stage1_cfg(base, winner, patch_level, skip_level, depth_level)

    run_ids: list[str] = []
    for run_id, loss_patch in WEIGHT_SWEEPS:
        cfg = _build_stage2_cfg(winner_cfg, run_id, loss_patch)
        _write_json(src_dir / f"config_{run_id}.json", cfg)
        run_ids.append(run_id)
        print(f"generated: config_{run_id}.json (winner={winner})")
    return run_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 8+6 ablation configs (stage1 architecture + stage2 weights).")
    parser.add_argument("--mode", choices=("stage1", "stage2", "all"), default="all")
    parser.add_argument(
        "--stage2_winner",
        type=str,
        default="arch_5_pMW_sA_dH",
        help="Winner run-id from stage1, used as fixed architecture in stage2.",
    )
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    for legacy in ("stage1_arch_l8_2pow3.bat", "stage2_weight_star.bat"):
        p = src_dir / legacy
        if p.exists():
            p.unlink(missing_ok=True)
    all_run_ids: list[str] = []
    if args.mode in {"stage1", "all"}:
        all_run_ids.extend(generate_stage1(src_dir))
    if args.mode in {"stage2", "all"}:
        all_run_ids.extend(generate_stage2(src_dir, winner=str(args.stage2_winner)))

    if all_run_ids:
        _write_run_script(src_dir, f"{ABLATE_SERIES}.bat", all_run_ids)
        print(f"generated: {ABLATE_SERIES}.bat")


if __name__ == "__main__":
    main()
