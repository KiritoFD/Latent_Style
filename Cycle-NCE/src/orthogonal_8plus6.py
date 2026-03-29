from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


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
    "dL": {"num_hires_blocks": 2, "num_res_blocks": 1, "num_decoder_blocks": 2},
    "dH": {"num_hires_blocks": 2, "num_res_blocks": 2, "num_decoder_blocks": 3},
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
    path = src_dir / "config.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_run_script(path: Path, run_ids: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
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


def _apply_shared_defaults(cfg: dict) -> None:
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("training", {})
    cfg.setdefault("checkpoint", {})

    # Stage-1 center values (also used as Stage-2 baseline anchor).
    cfg["training"]["num_epochs"] = 80
    cfg["training"]["save_interval"] = 40
    cfg["training"]["full_eval_interval"] = 40
    cfg["training"]["full_eval_on_last_epoch"] = True
    cfg["training"]["learning_rate"] = 5e-4
    cfg["loss"]["w_swd"] = 150.0
    cfg["loss"]["w_color"] = 50.0
    cfg["loss"]["w_identity"] = 30.0

    # Keep C-G-W family fixed; only orthogonal factors change per run.
    cfg["model"]["base_dim"] = int(cfg["model"].get("base_dim", 96))
    cfg["model"]["hires_block_type"] = "window_attn"
    cfg["model"]["body_block_type"] = "global_attn"
    cfg["model"]["decoder_block_type"] = "window_attn"
    cfg["model"]["window_attn_window_size"] = int(cfg["model"].get("window_attn_window_size", 8))
    cfg["model"]["style_modulator_type"] = str(cfg["model"].get("style_modulator_type", "cross_attn"))
    cfg["model"]["style_attn_num_tokens"] = int(cfg["model"].get("style_attn_num_tokens", 64))
    cfg["model"]["style_attn_num_heads"] = int(cfg["model"].get("style_attn_num_heads", 4))
    cfg["model"]["style_attn_sharpen_scale"] = float(cfg["model"].get("style_attn_sharpen_scale", 2.5))
    cfg["model"]["skip_fusion_mode"] = str(cfg["model"].get("skip_fusion_mode", "concat_conv"))

    # Keep full-eval lightweight by default.
    cfg["training"]["full_eval_enable_art_fid"] = False
    cfg["training"]["full_eval_enable_kid"] = False


def _build_stage1_config(base_cfg: dict, run_id: str, patch_level: str, skip_level: str, depth_level: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    _apply_shared_defaults(cfg)
    cfg["loss"]["swd_patch_sizes"] = list(PATCH_LEVELS[patch_level])
    cfg["model"]["skip_fusion_mode"] = SKIP_LEVELS[skip_level]
    cfg["model"].update(DEPTH_LEVELS[depth_level])
    cfg["checkpoint"]["save_dir"] = f"../{run_id}"
    return cfg


def _build_stage2_config(stage1_winner_cfg: dict, run_id: str, loss_patch: dict) -> dict:
    cfg = copy.deepcopy(stage1_winner_cfg)
    _apply_shared_defaults(cfg)
    cfg["loss"].update(loss_patch)
    cfg["checkpoint"]["save_dir"] = f"../{run_id}"
    return cfg


def generate_all(src_dir: Path, stage2_winner: str) -> None:
    base = _load_base_config(src_dir)
    stage1_ids: list[str] = []
    stage1_cfgs: dict[str, dict] = {}
    for run_id, patch_level, skip_level, depth_level in STAGE1_RUNS:
        cfg = _build_stage1_config(base, run_id, patch_level, skip_level, depth_level)
        _write_json(src_dir / f"config_{run_id}.json", cfg)
        stage1_ids.append(run_id)
        stage1_cfgs[run_id] = cfg
        print(f"generated: config_{run_id}.json")

    _write_run_script(src_dir / "stage1_arch_l8_2pow3.bat", stage1_ids)
    print("generated: stage1_arch_l8_2pow3.bat")

    if stage2_winner not in stage1_cfgs:
        valid = ", ".join(stage1_ids)
        raise ValueError(f"--stage2_winner must be one of: {valid}")

    winner_cfg = stage1_cfgs[stage2_winner]
    stage2_ids: list[str] = []
    for run_id, loss_patch in WEIGHT_SWEEPS:
        cfg = _build_stage2_config(winner_cfg, run_id, loss_patch)
        _write_json(src_dir / f"config_{run_id}.json", cfg)
        stage2_ids.append(run_id)
        print(f"generated: config_{run_id}.json (winner={stage2_winner})")

    _write_run_script(src_dir / "stage2_weight_star.bat", stage2_ids)
    print("generated: stage2_weight_star.bat")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Two-Stage Orthogonal Exploration configs: Stage1(8 arch) + Stage2(1+6 weights)."
    )
    parser.add_argument(
        "--stage2_winner",
        type=str,
        default="arch_5_pMW_sA_dH",
        help="Stage1 winner run_id used as the fixed architecture for Stage2 weight sweep.",
    )
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    generate_all(src_dir=src_dir, stage2_winner=str(args.stage2_winner))


if __name__ == "__main__":
    main()
