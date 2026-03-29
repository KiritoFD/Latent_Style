from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ARCH_SERIES_NAME = "arch_ablate"
FINAL_SERIES_NAME = "ca_pram"
BASELINE_NAME = "final_12_base_ref"


ARCH_ABLATIONS = {
    "A1_swin_h2_g1_d2": {
        "model": {
            "hires_block_type": "window_attn",
            "body_block_type": "global_attn",
            "decoder_block_type": "window_attn",
            "num_hires_blocks": 2,
            "num_res_blocks": 1,
            "num_decoder_blocks": 2,
            "window_attn_window_size": 8,
        },
    },
    "A2_swin_h2_g2_d2": {
        "model": {
            "hires_block_type": "window_attn",
            "body_block_type": "global_attn",
            "decoder_block_type": "window_attn",
            "num_hires_blocks": 2,
            "num_res_blocks": 2,
            "num_decoder_blocks": 2,
            "window_attn_window_size": 8,
        },
    },
    "A3_swin_h3_g2_d2": {
        "model": {
            "hires_block_type": "window_attn",
            "body_block_type": "global_attn",
            "decoder_block_type": "window_attn",
            "num_hires_blocks": 3,
            "num_res_blocks": 2,
            "num_decoder_blocks": 2,
            "window_attn_window_size": 8,
        },
    },
    "B1_weaver_h3_g2_d1": {
        "model": {
            "hires_block_type": "window_attn",
            "body_block_type": "global_attn",
            "decoder_block_type": "window_attn",
            "num_hires_blocks": 3,
            "num_res_blocks": 2,
            "num_decoder_blocks": 1,
            "window_attn_window_size": 8,
        },
    },
    "B2_weaver_h2_g2_d1": {
        "model": {
            "hires_block_type": "window_attn",
            "body_block_type": "global_attn",
            "decoder_block_type": "window_attn",
            "num_hires_blocks": 2,
            "num_res_blocks": 2,
            "num_decoder_blocks": 1,
            "window_attn_window_size": 8,
        },
    },
    "B3_weaver_h3_g2_d2": {
        "model": {
            "hires_block_type": "window_attn",
            "body_block_type": "global_attn",
            "decoder_block_type": "window_attn",
            "num_hires_blocks": 3,
            "num_res_blocks": 2,
            "num_decoder_blocks": 2,
            "window_attn_window_size": 8,
        },
    },
    "C1_asym_h1_g2_d2": {
        "model": {
            "hires_block_type": "conv",
            "body_block_type": "global_attn",
            "decoder_block_type": "conv",
            "num_hires_blocks": 1,
            "num_res_blocks": 2,
            "num_decoder_blocks": 2,
            "style_modulator_type": "cross_attn",
            "style_attn_num_tokens": 128,
            "style_attn_sharpen_scale": 2.5,
            "inject_gate_decoder": 1.0,
        },
    },
    "C2_asym_h2_g2_d3": {
        "model": {
            "hires_block_type": "conv",
            "body_block_type": "global_attn",
            "decoder_block_type": "conv",
            "num_hires_blocks": 2,
            "num_res_blocks": 2,
            "num_decoder_blocks": 3,
            "style_modulator_type": "cross_attn",
            "style_attn_num_tokens": 128,
            "style_attn_sharpen_scale": 2.5,
            "inject_gate_decoder": 1.0,
        },
    },
    "D1_cgw_h2_g2_d3_impasto_s3_r12": {
        "model": {
            "base_dim": 96,
            "hires_block_type": "conv",
            "body_block_type": "global_attn",
            "decoder_block_type": "window_attn",
            "num_hires_blocks": 2,
            "num_res_blocks": 2,
            "num_decoder_blocks": 3,
            "window_attn_window_size": 8,
            "style_modulator_type": "cross_attn",
            "style_attn_num_tokens": 64,
            "style_attn_num_heads": 4,
            "style_attn_sharpen_scale": 3.0,
            "residual_gain": 1.2,
            "inject_gate_decoder": 1.0,
        },
        "loss": {
            "w_swd": 150.0,
            "swd_patch_sizes": [1, 3, 5, 7, 11, 15, 23],
            "w_color": 50.0,
            "w_identity": 30.0,
        },
    },
    "D2_cgw_h2_g2_d4_impasto_s3_r12": {
        "model": {
            "base_dim": 96,
            "hires_block_type": "conv",
            "body_block_type": "global_attn",
            "decoder_block_type": "window_attn",
            "num_hires_blocks": 2,
            "num_res_blocks": 2,
            "num_decoder_blocks": 4,
            "window_attn_window_size": 8,
            "style_modulator_type": "cross_attn",
            "style_attn_num_tokens": 64,
            "style_attn_num_heads": 4,
            "style_attn_sharpen_scale": 3.0,
            "residual_gain": 1.2,
            "inject_gate_decoder": 1.0,
        },
        "loss": {
            "w_swd": 150.0,
            "swd_patch_sizes": [1, 3, 5, 7, 11, 15, 23],
            "w_color": 50.0,
            "w_identity": 30.0,
        },
    },
    "D3_cgw_h2_g2_d3_impasto_s4_r15": {
        "model": {
            "base_dim": 96,
            "hires_block_type": "conv",
            "body_block_type": "global_attn",
            "decoder_block_type": "window_attn",
            "num_hires_blocks": 2,
            "num_res_blocks": 2,
            "num_decoder_blocks": 3,
            "window_attn_window_size": 8,
            "style_modulator_type": "cross_attn",
            "style_attn_num_tokens": 64,
            "style_attn_num_heads": 4,
            "style_attn_sharpen_scale": 4.0,
            "residual_gain": 1.5,
            "inject_gate_decoder": 1.0,
        },
        "loss": {
            "w_swd": 150.0,
            "swd_patch_sizes": [1, 3, 5, 7, 11, 15, 23],
            "w_color": 50.0,
            "w_identity": 30.0,
        },
    },
}


FINAL_SENSITIVITY = {
    "final_1_lr4_id35_swd60_c5": {
        "training": {"learning_rate": 4e-4},
        "loss": {"w_identity": 35.0, "w_swd": 60.0, "w_color": 5.0},
    },
    "final_2_lr5_id30_swd80_c2": {
        "training": {"learning_rate": 5e-4},
        "loss": {"w_identity": 30.0, "w_swd": 80.0, "w_color": 2.0},
    },
    "final_3_lr6_id25_swd60_c5": {
        "training": {"learning_rate": 6e-4},
        "loss": {"w_identity": 25.0, "w_swd": 60.0, "w_color": 5.0},
    },
    "final_4_lr8_id30_swd80_c2": {
        "training": {"learning_rate": 8e-4},
        "loss": {"w_identity": 30.0, "w_swd": 80.0, "w_color": 2.0},
    },
    "final_5_lr5_id15_swd120_c5": {
        "training": {"learning_rate": 5e-4},
        "loss": {"w_identity": 15.0, "w_swd": 120.0, "w_color": 5.0},
    },
    "final_6_lr5_id20_swd150_c10": {
        "training": {"learning_rate": 5e-4},
        "loss": {"w_identity": 20.0, "w_swd": 150.0, "w_color": 10.0},
    },
    "final_7_lr8_id15_swd120_c5": {
        "training": {"learning_rate": 8e-4},
        "loss": {"w_identity": 15.0, "w_swd": 120.0, "w_color": 5.0},
    },
    "final_8_lr8_id20_swd150_c10": {
        "training": {"learning_rate": 8e-4},
        "loss": {"w_identity": 20.0, "w_swd": 150.0, "w_color": 10.0},
    },
    "final_9_dim128_tok64": {
        "model": {"base_dim": 128, "style_attn_num_tokens": 64},
        "training": {"batch_size": 224},
    },
    "final_10_dim96_tok128": {
        "model": {"base_dim": 96, "style_attn_num_tokens": 128},
        "training": {"batch_size": 256},
    },
    "final_11_dim128_tok128": {
        "model": {"base_dim": 128, "style_attn_num_tokens": 128},
        "training": {"batch_size": 192},
    },
}


def load_base_config() -> dict:
    base_path = Path(__file__).resolve().parent / "config.json"
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_global_defaults(cfg: dict) -> None:
    cfg.setdefault("loss", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg["training"]["batch_size"] = 256
    cfg["training"]["num_epochs"] = 80
    cfg["training"]["save_interval"] = 40
    cfg["training"]["full_eval_interval"] = 40
    cfg["training"]["full_eval_on_last_epoch"] = True
    cfg["training"]["full_eval_cache_dir"] = "./eval_cache"
    cfg["training"]["full_eval_clip_hf_cache_dir"] = "./eval_cache/hf"
    cfg["training"]["full_eval_image_classifier_path"] = "./eval_cache/eval_style_image_classifier.pt"
    cfg["training"]["full_eval_clip_backend"] = "hf"
    cfg["training"]["learning_rate"] = 5e-4
    cfg["loss"]["w_swd"] = 150.0
    cfg["loss"]["w_color"] = 50.0
    cfg["loss"]["w_identity"] = 30.0
    cfg["loss"]["swd_patch_sizes"] = [1, 5, 7, 15, 19, 23]
    cfg["loss"]["color_mode"] = "latent_decoupled_adain"
    cfg["loss"]["color_latent_channel_weights"] = [2.0, 1.0, 1.0, 1.0]
    cfg["loss"]["color_luma_range_weight"] = 2.0
    cfg["loss"]["color_luma_quantiles"] = [0.1, 0.9]
    cfg["loss"]["color_legacy_pool"] = 4
    cfg["model"]["style_modulator_type"] = str(cfg["model"].get("style_modulator_type", "cross_attn"))
    cfg["model"]["style_attn_num_tokens"] = int(cfg["model"].get("style_attn_num_tokens", 64))
    cfg["model"]["style_attn_num_heads"] = int(cfg["model"].get("style_attn_num_heads", 4))
    cfg["model"]["style_attn_sharpen_scale"] = float(cfg["model"].get("style_attn_sharpen_scale", 2.0))
    cfg["model"]["hires_block_type"] = str(cfg["model"].get("hires_block_type", "conv"))
    cfg["model"]["body_block_type"] = str(cfg["model"].get("body_block_type", "conv"))
    cfg["model"]["decoder_block_type"] = str(cfg["model"].get("decoder_block_type", "conv"))
    cfg["model"]["feature_attn_num_heads"] = int(cfg["model"].get("feature_attn_num_heads", 4))
    cfg["model"]["feature_attn_mlp_ratio"] = float(cfg["model"].get("feature_attn_mlp_ratio", 2.0))
    cfg["model"]["window_attn_window_size"] = int(cfg["model"].get("window_attn_window_size", 8))
    cfg["model"]["residual_gain"] = 1.0


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_run_script(out_dir: Path, script_name: str, config_names: list[str]) -> None:
    run_bat = out_dir / script_name
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        for cfg_name in config_names:
            f_bat.write("echo ==================================================\n")
            f_bat.write(f"echo Starting {cfg_name}...\n")
            f_bat.write("echo ==================================================\n")
            f_bat.write(f"uv run run.py --config config_{cfg_name}.json\n")
            f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")


def _cleanup_generated_outputs(out_dir: Path, stem_prefix: str, script_name: str) -> None:
    removed = 0
    for path in out_dir.glob(f"config_{stem_prefix}*.json"):
        if path.is_file():
            path.unlink(missing_ok=True)
            removed += 1
    script_path = out_dir / script_name
    if script_path.exists():
        script_path.unlink(missing_ok=True)
        removed += 1
    if removed > 0:
        print(f"cleaned old files: {removed}")


def _generate_series(
    *,
    base: dict,
    out_dir: Path,
    series_name: str,
    script_name: str,
    variants: dict[str, dict],
    save_dir_prefix: str,
    baseline_name: str | None = None,
) -> None:
    generated: list[str] = []

    if baseline_name is not None:
        baseline_cfg = copy.deepcopy(base)
        _apply_global_defaults(baseline_cfg)
        baseline_cfg.setdefault("checkpoint", {})
        baseline_cfg["checkpoint"]["save_dir"] = f"../{save_dir_prefix}_{baseline_name}"
        _write_json(out_dir / f"config_{baseline_name}.json", baseline_cfg)
        generated.append(baseline_name)
        print(f"generated: config_{baseline_name}.json")

    for name, patch in variants.items():
        cfg = copy.deepcopy(base)
        _apply_global_defaults(cfg)
        for section in ("model", "loss", "training", "data", "checkpoint", "inference"):
            if section in patch:
                cfg.setdefault(section, {})
                cfg[section].update(patch[section])
        flat_patch = {k: v for k, v in patch.items() if k not in {"model", "loss", "training", "data", "checkpoint", "inference"}}
        if flat_patch:
            cfg.setdefault("model", {})
            cfg["model"].update(flat_patch)
        cfg.setdefault("checkpoint", {})
        cfg["checkpoint"]["save_dir"] = f"../{save_dir_prefix}_{name}"
        _write_json(out_dir / f"config_{name}.json", cfg)
        generated.append(name)
        print(f"generated: config_{name}.json")

    _write_run_script(out_dir, script_name=script_name, config_names=generated)
    print(f"Generated {len(generated)} configs for {series_name} and {script_name}")


def generate_architecture_ablations() -> None:
    out_dir = Path(__file__).resolve().parent
    base = load_base_config()
    _cleanup_generated_outputs(out_dir, stem_prefix="arch_", script_name=f"{ARCH_SERIES_NAME}.bat")
    _generate_series(
        base=base,
        out_dir=out_dir,
        series_name=ARCH_SERIES_NAME,
        script_name=f"{ARCH_SERIES_NAME}.bat",
        variants=ARCH_ABLATIONS,
        save_dir_prefix=ARCH_SERIES_NAME,
    )


def generate_final_sensitivity() -> None:
    out_dir = Path(__file__).resolve().parent
    base = load_base_config()
    _cleanup_generated_outputs(out_dir, stem_prefix="final_", script_name=f"{FINAL_SERIES_NAME}.bat")
    _generate_series(
        base=base,
        out_dir=out_dir,
        series_name=FINAL_SERIES_NAME,
        script_name=f"{FINAL_SERIES_NAME}.bat",
        variants=FINAL_SENSITIVITY,
        save_dir_prefix=FINAL_SERIES_NAME,
        baseline_name=BASELINE_NAME,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate architecture ablations or final sensitivity sweeps.")
    parser.add_argument("--mode", choices=("arch", "final", "all"), default="arch")
    args = parser.parse_args()

    if args.mode in {"arch", "all"}:
        generate_architecture_ablations()
    if args.mode in {"final", "all"}:
        generate_final_sensitivity()


if __name__ == "__main__":
    main()
