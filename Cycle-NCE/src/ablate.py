import copy
import json
from pathlib import Path


SERIES_NAME = "cross_attn"


DIMS = {
    "Run_1_lr_high_8e4": {
        "training": {"learning_rate": 8e-4},
    },
    "Run_2_lr_low_2e4": {
        "training": {"learning_rate": 2e-4},
    },
    "Run_3_id_loose_15": {
        "loss": {"w_identity": 15.0},
    },
    "Run_4_id_tight_45": {
        "loss": {"w_identity": 45.0},
    },
    "Run_5_swd_max_200": {
        "loss": {"w_swd": 200.0},
    },
    "Run_6_color_bold_100": {
        "loss": {"w_color": 100.0},
    },
    "Run_7_lum_strict_10": {
        "loss": {"color_luma_range_weight": 10.0},
    },
    "Run_8_arch_old_dict": {
        "model": {"style_modulator_type": "texture_dict"},
    },
}

BASELINE_NAME = "Run_0_Baseline"


def _apply_global_defaults(cfg: dict) -> None:
    cfg.setdefault("loss", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg["training"]["num_epochs"] = 60
    cfg["training"]["save_interval"] = 60
    cfg["training"]["full_eval_interval"] = 60
    cfg["training"]["full_eval_on_last_epoch"] = True
    cfg["loss"]["swd_patch_sizes"] = [1, 3, 5, 9, 15, 25]
    cfg["loss"]["color_mode"] = "latent_decoupled_adain"
    cfg["loss"]["color_latent_channel_weights"] = [2.0, 1.0, 1.0, 1.0]
    cfg["loss"]["color_luma_range_weight"] = float(cfg["loss"].get("color_luma_range_weight", 2.0))
    cfg["loss"]["color_luma_quantiles"] = [0.1, 0.9]
    cfg["loss"]["color_legacy_pool"] = 4
    cfg["model"]["style_modulator_type"] = str(cfg["model"].get("style_modulator_type", "cross_attn"))
    cfg["model"]["residual_gain"] = 1.0


def load_base_config() -> dict:
    base_path = Path(__file__).resolve().parent / "config.json"
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_old_outputs(out_dir: Path) -> None:
    patterns = [
        "config_Exp*_*.json",
        f"{SERIES_NAME}.bat",
    ]
    removed = 0
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
    if removed > 0:
        print(f"cleaned old files: {removed}")


def generate() -> None:
    base = load_base_config()
    out_dir = Path(__file__).resolve().parent
    _cleanup_old_outputs(out_dir)
    run_bat = out_dir / f"{SERIES_NAME}.bat"

    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")

        baseline_cfg = copy.deepcopy(base)
        _apply_global_defaults(baseline_cfg)
        baseline_cfg.setdefault("checkpoint", {})
        baseline_cfg["checkpoint"]["save_dir"] = f"../{SERIES_NAME}_{BASELINE_NAME}"
        baseline_fname = f"config_{BASELINE_NAME}.json"
        with open(out_dir / baseline_fname, "w", encoding="utf-8") as f:
            json.dump(baseline_cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")

        f_bat.write("echo ==================================================\n")
        f_bat.write(f"echo Running {BASELINE_NAME}...\n")
        f_bat.write("echo ==================================================\n")
        f_bat.write(f"uv run run.py --config {baseline_fname}\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")

        print(f"generated: {baseline_fname}")

        for name, patch_val in DIMS.items():
            cfg = copy.deepcopy(base)
            _apply_global_defaults(cfg)
            if "loss" in patch_val:
                cfg["loss"].update(patch_val["loss"])
            if "model" in patch_val:
                cfg["model"].update(patch_val["model"])
            if "training" in patch_val:
                cfg["training"].update(patch_val["training"])
            flat_loss_patch = {k: v for k, v in patch_val.items() if k not in {"loss", "model", "training"}}
            cfg["loss"].update(flat_loss_patch)
            cfg.setdefault("checkpoint", {})
            cfg["checkpoint"]["save_dir"] = f"../{SERIES_NAME}_{name}"

            fname = f"config_{name}.json"
            with open(out_dir / fname, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
                f.write("\n")

            print(f"generated: {fname}")

            f_bat.write("echo ==================================================\n")
            f_bat.write(f"echo Starting {name}...\n")
            f_bat.write("echo ==================================================\n")
            f_bat.write(f"uv run run.py --config {fname}\n")
            f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")

    print(f"Generated 1+{len(DIMS)} sensitivity configs and {run_bat.name}")


if __name__ == "__main__":
    generate()
