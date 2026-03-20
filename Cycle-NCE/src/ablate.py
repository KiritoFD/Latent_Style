import argparse
import copy
import json
from pathlib import Path


def load_base_config() -> dict:
    base_path = Path(__file__).resolve().parent / "config_decoder-D-sweetspot.json"
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_legacy_outputs(out_dir: Path) -> None:
    legacy_patterns = [
        "config_abl_*.json",
        "config_scale_*.json",
        "config_baseline.json",
        "ablation_full_60ep.bat",
        "run_color_mode_r16_e40.bat",
    ]
    removed = 0
    for pattern in legacy_patterns:
        for p in out_dir.glob(pattern):
            if p.is_file():
                p.unlink(missing_ok=True)
                removed += 1
    if removed > 0:
        print(f"cleaned legacy files: {removed}")


def create_color_mode_sweep() -> None:
    base = load_base_config()
    out_dir = Path(__file__).resolve().parent
    _cleanup_legacy_outputs(out_dir)

    base.setdefault("model", {})
    base["model"]["ada_mix_rank"] = 16

    base.setdefault("training", {})
    base["training"]["num_epochs"] = 60
    base["training"]["full_eval_interval"] = 20
    base["training"]["save_interval"] = 20
    base["training"]["full_eval_on_last_epoch"] = True

    base.setdefault("loss", {})
    base["loss"]["w_latent_color"] = 0.0
    base["loss"]["w_swd"] = 30.0
    base["loss"]["w_identity"] = 2.0
    base["loss"]["w_color"] = 2.0
    base["loss"]["w_delta_tv"] = 0.5
    base["loss"]["color_eps"] = 1e-6
    base["loss"]["color_legacy_pool"] = 4

    experiments = [
        (
            "color_01_adain_wc2_tv05_r16_e60",
            {
                "color_mode": "pseudo_rgb_adain",
                "w_color": 2.0,
                "w_delta_tv": 0.5,
            },
        ),
        (
            "color_02_adain_wc2_tv20_r16_e60",
            {
                "color_mode": "pseudo_rgb_adain",
                "w_color": 2.0,
                "w_delta_tv": 2.0,
            },
        ),
        (
            "color_03_adain_wc5_tv05_r16_e60",
            {
                "color_mode": "pseudo_rgb_adain",
                "w_color": 5.0,
                "w_delta_tv": 0.5,
            },
        ),
        (
            "color_04_adain_wc5_tv20_r16_e60",
            {
                "color_mode": "pseudo_rgb_adain",
                "w_color": 5.0,
                "w_delta_tv": 2.0,
            },
        ),
        (
            "color_05_hist_wc2_tv05_r16_e60",
            {
                "color_mode": "pseudo_rgb_hist",
                "w_color": 2.0,
                "w_delta_tv": 0.5,
            },
        ),
        (
            "color_06_hist_wc2_tv20_r16_e60",
            {
                "color_mode": "pseudo_rgb_hist",
                "w_color": 2.0,
                "w_delta_tv": 2.0,
            },
        ),
        (
            "color_07_hist_wc5_tv05_r16_e60",
            {
                "color_mode": "pseudo_rgb_hist",
                "w_color": 5.0,
                "w_delta_tv": 0.5,
            },
        ),
        (
            "color_08_hist_wc5_tv20_r16_e60",
            {
                "color_mode": "pseudo_rgb_hist",
                "w_color": 5.0,
                "w_delta_tv": 2.0,
            },
        ),
    ]

    run_bat = out_dir / "run_color_r16_e60.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write("echo Running Color Orthogonal Sweep (rank16, e60)\n")
        f_bat.write("echo ==========================================\n")

        for name, loss_delta in experiments:
            cfg = copy.deepcopy(base)
            cfg.setdefault("loss", {})
            cfg["loss"].update(loss_delta)
            cfg.setdefault("checkpoint", {})
            cfg["checkpoint"]["save_dir"] = f"../{name}"

            cfg_filename = f"config_{name}.json"
            cfg_path = out_dir / cfg_filename
            with open(cfg_path, "w", encoding="utf-8") as f_cfg:
                json.dump(cfg, f_cfg, indent=4, ensure_ascii=False)

            print(
                f"generated: {cfg_filename:52s} | "
                f"mode={cfg['loss']['color_mode']:24s} "
                f"w_color={cfg['loss']['w_color']:.2f} "
                f"w_tv={cfg['loss']['w_delta_tv']:.2f} "
                f"epochs={cfg['training']['num_epochs']} "
                f"full_eval={cfg['training']['full_eval_interval']} "
                f"rank={cfg['model']['ada_mix_rank']}"
            )

            f_bat.write("echo.\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"echo Running Experiment: {name}\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"uv run run.py --config {cfg_filename}\n")
            f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")

        f_bat.write("echo.\n")
        f_bat.write("echo Color orthogonal sweep finished.\n")

    print("\nrun_color_r16_e60.bat has been generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ablation configs and runner scripts")
    parser.add_argument(
        "--preset",
        type=str,
        default="color_modes",
        choices=["color_modes"],
        help="Which preset to generate",
    )
    args = parser.parse_args()
    create_color_mode_sweep()
