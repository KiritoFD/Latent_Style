import copy
import json
from pathlib import Path


def load_base_config() -> dict:
    base_path = Path(__file__).resolve().parent / "config_decoder-D-sweetspot.json"
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_sweep() -> None:
    base = load_base_config()
    out_dir = Path(__file__).resolve().parent

    # Shared defaults for the full ablation matrix.
    base.setdefault("model", {})
    base["model"]["lift_channels"] = 128
    base["model"]["base_dim"] = 96
    base["model"]["ada_mix_rank"] = 32
    base["model"]["num_decoder_blocks"] = 1
    base["model"]["residual_gain"] = 1.0

    base.setdefault("loss", {})
    base["loss"]["swd_distance_mode"] = "cdf"
    base["loss"]["swd_use_high_freq"] = True
    base["loss"]["w_color"] = 0.5
    base["loss"]["w_identity"] = 0.3
    base["loss"]["w_delta_tv"] = 0.005

    base.setdefault("training", {})
    base["training"]["batch_size"] = 320
    base["training"]["learning_rate"] = 1.4e-4
    base["training"]["min_learning_rate"] = 1.4e-5
    base["training"]["num_epochs"] = 60
    base["training"]["full_eval_interval"] = 20
    base["training"]["save_interval"] = 20
    base["training"]["use_gradient_checkpointing"] = True
    base["training"]["full_eval_on_last_epoch"] = True

    # Full architecture ablation matrix aligned to methodology chapters.
    experiments = [
        (
            "abl_heavy_decoder",
            {
                "model": {"num_decoder_blocks": 6},
                "training": {"batch_size": 256},
            },
        ),
        (
            "abl_no_residual",
            {
                "model": {"residual_gain": 0.0},
            },
        ),
        (
            "abl_vanilla_gn",
            {
                "model": {"ada_mix_rank": 1},
            },
        ),
        (
            "abl_no_skip_filter",
            {
                "model": {"style_skip_content_retention_boost": 1.0},
            },
        ),
        (
            "abl_no_id",
            {
                "loss": {"w_identity": 0.0},
            },
        ),
        (
            "abl_hard_sort",
            {
                "loss": {"swd_distance_mode": "sort"},
            },
        ),
        (
            "abl_no_hf_swd",
            {
                "loss": {"swd_use_high_freq": False},
            },
        ),
        (
            "abl_no_color",
            {
                "loss": {"w_color": 0.0},
            },
        ),
        (
            "abl_no_tv",
            {
                "loss": {"w_delta_tv": 0.0},
            },
        ),
        (
            "scale_c64",
            {
                "model": {"lift_channels": 64, "base_dim": 48, "ada_mix_rank": 16},
                "training": {"batch_size": 512},
            },
        ),
        (
            "scale_c256",
            {
                "model": {"lift_channels": 256, "base_dim": 192, "ada_mix_rank": 64},
                "training": {"batch_size": 128},
            },
        ),
        (
            "baseline",
            {
                "model": {"lift_channels": 128, "base_dim": 96, "ada_mix_rank": 32, "num_decoder_blocks": 1},
                "training": {"batch_size": 320},
            },
        ),
    ]

    run_bat = out_dir / "ablation_full_60ep.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write('set "AGG_ROOT=..\\ablation-full-aggregate"\n')
        f_bat.write('set "COOLDOWN_SEC=8"\n')
        f_bat.write("if not exist \"%AGG_ROOT%\" mkdir \"%AGG_ROOT%\"\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write("echo Starting 12 Full-Stack Ablation Experiments\n")
        f_bat.write("echo ==========================================\n\n")

        for name, delta in experiments:
            cfg = copy.deepcopy(base)

            # Apply one orthogonal scan delta.
            for section, section_delta in delta.items():
                cfg.setdefault(section, {})
                cfg[section].update(section_delta)

            cfg.setdefault("checkpoint", {})
            cfg["checkpoint"]["save_dir"] = f"../{name}"

            cfg_filename = f"config_{name}.json"
            cfg_path = out_dir / cfg_filename
            with open(cfg_path, "w", encoding="utf-8") as f_cfg:
                json.dump(cfg, f_cfg, indent=4, ensure_ascii=False)

            print(
                f"generated: {cfg_filename:42s} | "
                f"epochs={cfg['training']['num_epochs']} "
                f"bs={cfg['training']['batch_size']} "
                f"eval={cfg['training']['full_eval_interval']} "
                f"lift={cfg['model']['lift_channels']} "
                f"base={cfg['model']['base_dim']} "
                f"rank={cfg['model']['ada_mix_rank']} "
                f"dec={cfg['model']['num_decoder_blocks']} "
                f"res={cfg['model']['residual_gain']:.2f} "
                f"w_id={cfg['loss']['w_identity']:.2f}"
            )

            f_bat.write("echo.\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"echo Running Experiment: {name}\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"uv run run.py --config {cfg_filename}\n")
            f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
            f_bat.write(
                "robocopy "
                f"\"..\\{name}\\full_eval\" "
                f"\"%AGG_ROOT%\\{name}\\full_eval\" "
                "/E /R:1 /W:1 /XD images\n"
            )
            f_bat.write("if %errorlevel% geq 8 exit /b %errorlevel%\n")
            f_bat.write("echo Cooling down %COOLDOWN_SEC%s to reduce VRAM fragmentation...\n")
            f_bat.write("timeout /t %COOLDOWN_SEC% /nobreak >nul\n")

        f_bat.write("\n")
        f_bat.write("echo.\n")
        f_bat.write("echo Aggregating summary_history metrics ...\n")
        f_bat.write(
            "uv run python ..\\scripts\\collect_ablation_results.py "
            "--root \"%AGG_ROOT%\" "
            "--output-dir \"%AGG_ROOT%\" "
            "--epoch-dir epoch_0060\n"
        )
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")

    print("\nablation_full_60ep.bat has been generated.")


if __name__ == "__main__":
    create_sweep()
