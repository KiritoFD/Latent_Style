import copy
import json
from pathlib import Path


def load_base_config() -> dict:
    base_path = Path(__file__).resolve().parent / "config.json"
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_sweep() -> None:
    base = load_base_config()
    out_dir = Path(__file__).resolve().parent

    # Final 4-way shootout:
    # focus on HF ratio, patch receptive field, and TV damping under hard numerical defenses.
    # Format: (name, hf_ratio, w_delta_tv, patch_sizes)
    experiments = [
        ("M1-Aggressive-Fine", 5.0, 0.05, [5, 7, 11, 15, 23]),
        ("M2-Smooth-Impasto", 5.0, 0.15, [5, 7, 11, 15, 23]),
        ("M3-Macro-Flowing", 5.0, 0.05, [11, 15, 23]),
        ("M4-Gentle-Balanced", 3.0, 0.05, [5, 7, 11, 15, 23]),
    ]

    run_bat = out_dir / "run_final_shootout_4.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write('set "AGG_ROOT=..\\final-shootout-aggregate"\n')
        f_bat.write("if not exist \"%AGG_ROOT%\" mkdir \"%AGG_ROOT%\"\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write("echo Starting 4-Way Final Shootout (120 Epochs)\n")
        f_bat.write("echo ==========================================\n\n")

        for name, hf_ratio, w_tv, patches in experiments:
            cfg = copy.deepcopy(base)

            # 1) Force-enable hard numerical defenses + full residual power.
            cfg.setdefault("model", {})
            cfg["model"]["residual_gain"] = 1.0
            cfg["model"]["output_clamp_enabled"] = True
            cfg["model"]["decoder_mod_clamp_enabled"] = True
            cfg["model"]["decoder_mag_stabilizer_enabled"] = True

            # 2) Loss axes under test (identity is fixed for clean decoupling).
            cfg.setdefault("loss", {})
            cfg["loss"]["swd_use_high_freq"] = True
            cfg["loss"]["w_identity"] = 0.6
            cfg["loss"]["w_delta_tv"] = float(w_tv)
            cfg["loss"]["swd_hf_weight_ratio"] = float(hf_ratio)
            cfg["loss"]["swd_patch_sizes"] = patches

            # 3) Training schedule (120 Epochs, eval every 40).
            cfg.setdefault("training", {})
            cfg["training"]["num_epochs"] = 120
            cfg["training"]["full_eval_interval"] = 40
            cfg["training"]["full_eval_on_last_epoch"] = True
            cfg["training"]["save_interval"] = 20

            # 4) Output path (prefix keeps downstream collectors compatible).
            cfg.setdefault("checkpoint", {})
            exp_dir = f"ablate_{name}"
            cfg["checkpoint"]["save_dir"] = f"../{exp_dir}"

            cfg_filename = f"config_{name}.json"
            cfg_path = out_dir / cfg_filename
            with open(cfg_path, "w", encoding="utf-8") as f_cfg:
                json.dump(cfg, f_cfg, indent=4, ensure_ascii=False)

            print(
                f"Generated: {cfg_filename:30s} | "
                f"HF={hf_ratio:.1f} TV={w_tv:.2f} Patches={len(patches)} | save_dir={exp_dir}"
            )

            # 写入 Bat 脚本
            f_bat.write("echo.\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"echo Running Experiment: {name}\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"uv run run.py --config {cfg_filename}\n")
            f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
            
            # 同步拷贝评估结果用于聚合
            f_bat.write(
                "robocopy "
                f"\"..\\{exp_dir}\\full_eval\" "
                f"\"%AGG_ROOT%\\{exp_dir}\\full_eval\" "
                "/E /R:1 /W:1 /XD images\n"
            )
            f_bat.write("if %errorlevel% geq 8 exit /b %errorlevel%\n")

        # 最终收集 120 Epoch 的聚合数据
        f_bat.write("\n")
        f_bat.write("echo.\n")
        f_bat.write("echo Aggregating summary_history metrics for Epoch 120 ...\n")
        f_bat.write(
            "uv run python ..\\scripts\\collect_ablation_results.py "
            "--root \"%AGG_ROOT%\" "
            "--output-dir \"%AGG_ROOT%\" "
            "--epoch-dir epoch_0120\n"
        )
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")

    print("\nrun_final_shootout_4.bat has been generated.")


if __name__ == "__main__":
    create_sweep()
