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

    # 8-way ERF/NCE ablation:
    # - ERF-heavy sets around 9/15, cap macro at 19 (remove 23)
    # - include one extreme micro set [2,3,4,5] as counter-example
    # - hold schedule fixed at 80 epochs, eval at 40/80
    experiments = [
        {
            "name": "ablate_E1_Macro19_Rigid_LR14e4",
            "patches": [7, 11, 15, 19],
            "nce_layer_weights": [1.0, 1.0, 1.0],
            "lr": 1.4e-4,
            "w_nce": 2.0,
        },
        {
            "name": "ablate_E2_15Series_Rigid_LR14e4",
            "patches": [9, 11, 13, 15],
            "nce_layer_weights": [1.0, 1.0, 1.0],
            "lr": 1.4e-4,
            "w_nce": 2.0,
        },
        {
            "name": "ablate_E3_15Series_Soft_LR14e4",
            "patches": [9, 11, 13, 15],
            "nce_layer_weights": [0.2, 0.5, 1.0],
            "lr": 1.4e-4,
            "w_nce": 2.0,
        },
        {
            "name": "ablate_E4_9Series_Rigid_LR14e4",
            "patches": [5, 7, 9, 11],
            "nce_layer_weights": [1.0, 1.0, 1.0],
            "lr": 1.4e-4,
            "w_nce": 2.0,
        },
        {
            "name": "ablate_E5_9Series_Soft_LR14e4",
            "patches": [5, 7, 9, 11],
            "nce_layer_weights": [0.2, 0.5, 1.0],
            "lr": 1.4e-4,
            "w_nce": 2.0,
        },
        {
            "name": "ablate_E6_9Series_Free_LR30e4",
            "patches": [5, 7, 9, 11],
            "nce_layer_weights": [0.0, 0.4, 1.0],
            "lr": 3.0e-4,
            "w_nce": 2.0,
        },
        {
            "name": "ablate_E7_15Series_Free_LR14e4_wNCE1",
            "patches": [9, 11, 13, 15],
            "nce_layer_weights": [0.0, 0.4, 1.0],
            "lr": 1.4e-4,
            "w_nce": 1.0,
        },
        {
            "name": "ablate_E8_MicroExtreme_Soft_LR14e4",
            "patches": [2, 3, 4, 5],
            "nce_layer_weights": [0.2, 0.5, 1.0],
            "lr": 1.4e-4,
            "w_nce": 2.0,
        },
    ]

    run_bat = out_dir / "8x80.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write('set "AGG_ROOT=..\\ablate-8x80-aggregate"\n')
        f_bat.write("if not exist \"%AGG_ROOT%\" mkdir \"%AGG_ROOT%\"\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write("echo Starting 8-way ablation (80 Epochs, eval@40/80)\n")
        f_bat.write("echo ==========================================\n\n")

        for exp in experiments:
            cfg = copy.deepcopy(base)

            # Keep model-level defaults from base; only ablate target axes.
            cfg.setdefault("loss", {})
            cfg["loss"]["swd_patch_sizes"] = [int(p) for p in exp["patches"]]
            cfg["loss"]["nce_layer_weights"] = [float(v) for v in exp["nce_layer_weights"]]
            cfg["loss"]["w_nce"] = float(exp["w_nce"])

            cfg.setdefault("training", {})
            cfg["training"]["learning_rate"] = float(exp["lr"])
            cfg["training"]["min_learning_rate"] = float(exp["lr"]) * 0.1
            cfg["training"]["num_epochs"] = 80
            cfg["training"]["full_eval_interval"] = 40
            cfg["training"]["full_eval_on_last_epoch"] = True
            cfg["training"]["save_interval"] = 20

            cfg.setdefault("checkpoint", {})
            exp_dir = exp["name"]
            cfg["checkpoint"]["save_dir"] = f"../{exp_dir}"

            cfg_filename = f"config_{exp['name']}.json"
            cfg_path = out_dir / cfg_filename
            with open(cfg_path, "w", encoding="utf-8") as f_cfg:
                json.dump(cfg, f_cfg, indent=4, ensure_ascii=False)

            print(
                f"Generated: {cfg_filename:45s} | "
                f"patch={exp['patches']} nce={exp['nce_layer_weights']} "
                f"lr={exp['lr']:.1e} w_nce={exp['w_nce']:.1f}"
            )

            f_bat.write("echo.\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"echo Running Experiment: {exp['name']}\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"uv run run.py --config {cfg_filename}\n")
            f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
            f_bat.write(
                "robocopy "
                f"\"..\\{exp_dir}\\full_eval\" "
                f"\"%AGG_ROOT%\\{exp_dir}\\full_eval\" "
                "/E /R:1 /W:1 /XD images\n"
            )
            f_bat.write("if %errorlevel% geq 8 exit /b %errorlevel%\n")

        # Collect intermediate and final snapshots.
        f_bat.write("\n")
        f_bat.write("echo.\n")
        f_bat.write("echo Aggregating epoch_0040 metrics ...\n")
        f_bat.write(
            "uv run python ..\\scripts\\collect_ablation_results.py "
            "--root \"%AGG_ROOT%\" "
            "--output-dir \"%AGG_ROOT%\" "
            "--epoch-dir epoch_0040 "
            "--summary-csv summary_history_metrics_e040.csv\n"
        )
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write("echo Aggregating epoch_0080 metrics ...\n")
        f_bat.write(
            "uv run python ..\\scripts\\collect_ablation_results.py "
            "--root \"%AGG_ROOT%\" "
            "--output-dir \"%AGG_ROOT%\" "
            "--epoch-dir epoch_0080 "
            "--summary-csv summary_history_metrics_e080.csv\n"
        )
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")

    print("\n8x80.bat has been generated.")


if __name__ == "__main__":
    create_sweep()
