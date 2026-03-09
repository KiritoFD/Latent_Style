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
    train_base = base.get("training", {})
    batch_size = int(train_base.get("batch_size", 160))
    sweep_tag = "nce"

    # 6-way orthogonal ablation around NCE layer strategy and SWD patch scale.
    # Keep base LR/w_nce by default; only override target axes.
    experiments = [
        {
            "name": "A1_Deep_Only",
            "nce_layer_weights": [0.0, 0.0, 1.0],
            "patches": [15, 23],
            "w_delta_tv": 0.0,
            "w_identity": 0.5,
        },
        {
            "name": "A2_Shallow_Only",
            "nce_layer_weights": [1.0, 0.0, 0.0],
            "patches": [5, 7],
            "w_delta_tv": 0.0,
            "w_identity": 0.5,
        },
        {
            "name": "A3_Patch_Coarse",
            "nce_layer_weights": [0.0, 0.5, 1.0],
            "patches": [15, 23, 31],
            "w_delta_tv": 0.0,
            "w_identity": 0.5,
        },
        {
            "name": "A4_Patch_Fine",
            "nce_layer_weights": [0.5, 1.0, 0.0],
            "patches": [5, 7, 11],
            "w_delta_tv": 0.0,
            "w_identity": 0.5,
        },
        {
            "name": "A5_High_TV",
            "nce_layer_weights": [0.0, 0.4, 1.0],
            "patches": [7, 11, 15, 23],
            "w_delta_tv": 0.05,
            "w_identity": 0.5,
        },
        {
            "name": "A6_Strong_ID",
            "nce_layer_weights": [0.0, 0.4, 1.0],
            "patches": [7, 11, 15, 23],
            "w_delta_tv": 0.0,
            "w_identity": 1.0,
        },
    ]

    run_bat = out_dir / "ab.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write(f'set "AGG_ROOT=..\\{sweep_tag}-aggregate"\n')
        f_bat.write("if not exist \"%AGG_ROOT%\" mkdir \"%AGG_ROOT%\"\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write("echo Starting 6-way orthogonal ablation (80 Epochs, eval@40/80)\n")
        f_bat.write(f"echo Sweep tag={sweep_tag}, base batch_size={batch_size}\n")
        f_bat.write("echo ==========================================\n\n")

        for exp in experiments:
            cfg = copy.deepcopy(base)

            # Keep model-level defaults from base; only ablate target axes.
            cfg.setdefault("loss", {})
            cfg["loss"]["swd_patch_sizes"] = [int(p) for p in exp["patches"]]
            cfg["loss"]["nce_layer_weights"] = [float(v) for v in exp["nce_layer_weights"]]
            cfg["loss"]["w_delta_tv"] = float(exp["w_delta_tv"])
            cfg["loss"]["w_identity"] = float(exp["w_identity"])

            cfg.setdefault("training", {})
            cfg["training"]["num_epochs"] = 80
            cfg["training"]["full_eval_interval"] = 40
            cfg["training"]["full_eval_on_last_epoch"] = True
            cfg["training"]["save_interval"] = 20

            cfg.setdefault("checkpoint", {})
            exp_dir = f"{sweep_tag}_{exp['name']}"
            cfg["checkpoint"]["save_dir"] = f"../{exp_dir}"

            cfg_filename = f"config_{sweep_tag}_{exp['name']}.json"
            cfg_path = out_dir / cfg_filename
            with open(cfg_path, "w", encoding="utf-8") as f_cfg:
                json.dump(cfg, f_cfg, indent=4, ensure_ascii=False)

            print(
                f"Generated: {cfg_filename:45s} | "
                f"patch={exp['patches']} nce={exp['nce_layer_weights']} "
                f"w_delta_tv={exp['w_delta_tv']:.2f} w_identity={exp['w_identity']:.2f}"
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

    print("\nab.bat has been generated.")
    print(f"Sweep tag: {sweep_tag}, batch_size={batch_size}")


if __name__ == "__main__":
    create_sweep()
