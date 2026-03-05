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

    # Decoder-focused orthogonal sweep:
    # format: (name, swd_use_high_freq, swd_hf_weight_ratio, w_identity, w_delta_tv)
    experiments = [
        ("decoder-A-anchor-nohf", False, 2.0, 1.2, 0.005),
        ("decoder-B-hf-strict-id", True, 2.0, 1.2, 0.005),
        ("decoder-C-relaxed-id-nohf", False, 2.0, 0.25, 0.005),
        ("decoder-D-sweetspot", True, 2.0, 0.3, 0.005),
        ("decoder-E-extreme-brush", True, 5.0, 0.05, 0.005),
        ("decoder-F-tv-off", True, 2.0, 0.3, 0.0),
    ]

    run_bat = out_dir / "run_decoder_ablate_6.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write('set "AGG_ROOT=..\\decoder-ablation-aggregate"\n')
        f_bat.write("if not exist \"%AGG_ROOT%\" mkdir \"%AGG_ROOT%\"\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write("echo Starting 6 Decoder Ablations\n")
        f_bat.write("echo ==========================================\n\n")

        for name, use_hf, hf_ratio, w_identity, w_delta_tv in experiments:
            cfg = copy.deepcopy(base)

            cfg.setdefault("loss", {})
            cfg["loss"]["swd_patch_sizes"] = [5, 7, 11, 15, 23]
            cfg["loss"]["swd_use_high_freq"] = bool(use_hf)
            cfg["loss"]["swd_hf_weight_ratio"] = float(hf_ratio)
            cfg["loss"]["w_identity"] = float(w_identity)
            cfg["loss"]["w_delta_tv"] = float(w_delta_tv)

            cfg.setdefault("training", {})
            cfg["training"]["num_epochs"] = 80
            cfg["training"]["full_eval_interval"] = 40
            cfg["training"]["full_eval_on_last_epoch"] = True
            cfg["training"]["save_interval"] = 20

            cfg.setdefault("checkpoint", {})
            cfg["checkpoint"]["save_dir"] = f"../{name}"

            cfg_filename = f"config_{name}.json"
            cfg_path = out_dir / cfg_filename
            with open(cfg_path, "w", encoding="utf-8") as f_cfg:
                json.dump(cfg, f_cfg, indent=4, ensure_ascii=False)

            print(
                f"generated: {cfg_filename:42s} | "
                f"hf={int(use_hf)} hf_ratio={hf_ratio:.2f} id={w_identity:.2f} tv={w_delta_tv:.3f}"
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

        f_bat.write("\n")
        f_bat.write("echo.\n")
        f_bat.write("echo Aggregating summary_history metrics ...\n")
        f_bat.write(
            "uv run python ..\\scripts\\collect_ablation_results.py "
            "--root \"%AGG_ROOT%\" "
            "--output-dir \"%AGG_ROOT%\" "
            "--epoch-dir epoch_0080\n"
        )
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")

    print("\nrun_decoder_ablate_6.bat has been generated.")


if __name__ == "__main__":
    create_sweep()
