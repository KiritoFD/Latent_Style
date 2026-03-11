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

    # Orthogonal ablation sweep around decoder-D-sweetspot.
    # exp0 = baseline; exp1~exp5 each modifies one axis only.
    experiments = [
        ("exp0-baseline", {}),
        ("exp1-hf-ratio-4p0", {"loss": {"swd_hf_weight_ratio": 4.0}}),
        ("exp2-large-patches", {"loss": {"swd_patch_sizes": [11, 15, 23, 31]}}),
        ("exp3-hard-cdf", {"loss": {"swd_cdf_num_bins": 128, "swd_cdf_tau": 0.005}}),
        ("exp4-skip-retain-plus20", {"model": {"style_skip_content_retention_boost": 0.2}}),
        ("exp5-proj-1280", {"loss": {"swd_num_projections": 1280}}),
    ]

    run_bat = out_dir / "run_decoder_orthogonal_ablate_6.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write('set "AGG_ROOT=..\\decoder-orthogonal-ablation-aggregate"\n')
        f_bat.write("if not exist \"%AGG_ROOT%\" mkdir \"%AGG_ROOT%\"\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write("echo Starting 6 Orthogonal Decoder Ablations\n")
        f_bat.write("echo ==========================================\n\n")

        for name, delta in experiments:
            cfg = copy.deepcopy(base)

            # Fixed infra + schedule requested by the experiment protocol.
            cfg.setdefault("training", {})
            cfg["training"]["batch_size"] = 320
            cfg["training"]["use_gradient_checkpointing"] = True
            cfg["training"]["num_epochs"] = 120
            cfg["training"]["full_eval_interval"] = 40
            cfg["training"]["full_eval_on_last_epoch"] = True
            cfg["training"]["save_interval"] = 20

            # New baseline loss settings.
            cfg.setdefault("loss", {})
            cfg["loss"]["w_swd"] = 40.0
            cfg["loss"]["w_identity"] = 0.45

            # Apply one orthogonal delta for this experiment.
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
                f"eval={cfg['training']['full_eval_interval']} "
                f"w_swd={cfg['loss']['w_swd']:.1f} "
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

        f_bat.write("\n")
        f_bat.write("echo.\n")
        f_bat.write("echo Aggregating summary_history metrics ...\n")
        f_bat.write(
            "uv run python ..\\scripts\\collect_ablation_results.py "
            "--root \"%AGG_ROOT%\" "
            "--output-dir \"%AGG_ROOT%\" "
            "--epoch-dir epoch_0120\n"
        )
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")

    print("\nrun_decoder_orthogonal_ablate_6.bat has been generated.")


if __name__ == "__main__":
    create_sweep()
