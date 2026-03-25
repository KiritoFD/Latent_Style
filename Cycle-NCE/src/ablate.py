import copy
import json
from pathlib import Path


def load_base_config() -> dict:
    base_path = Path(__file__).resolve().parent / "config_style_oa_5_lr5e4_wc2_swd60_id30_e120.json"
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_old_outputs(out_dir: Path) -> None:
    old_patterns = [
        "config_color_0*_*.json",
        "config_color_ablation_exp*.json",
        "config_weight_exp*.json",
        "config_tmp_batch*.json",
        "run_color_r16_e60.bat",
        "run_color_ablation_anchor_4.bat",
        "anchor4.bat",
        "weight.bat",
        "style_oa.bat",
        "patch_size_ablation.bat",
        "config_patch_size_ablation_*.json",
    ]
    removed = 0
    for pattern in old_patterns:
        for p in out_dir.glob(pattern):
            if p.is_file():
                p.unlink(missing_ok=True)
                removed += 1
    if removed > 0:
        print(f"cleaned old files: {removed}")


def generate_patch_size_ablation() -> None:
    base = load_base_config()
    out_dir = Path(__file__).resolve().parent
    _cleanup_old_outputs(out_dir)

    experiments = [
        ("patch_size_ablation_1_ps3-5-7-11", [3, 5, 7, 11]),
        ("patch_size_ablation_2_ps5-9-15", [5, 9, 15]),
        ("patch_size_ablation_3_ps7-11-19", [7, 11, 19]),
        ("patch_size_ablation_4_ps11-15-23", [11, 15, 23]),
    ]

    run_bat = out_dir / "patch_size_ablation.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write("echo Running SWD Patch Size Ablation (4 exps)\n")
        f_bat.write("echo ==========================================\n")

        for i, (name, patch_sizes) in enumerate(experiments, start=1):
            cfg = copy.deepcopy(base)
            cfg.setdefault("loss", {})
            cfg["loss"]["swd_patch_sizes"] = list(patch_sizes)
            cfg.setdefault("checkpoint", {})
            cfg["checkpoint"]["save_dir"] = f"../{name}"

            cfg_filename = f"config_{name}.json"
            cfg_path = out_dir / cfg_filename
            with open(cfg_path, "w", encoding="utf-8") as f_cfg:
                json.dump(cfg, f_cfg, indent=4, ensure_ascii=False)
                f_cfg.write("\n")

            print(
                f"generated: {cfg_filename:74s} | "
                f"exp={i} swd_patch_sizes={cfg['loss']['swd_patch_sizes']}"
            )

            f_bat.write("echo.\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"echo Running Experiment {i}: {name}\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"uv run run.py --config {cfg_filename}\n")
            f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")

        f_bat.write("echo.\n")
        f_bat.write("echo SWD patch size ablation finished.\n")

    print("\npatch_size_ablation.bat has been generated.")


if __name__ == "__main__":
    generate_patch_size_ablation()
