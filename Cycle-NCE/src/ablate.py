import copy
import json
from pathlib import Path


SERIES_NAME = "strong"
IDENTITY_WEIGHT = 30
SWD_WEIGHTS = [100, 150]
COLOR_WEIGHTS = [50, 80, 20]


def load_base_config() -> dict:
    base_path = Path(__file__).resolve().parent / "config.json"
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_old_outputs(out_dir: Path) -> None:
    old_patterns = [
        "config_strong_*.json",
        "strong.bat",
    ]
    removed = 0
    for pattern in old_patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
    if removed > 0:
        print(f"cleaned old files: {removed}")


def _build_experiment_name(series_name: str, identity_weight: int, swd_weight: int, color_weight: int) -> str:
    return f"{series_name}_idt{identity_weight}_swd{swd_weight}_color{color_weight}"


def generate_strong_ablation() -> None:
    base = load_base_config()
    out_dir = Path(__file__).resolve().parent
    _cleanup_old_outputs(out_dir)

    experiments = []
    for swd_weight in SWD_WEIGHTS:
        for color_weight in COLOR_WEIGHTS:
            name = _build_experiment_name(
                SERIES_NAME,
                IDENTITY_WEIGHT,
                swd_weight,
                color_weight,
            )
            experiments.append((name, swd_weight, color_weight))

    run_bat = out_dir / f"{SERIES_NAME}.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write(f"echo Running {SERIES_NAME} ablation ({len(experiments)} exps)\n")
        f_bat.write("echo ==========================================\n")

        for index, (name, swd_weight, color_weight) in enumerate(experiments, start=1):
            cfg = copy.deepcopy(base)
            cfg.setdefault("loss", {})
            cfg["loss"]["w_identity"] = float(IDENTITY_WEIGHT)
            cfg["loss"]["w_swd"] = float(swd_weight)
            cfg["loss"]["w_color"] = float(color_weight)
            cfg.setdefault("checkpoint", {})
            cfg["checkpoint"]["save_dir"] = f"../{name}"

            cfg_filename = f"config_{name}.json"
            cfg_path = out_dir / cfg_filename
            with open(cfg_path, "w", encoding="utf-8") as f_cfg:
                json.dump(cfg, f_cfg, indent=4, ensure_ascii=False)
                f_cfg.write("\n")

            print(
                f"generated: {cfg_filename:52s} | "
                f"exp={index} idt={IDENTITY_WEIGHT} swd={swd_weight} color={color_weight}"
            )

            f_bat.write("echo.\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"echo Running Experiment {index}: {name}\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"uv run run.py --config {cfg_filename}\n")
            f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")

        f_bat.write("echo.\n")
        f_bat.write(f"echo {SERIES_NAME} ablation finished.\n")

    print(f"\n{run_bat.name} has been generated.")


if __name__ == "__main__":
    generate_strong_ablation()
