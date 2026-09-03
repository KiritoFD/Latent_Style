from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SERIES_NAME = "Color120"

FORCED_TRAINING_OVERRIDES = {
    "num_epochs": 120,
    "save_interval": 40,
    "full_eval_interval": 40,
    "full_eval_on_last_epoch": True,
}


def _load_base_config(src_dir: Path, base_config_arg: str | None) -> tuple[dict, Path]:
    if base_config_arg:
        base_path = Path(base_config_arg).expanduser()
        if not base_path.is_absolute():
            base_path = (src_dir / base_path).resolve()
    else:
        candidates = [
            (src_dir / "config.json").resolve(),
            (src_dir.parent / "config.json").resolve(),
        ]
        base_path = next((p for p in candidates if p.exists()), candidates[-1])
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    with open(base_path, "r", encoding="utf-8-sig") as f:
        return json.load(f), base_path


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _deep_update(dst: dict, src: dict) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


EXPERIMENTS: list[tuple[str, dict]] = [
    (
        "C01_HF_Tyrant",
        {
            "model": {
                "ablation_no_residual": True,
                "num_decoder_blocks": 2,
                "residual_gain": 2.0,
                "output_moment_match": True,
                "output_moment_match_train_only": True,
            },
            "loss": {
                "w_identity": 0.0,
                "swd_patch_sizes": [3, 5],
                "w_swd": 250.0,
                "swd_use_high_freq": True,
                "swd_hf_weight_ratio": 50.0,
                "w_color": 100.0,
            },
        },
    ),
    (
        "C02_HF_Leakage",
        {
            "model": {
                "ablation_no_residual": False,
                "skip_routing_mode": "adaptive",
                "skip_naive_gain": 0.15,
                "num_decoder_blocks": 2,
                "residual_gain": 2.0,
                "output_moment_match": True,
                "output_moment_match_train_only": True,
            },
            "loss": {
                "w_identity": 0.0,
                "swd_patch_sizes": [3, 5, 7],
                "w_swd": 250.0,
                "swd_use_high_freq": True,
                "swd_hf_weight_ratio": 25.0,
                "w_color": 80.0,
            },
        },
    ),
]


def generate(src_dir: Path, base_config_arg: str | None = None) -> list[str]:
    base, base_path = _load_base_config(src_dir, base_config_arg)
    print(f"Base config: {base_path}")
    output_dir = src_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ids: list[str] = []
    for name, patch in EXPERIMENTS:
        cfg = copy.deepcopy(base)
        cfg.setdefault("model", {})
        cfg.setdefault("loss", {})
        cfg.setdefault("checkpoint", {})
        cfg.setdefault("training", {})
        _deep_update(cfg, patch)
        _deep_update(cfg["training"], FORCED_TRAINING_OVERRIDES)
        cfg["checkpoint"]["save_dir"] = f"../{SERIES_NAME}_{name}"
        cfg_name = f"config_{name}.json"
        _write_json(output_dir / cfg_name, cfg)
        run_ids.append(name)
        print(f"generated: {cfg_name}")

    if len(run_ids) != len(EXPERIMENTS):
        raise RuntimeError(f"Expected {len(EXPERIMENTS)} runs, got {len(run_ids)}")
    return run_ids


def _write_run_script(src_dir: Path, run_ids: list[str]) -> None:
    script_path = src_dir / f"{SERIES_NAME}.bat"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write("setlocal\n")
        f.write("cd /d %~dp0\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        total = len(run_ids)
        for idx, run_id in enumerate(run_ids, start=1):
            cfg_name = f"config_{run_id}.json"
            f.write(f"echo [{idx}/{total}] Running {run_id}...\n")
            f.write(f"uv run run.py --config {cfg_name}\n")
            f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        f.write("echo All training runs finished.\n")
        f.write("set \"SRC_DIR=%cd%\"\n")
        f.write("set \"ROOT_DIR=%SRC_DIR%\\..\"\n")
        f.write(f"set \"TARGET_DIR=%ROOT_DIR%\\{SERIES_NAME}\"\n")
        f.write("if not exist \"%TARGET_DIR%\" mkdir \"%TARGET_DIR%\"\n\n")
        f.write(f"for /d %%D in (\"%ROOT_DIR%\\{SERIES_NAME}_*\") do (\n")
        f.write("  echo Moving %%~nxD to %TARGET_DIR%...\n")
        f.write("  robocopy \"%%~fD\" \"%TARGET_DIR%\\%%~nxD\" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP\n")
        f.write("  if errorlevel 8 exit /b 8\n")
        f.write("  if exist \"%%~fD\" rmdir \"%%~fD\" /S /Q\n")
        f.write(")\n\n")
        f.write("echo Move finished. Running batch distill eval...\n")
        f.write("cd /d \"%ROOT_DIR%\"\n")
        f.write(f"uv run src/batch_distill_full_eval.py --exp_dir {SERIES_NAME}\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        f.write("echo Distill/eval finished. Exporting CSV summary...\n")
        f.write(f"python import_summary_history_to_csv.py -i {SERIES_NAME} -o {SERIES_NAME}.csv\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        f.write("echo All done.\n")
    print(f"generated: {script_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Color120 experiment suite.")
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="Optional base config path. Default uses src/config.json.",
    )
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    run_ids = generate(src_dir, base_config_arg=args.base_config)
    _write_run_script(src_dir, run_ids)


if __name__ == "__main__":
    main()
