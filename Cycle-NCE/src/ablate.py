from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SERIES_NAME = "repuls"

FORCED_TRAINING_OVERRIDES = {
    "batch_size": 256,
    "num_epochs": 80,
    "save_interval": 40,
    "full_eval_interval": 40,
    "full_eval_on_last_epoch": True,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.125,
    "warmup_start_factor": 0.0,
}

FORCED_MODEL_OVERRIDES = {
    "ablation_no_residual": False,
    "ablation_direct_delta_blend": True,
    "raw_latent_splat_highway": True,
}

FORCED_LOSS_OVERRIDES = {
    "swd_num_projections": 384,
    "swd_distance_mode": "sort",
}


def _load_base_config(src_dir: Path, base_config_arg: str | None) -> tuple[dict, Path]:
    if base_config_arg:
        base_path = Path(base_config_arg).expanduser()
        if not base_path.is_absolute():
            base_path = (src_dir / base_path).resolve()
    else:
        candidates = [
            (src_dir / "config_skip.json").resolve(),
            (src_dir / "config_repulse.json").resolve(),
            (src_dir / "config_in-idt.json").resolve(),
            (src_dir / "config.json").resolve(),
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
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


EXPERIMENTS: list[tuple[str, dict]] = [
    (
        "01_safe_baseline",
        {
            "model": {
                "skip_routing_mode": "naive",
                "skip_naive_gain": 0.6,
                "semantic_attn_temperature": 0.08,
            },
            "loss": {
                "w_swd_unified": 20.0,
                "w_identity": 10.0,
                "swd_patch_sizes": [3, 7, 15],
            },
            "training": {"learning_rate": 1e-05},
        },
    ),
    (
        "02_macro_only",
        {
            "model": {
                "skip_routing_mode": "naive",
                "skip_naive_gain": 0.6,
                "semantic_attn_temperature": 0.08,
            },
            "loss": {
                "w_swd_unified": 30.0,
                "w_identity": 10.0,
                "swd_patch_sizes": [7, 15, 25],
            },
            "training": {"learning_rate": 1e-05},
        },
    ),
    (
        "03_no_skip_extreme",
        {
            "model": {
                "skip_routing_mode": "none",
                "semantic_attn_temperature": 0.08,
            },
            "loss": {
                "w_swd_unified": 30.0,
                "w_identity": 15.0,
                "swd_patch_sizes": [3, 7, 15],
            },
            "training": {"learning_rate": 1e-05},
        },
    ),
    (
        "04_strong_identity",
        {
            "model": {
                "skip_routing_mode": "naive",
                "skip_naive_gain": 0.8,
                "semantic_attn_temperature": 0.08,
            },
            "loss": {
                "w_swd_unified": 20.0,
                "w_identity": 30.0,
                "swd_patch_sizes": [3, 7, 15],
            },
            "training": {"learning_rate": 1e-05},
        },
    ),
    (
        "05_high_style_push",
        {
            "model": {
                "skip_routing_mode": "naive",
                "skip_naive_gain": 0.4,
                "semantic_attn_temperature": 0.08,
            },
            "loss": {
                "w_swd_unified": 60.0,
                "w_identity": 10.0,
                "swd_patch_sizes": [3, 7, 15, 25],
            },
            "training": {"learning_rate": 1e-05},
        },
    ),
    (
        "06_soft_attention",
        {
            "model": {
                "skip_routing_mode": "naive",
                "skip_naive_gain": 0.6,
                "semantic_attn_temperature": 0.2,
            },
            "loss": {
                "w_swd_unified": 30.0,
                "w_identity": 10.0,
                "swd_patch_sizes": [3, 7, 15],
            },
            "training": {"learning_rate": 1e-05},
        },
    ),
    (
        "07_sharp_attention",
        {
            "model": {
                "skip_routing_mode": "naive",
                "skip_naive_gain": 0.6,
                "semantic_attn_temperature": 0.03,
            },
            "loss": {
                "w_swd_unified": 30.0,
                "w_identity": 10.0,
                "swd_patch_sizes": [3, 7, 15],
            },
            "training": {"learning_rate": 1e-05},
        },
    ),
    (
        "08_high_lr_stress_test",
        {
            "model": {
                "skip_routing_mode": "naive",
                "skip_naive_gain": 0.5,
                "semantic_attn_temperature": 0.08,
            },
            "loss": {
                "w_swd_unified": 20.0,
                "w_identity": 10.0,
                "swd_patch_sizes": [3, 7, 15],
            },
            "training": {
                "learning_rate": 5e-05,
                "warmup_ratio": 0.2,
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

        _deep_update(cfg["training"], FORCED_TRAINING_OVERRIDES)
        _deep_update(cfg["model"], FORCED_MODEL_OVERRIDES)
        _deep_update(cfg, patch)
        _deep_update(cfg["loss"], FORCED_LOSS_OVERRIDES)

        if "w_swd_unified" in cfg["loss"]:
            cfg["loss"].pop("w_swd", None)
            cfg["loss"].pop("w_swd_micro", None)
            cfg["loss"].pop("w_swd_macro", None)
        elif "w_swd_micro" in cfg["loss"] or "w_swd_macro" in cfg["loss"]:
            cfg["loss"].pop("w_swd", None)

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
    final_epoch = int(FORCED_TRAINING_OVERRIDES["num_epochs"])
    distill_epochs = 200
    distill_tag = f"distill_epochs{distill_epochs}"
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
        f.write("echo Move finished. Exporting pre-distill CSV summary...\n")
        f.write("cd /d \"%ROOT_DIR%\"\n")
        f.write(f"python import_summary_history_to_csv.py -i {SERIES_NAME} -o {SERIES_NAME}_pre_distill.csv\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        f.write("echo Running batch distill + post-distill eval...\n")
        f.write(
            f"uv run src/batch_distill_full_eval.py --exp_dir {SERIES_NAME} --recursive --distill_mode tokenizer --distill_epochs {distill_epochs} --batch_size 32\n"
        )
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        f.write("echo Distill/eval finished. Exporting post-distill CSV summary...\n")
        f.write(f"python import_summary_history_to_csv.py -i {SERIES_NAME} -o {SERIES_NAME}_post_distill.csv\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f.write(f"python import_summary_history_to_csv.py -i {SERIES_NAME} -o {SERIES_NAME}.csv\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        f.write("echo Running MA probe per experiment (base + tokenized)...\n")
        f.write(f"set \"FINAL_EPOCH={final_epoch:04d}\"\n")
        f.write(f"set \"DISTILL_TAG={distill_tag}\"\n")
        f.write(f"for /d %%D in (\"%TARGET_DIR%\\{SERIES_NAME}_*\") do (\n")
        f.write("  if exist \"%%~fD\\epoch_%FINAL_EPOCH%.pt\" (\n")
        f.write("    echo Probing BASE %%~nxD with %%~fD\\epoch_%FINAL_EPOCH%.pt...\n")
        f.write("    uv run src/probe_ma.py --checkpoint \"%%~fD\\epoch_%FINAL_EPOCH%.pt\" --num-samples 8 --json-out \"%%~fD\\ma_probe_base_epoch_%FINAL_EPOCH%.json\"\n")
        f.write("    if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f.write("  ) else (\n")
        f.write("    echo WARNING: base checkpoint not found for %%~nxD at %%~fD\\epoch_%FINAL_EPOCH%.pt\n")
        f.write("  )\n")
        f.write("  if exist \"%%~fD\\tokenizer_distill\\epoch_%FINAL_EPOCH%_%DISTILL_TAG%\\epoch_%FINAL_EPOCH%_tokenized.pt\" (\n")
        f.write("    echo Probing TOKENIZED %%~nxD with tokenizer_distill\\epoch_%FINAL_EPOCH%_%DISTILL_TAG%\\epoch_%FINAL_EPOCH%_tokenized.pt...\n")
        f.write("    uv run src/probe_ma.py --checkpoint \"%%~fD\\tokenizer_distill\\epoch_%FINAL_EPOCH%_%DISTILL_TAG%\\epoch_%FINAL_EPOCH%_tokenized.pt\" --num-samples 8 --json-out \"%%~fD\\ma_probe_tokenized_epoch_%FINAL_EPOCH%.json\"\n")
        f.write("    if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f.write("  ) else (\n")
        f.write("    echo WARNING: tokenized checkpoint not found for %%~nxD at %%~fD\\tokenizer_distill\\epoch_%FINAL_EPOCH%_%DISTILL_TAG%\\epoch_%FINAL_EPOCH%_tokenized.pt\n")
        f.write("  )\n")
        f.write("  if exist \"%%~fD\\ma_probe*.json\" (\n")
        f.write("    uv run src/probe_ma_sweep.py --input-glob \"%%~fD\\ma_probe*.json\" --output-dir \"%%~fD\" --output-prefix ma_probe_view\n")
        f.write("    if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f.write("  )\n")
        f.write(")\n\n")
        f.write("echo Building cross-experiment MA summary...\n")
        f.write(f"uv run src/probe_ma_sweep.py --input-glob \"%TARGET_DIR%\\{SERIES_NAME}_*\\ma_probe*.json\" --output-dir \"%TARGET_DIR%\" --output-prefix ma_probe_all_pairs\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        f.write(f"for /d %%D in (\"%TARGET_DIR%\\{SERIES_NAME}_*\") do (\n")
        f.write("  if exist \"%TARGET_DIR%\\ma_probe_all_pairs.html\" copy /Y \"%TARGET_DIR%\\ma_probe_all_pairs.html\" \"%%~fD\\ma_probe_all_pairs.html\" >nul\n")
        f.write("  if exist \"%TARGET_DIR%\\ma_probe_all_pairs.csv\" copy /Y \"%TARGET_DIR%\\ma_probe_all_pairs.csv\" \"%%~fD\\ma_probe_all_pairs.csv\" >nul\n")
        f.write("  if exist \"%TARGET_DIR%\\ma_probe_all_pairs.json\" copy /Y \"%TARGET_DIR%\\ma_probe_all_pairs.json\" \"%%~fD\\ma_probe_all_pairs.json\" >nul\n")
        f.write(")\n\n")
        f.write("echo All done.\n")
    print(f"generated: {script_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repuls experiment suite.")
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="Optional base config path. Default uses src/config_skip.json.",
    )
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    run_ids = generate(src_dir, base_config_arg=args.base_config)
    _write_run_script(src_dir, run_ids)


if __name__ == "__main__":
    main()
