from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SERIES_NAME = "Gate"
DEFAULT_BASE_CONFIG = "Layer-Norm.json"

FORCED_TRAINING_OVERRIDES = {
    "batch_size": 256,
    "num_epochs": 120,
    "save_interval": 30,
    "full_eval_interval": 30,
    "full_eval_on_last_epoch": True,
    "warmup_ratio": 0.1,
    "learning_rate": 8e-5,
}

FORCED_LOSS_OVERRIDES = {}


def _load_base_config(src_dir: Path, base_config_arg: str | None) -> tuple[dict, Path]:
    base_name = base_config_arg or DEFAULT_BASE_CONFIG
    base_path = Path(base_name).expanduser()
    if not base_path.is_absolute():
        base_path = (src_dir / base_path).resolve()
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
    ("01_baseline", {}),
    ("02_weak_decoder", {"model": {"num_decoder_blocks": 1}}),
    (
        "03_restore_skip_shortcut",
        {"model": {"skip_routing_mode": "adaptive", "skip_fusion_mode": "add_proj"}},
    ),
    ("04_attn_gate_fixed", {"model": {"attn_gate_mode": "fixed"}}),
    ("05_attn_gate_learned", {"model": {"attn_gate_mode": "learned"}}),
    (
        "06_gate_learned_idt_energy",
        {"model": {"attn_gate_mode": "learned"}, "loss": {"idt_mode": "energy", "w_identity": 200.0}},
    ),
    ("07_aux_loss_weak", {"loss": {"w_aux_delta_variance": 0.1}}),
    ("08_aux_loss_strong", {"loss": {"w_aux_delta_variance": 1.0}}),
    (
        "09_gate_and_bipolar",
        {"model": {"attn_gate_mode": "learned"}, "loss": {"swd_patch_sizes": [3, 25]}},
    ),
    (
        "10_gate_and_low_color",
        {"model": {"attn_gate_mode": "learned"}, "loss": {"w_color": 10.0, "w_swd_unified": 60.0}},
    ),
    (
        "11_gate_bipolar_low_color",
        {
            "model": {"attn_gate_mode": "learned"},
            "loss": {"w_color": 10.0, "w_swd_unified": 60.0, "swd_patch_sizes": [3, 25]},
        },
    ),
    (
        "12_gate_energy_bipolar_low_color",
        {
            "model": {"attn_gate_mode": "learned"},
            "loss": {
                "idt_mode": "energy",
                "w_identity": 200.0,
                "w_color": 10.0,
                "w_swd_unified": 60.0,
                "swd_patch_sizes": [3, 25],
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
        _deep_update(cfg["loss"], FORCED_LOSS_OVERRIDES)
        _deep_update(cfg, patch)

        if "w_swd_unified" in cfg["loss"]:
            cfg["loss"].pop("w_swd", None)
            cfg["loss"].pop("w_swd_micro", None)
            cfg["loss"].pop("w_swd_macro", None)

        cfg["checkpoint"]["save_dir"] = f"../{SERIES_NAME}_{name}"
        cfg_name = f"config_{name}.json"
        _write_json(output_dir / cfg_name, cfg)
        run_ids.append(name)
        print(f"generated: {cfg_name}")

    if len(run_ids) != len(EXPERIMENTS):
        raise RuntimeError(f"Expected {len(EXPERIMENTS)} runs, got {len(run_ids)}")
    return run_ids


def _write_run_script(src_dir: Path, run_ids: list[str], base_config_arg: str | None = None) -> None:
    base_cfg, _ = _load_base_config(src_dir, base_config_arg)
    final_epoch = int(FORCED_TRAINING_OVERRIDES.get("num_epochs", base_cfg.get("training", {}).get("num_epochs", 100)))

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
            f.write(f"uv run .\\run.py --config .\\{cfg_name}\n")
            f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")

        f.write("echo All training runs finished.\n")
        f.write("set \"SRC_DIR=%cd%\"\n")
        f.write(f"set \"TARGET_DIR=%SRC_DIR%\\..\\{SERIES_NAME}\"\n")
        f.write("if not exist \"%TARGET_DIR%\" mkdir \"%TARGET_DIR%\"\n\n")

        f.write(f"for /d %%D in (\"%SRC_DIR%\\..\\{SERIES_NAME}_*\") do (\n")
        f.write("  echo Moving %%~nxD to %TARGET_DIR%...\n")
        f.write("  robocopy \"%%~fD\" \"%TARGET_DIR%\\%%~nxD\" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP\n")
        f.write("  if errorlevel 8 exit /b 8\n")
        f.write("  if exist \"%%~fD\" rmdir \"%%~fD\" /S /Q\n")
        f.write(")\n\n")

        f.write("echo Running batch distill + full eval...\n")
        f.write(f"uv run .\\batch_distill_full_eval.py --exp_dir ..\\{SERIES_NAME}\\\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")

        f.write("echo Exporting summary CSV...\n")
        f.write(f"python .\\import_summary_history_to_csv.py -i ..\\{SERIES_NAME} -o .\\{SERIES_NAME}.csv\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")

        f.write("echo Running MA probe sweep...\n")
        f.write(f"set \"FINAL_EPOCH={final_epoch:04d}\"\n")
        f.write(f"for /d %%D in (\"%TARGET_DIR%\\{SERIES_NAME}_*\") do (\n")
        f.write("  if exist \"%%~fD\\epoch_%FINAL_EPOCH%.pt\" (\n")
        f.write("    uv run .\\probe_ma.py --checkpoint \"%%~fD\\epoch_%FINAL_EPOCH%.pt\" --num-samples 8 --json-out \"%%~fD\\ma_probe_base_epoch_%FINAL_EPOCH%.json\"\n")
        f.write("    if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f.write("  )\n")
        f.write(")\n\n")

        f.write("echo All done.\n")

    print(f"generated: {script_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate balanced ablation suite from Layer-Norm baseline.")
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help=f"Optional base config path. Default uses src/{DEFAULT_BASE_CONFIG}.",
    )
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    run_ids = generate(src_dir, base_config_arg=args.base_config)
    _write_run_script(src_dir, run_ids, base_config_arg=args.base_config)


if __name__ == "__main__":
    main()
