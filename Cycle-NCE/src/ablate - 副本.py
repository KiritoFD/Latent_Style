from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ABLATE_SERIES = "ablate_patch_hf_12"

# 4 patch presets x 3 HF modes = 12 runs
# Note: p_base keeps the current default patch regime to complete a 12-run matrix.
PATCH_PRESETS: list[tuple[str, list[int]]] = [
    ("p_base", [7, 11, 15, 19, 25]),
    ("p_1_5_9_15", [1, 5, 9, 15]),
    ("p_5_9_15_25", [5, 9, 15, 25]),
    ("p_5_9_15", [5, 9, 15]),
]

# hf_off, hf_on@1.0, hf_on@3.0
HF_PRESETS: list[tuple[str, bool, float]] = [
    ("hf_off", False, 1.0),
    ("hf_1p0", True, 1.0),
    ("hf_3p0", True, 3.0),
]


def _load_base_config(src_dir: Path) -> dict:
    base_path = src_dir / "config.json"
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_run_script(src_dir: Path, script_name: str, run_ids: list[str]) -> None:
    script_path = src_dir / script_name
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write("setlocal\n")
        f.write("cd /d %~dp0\n")
        f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")
        for run_id in run_ids:
            f.write("echo ==================================================\n")
            f.write(f"echo Starting {run_id}...\n")
            f.write("echo ==================================================\n")
            f.write(f"uv run run.py --config config_{run_id}.json\n")
            f.write("if %errorlevel% neq 0 exit /b %errorlevel%\n\n")


def _apply_common_defaults(cfg: dict) -> None:
    cfg.setdefault("loss", {})
    cfg.setdefault("checkpoint", {})
    cfg.setdefault("training", {})

    # Keep stable infra/perf defaults for ablation comparability.
    cfg["training"]["channels_last"] = True
    cfg["training"]["use_gradient_checkpointing"] = True
    cfg["training"]["use_compile"] = False

    # Ensure core weights are present.
    cfg["loss"]["w_swd"] = float(cfg["loss"].get("w_swd", 150.0))
    cfg["loss"]["w_color"] = float(cfg["loss"].get("w_color", 50.0))
    cfg["loss"]["w_identity"] = float(cfg["loss"].get("w_identity", 30.0))


def _build_cfg(base: dict, run_id: str, patch_sizes: list[int], hf_on: bool, hf_ratio: float) -> dict:
    cfg = copy.deepcopy(base)
    _apply_common_defaults(cfg)

    cfg["loss"]["swd_patch_sizes"] = [int(v) for v in patch_sizes]
    cfg["loss"]["swd_use_high_freq"] = bool(hf_on)
    cfg["loss"]["swd_hf_weight_ratio"] = float(hf_ratio)
    cfg["checkpoint"]["save_dir"] = f"../{run_id}"
    return cfg


def generate(src_dir: Path) -> list[str]:
    base = _load_base_config(src_dir)
    run_ids: list[str] = []
    for patch_tag, patch_sizes in PATCH_PRESETS:
        for hf_tag, hf_on, hf_ratio in HF_PRESETS:
            run_id = f"{patch_tag}_{hf_tag}"
            cfg = _build_cfg(base, run_id, patch_sizes, hf_on, hf_ratio)
            _write_json(src_dir / f"config_{run_id}.json", cfg)
            run_ids.append(run_id)
            print(f"generated: config_{run_id}.json")
    return run_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate patch x HF ablation configs (12 runs).")
    parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    run_ids = generate(src_dir)
    if len(run_ids) != 12:
        raise RuntimeError(f"Expected 12 runs, got {len(run_ids)}")
    _write_run_script(src_dir, f"{ABLATE_SERIES}.bat", run_ids)
    print(f"generated: {ABLATE_SERIES}.bat")


if __name__ == "__main__":
    main()
