from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "config.json"
OUT_ROOT = ROOT / "experiments"
CONFIG_DIR = OUT_ROOT
MANIFEST_PATH = OUT_ROOT / "arch_stress_test_60_manifest.json"
BAT_PATH = ROOT / "run_arch_stress_test_60.bat"
EVAL_BAT_PATH = ROOT / "eval_arch_stress_test_60.bat"


def load_base_config() -> dict:
    return json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))


def deep_update(dst: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _model_patch(skip: str, retention: float) -> dict:
    patch = {
        "skip_routing_mode": "none",
        "style_skip_content_retention_boost": retention,
    }
    if skip == "add_proj":
        patch["skip_fusion_mode"] = "add_proj"
        patch["skip_routing_mode"] = "normalized"
    return patch


def build_experiments() -> list[dict]:
    common_training = {
        "batch_size": 160,
        "num_epochs": 60,
        "save_interval": 20,
        "resume_checkpoint": "",
    }

    kin = 1.0
    swd = 40.0
    col = 15.0
    rep = 0.1

    specs = [
        ("arch60_00_baseline_naked", {"skip": "none", "ot_mse": 0.0, "retention": 0.0}),
        ("arch60_01_arch_add_proj", {"skip": "add_proj", "ot_mse": 0.0, "retention": 0.0}),
        ("arch60_02_math_ot_mse", {"skip": "none", "ot_mse": 1.0, "retention": 0.0}),
        ("arch60_03_feat_retention", {"skip": "none", "ot_mse": 0.0, "retention": 0.5}),
        ("arch60_04_ultimate_armor", {"skip": "add_proj", "ot_mse": 1.0, "retention": 0.0}),
    ]

    experiments: list[dict] = []
    for name, params in specs:
        experiments.append(
            {
                "name": name,
                "notes": f"60-epoch high-SWD structural stress test: {name}",
                "patch": {
                    "bridge": {
                        "objective_mode": "omf",
                        "loss_type": "mse" if params["ot_mse"] > 0.0 else "omf",
                        "w_kinetic": kin,
                        "terminal_swd_weight": swd,
                        "w_color": col,
                        "w_repulsive": rep,
                        "w_flow": params["ot_mse"],
                    },
                    "training": common_training,
                    "model": _model_patch(params["skip"], params["retention"]),
                },
            }
        )
    return experiments


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate_configs() -> list[dict]:
    base = load_base_config()
    experiments = build_experiments()
    manifests: list[dict] = []
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        cfg = copy.deepcopy(base)
        deep_update(cfg, exp["patch"])
        exp_name = exp["name"]
        cfg["checkpoint"]["save_dir"] = f"./experiments/{exp_name}"
        cfg["ablation"] = {
            "name": exp_name,
            "notes": exp["notes"],
        }
        cfg_path = CONFIG_DIR / f"{exp_name}.json"
        write_json(cfg_path, cfg)
        manifests.append(
            {
                "name": exp_name,
                "notes": exp["notes"],
                "config_path": f"experiments/{exp_name}.json",
                "checkpoint_path": f"experiments/{exp_name}/epoch_0060.pt",
            }
        )

    write_json(MANIFEST_PATH, {"experiments": manifests})
    return manifests


def generate_bat(manifests: list[dict]) -> None:
    lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\"",
        "",
        "if not exist \"experiments\" mkdir \"experiments\"",
        "set \"STATUS_LOG=experiments\\arch_stress_test_60_run_status.csv\"",
        "echo name,train_status,train_rc,checkpoint_exists>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "set /a TRAIN_FAIL_COUNT=0",
        "",
        "echo Running arch stress test 60 matrix...",
        "echo.",
        "",
    ]
    for item in manifests:
        name = item["name"]
        cfg = item["config_path"].replace("/", "\\")
        ckpt = item["checkpoint_path"].replace("/", "\\")
        lines.extend(
            [
                f"echo [{name}] train",
                f"python run.py --config \"{cfg}\"",
                "set \"TRAIN_RC=!ERRORLEVEL!\"",
                "if not \"!TRAIN_RC!\"==\"0\" (",
                "  set /a FAIL_COUNT+=1",
                "  set /a TRAIN_FAIL_COUNT+=1",
                "  set \"TRAIN_STATUS=FAIL\"",
                ") else (",
                "  set \"TRAIN_STATUS=OK\"",
                ")",
                "",
                f"if exist \"{ckpt}\" (set \"CKPT_STATUS=YES\") else (set \"CKPT_STATUS=NO\")",
                f"echo {name},!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>\"%STATUS_LOG%\"",
                "echo.",
            ]
        )
    lines.extend(
        [
            "echo.",
            "echo Arch stress test 60 finished.",
            "echo Status log: %STATUS_LOG%",
            "echo Total failures: !FAIL_COUNT! ^| train: !TRAIN_FAIL_COUNT!",
            "if not \"!FAIL_COUNT!\"==\"0\" exit /b 1",
            "exit /b 0",
        ]
    )
    BAT_PATH.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def generate_eval_bat(manifests: list[dict]) -> None:
    lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\"",
        "",
        "if not exist \"experiments\\full_eval\" mkdir \"experiments\\full_eval\"",
        "set \"STATUS_LOG=experiments\\arch_stress_test_60_eval_status.csv\"",
        "echo name,eval_status,eval_rc>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "",
        "echo Evaluating arch stress test 60 matrix...",
        "echo.",
        "",
    ]
    for item in manifests:
        name = item["name"]
        exp_dir = item["checkpoint_path"].replace("/epoch_0060.pt", "").replace("/", "\\")
        eval_out = f"experiments\\full_eval\\{name}"
        lines.extend(
            [
                f"echo [{name}] eval",
                f"python run_evaluation.py \"{exp_dir}\" --output \"{eval_out}\" --batch_size 2",
                "set \"EVAL_RC=!ERRORLEVEL!\"",
                "if not \"!EVAL_RC!\"==\"0\" (",
                "  set /a FAIL_COUNT+=1",
                "  set \"EVAL_STATUS=FAIL\"",
                ") else (",
                "  set \"EVAL_STATUS=OK\"",
                ")",
                f"echo {name},!EVAL_STATUS!,!EVAL_RC!>>\"%STATUS_LOG%\"",
                "echo.",
            ]
        )
    lines.extend(
        [
            "echo.",
            "echo Arch stress test 60 eval finished.",
            "echo Status log: %STATUS_LOG%",
            "echo Total failures: !FAIL_COUNT!",
            "if not \"!FAIL_COUNT!\"==\"0\" exit /b 1",
            "exit /b 0",
        ]
    )
    EVAL_BAT_PATH.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def main() -> None:
    manifests = generate_configs()
    generate_bat(manifests)
    generate_eval_bat(manifests)
    print(f"Generated {len(manifests)} configs under: {CONFIG_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Batch runner: {BAT_PATH}")
    print(f"Eval runner: {EVAL_BAT_PATH}")


if __name__ == "__main__":
    main()
