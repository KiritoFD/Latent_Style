from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "config.json"
OUT_ROOT = ROOT / "experiments"
CONFIG_DIR = OUT_ROOT
MANIFEST_PATH = OUT_ROOT / "latent_anchor_stress_60_manifest.json"
TRAIN_BAT_PATH = ROOT / "run_latent_anchor_stress_60.bat"
EVAL_BAT_PATH = ROOT / "eval_latent_anchor_stress_60.bat"


def load_base_config() -> dict:
    return json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))


def deep_update(dst: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def build_experiments() -> list[dict]:
    common_training = {
        "batch_size": 160,
        "num_epochs": 60,
        "save_interval": 20,
        "resume_checkpoint": "",
    }
    base_bridge = {
        "objective_mode": "omf",
        "loss_type": "omf",
        "w_kinetic": 1.0,
        "terminal_swd_weight": 40.0,
        "w_color": 15.0,
        "w_repulsive": 0.1,
        "w_cycle": 2.0,
        "w_flow": 0.0,
    }
    specs = [
        ("latent60_01_naked", {"w_nce": 0.0, "w_low_freq": 0.0}),
        ("latent60_02_patch_nce", {"w_nce": 0.5, "w_low_freq": 0.0}),
        ("latent60_03_low_freq", {"w_nce": 0.0, "w_low_freq": 15.0}),
        ("latent60_04_nce_low_freq", {"w_nce": 0.5, "w_low_freq": 15.0}),
    ]
    experiments: list[dict] = []
    for name, params in specs:
        bridge_patch = {
            **base_bridge,
            "w_nce": params["w_nce"],
            "w_low_freq": params["w_low_freq"],
        }
        experiments.append(
            {
                "name": name,
                "notes": f"60-epoch latent anchor stress test: {name}",
                "patch": {
                    "bridge": bridge_patch,
                    "training": common_training,
                    "model": {"skip_routing_mode": "none", "style_skip_content_retention_boost": 0.0},
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
    for exp in experiments:
        cfg = copy.deepcopy(base)
        deep_update(cfg, exp["patch"])
        exp_name = exp["name"]
        cfg["checkpoint"]["save_dir"] = f"./experiments/{exp_name}"
        cfg["ablation"] = {"name": exp_name, "notes": exp["notes"]}
        cfg_path = CONFIG_DIR / f"{exp_name}.json"
        write_json(cfg_path, cfg)
        manifests.append(
            {
                "name": exp_name,
                "config_path": f"experiments/{exp_name}.json",
                "checkpoint_dir": f"experiments/{exp_name}",
                "checkpoint_path": f"experiments/{exp_name}/epoch_0060.pt",
                "eval_output": f"experiments/full_eval/{exp_name}",
            }
        )
    write_json(MANIFEST_PATH, {"experiments": manifests})
    return manifests


def generate_train_bat(manifests: list[dict]) -> None:
    lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\"",
        "",
        "if not exist \"experiments\" mkdir \"experiments\"",
        "set \"STATUS_LOG=experiments\\latent_anchor_stress_60_run_status.csv\"",
        "echo name,train_status,train_rc,checkpoint_exists>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "set /a TRAIN_FAIL_COUNT=0",
        "",
        "echo Running latent-anchor stress test 60 matrix...",
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
            "echo Latent-anchor stress test 60 finished.",
            "echo Status log: %STATUS_LOG%",
            "echo Total failures: !FAIL_COUNT! ^| train: !TRAIN_FAIL_COUNT!",
            "if not \"!FAIL_COUNT!\"==\"0\" exit /b 1",
            "exit /b 0",
        ]
    )
    TRAIN_BAT_PATH.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def generate_eval_bat(manifests: list[dict]) -> None:
    lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\"",
        "",
        "if not exist \"experiments\\full_eval\" mkdir \"experiments\\full_eval\"",
        "set \"STATUS_LOG=experiments\\latent_anchor_stress_60_eval_status.csv\"",
        "echo name,eval_status,eval_rc>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "",
        "echo Evaluating latent-anchor stress test 60 matrix...",
        "echo.",
        "",
    ]
    for item in manifests:
        name = item["name"]
        ckpt_dir = item["checkpoint_dir"].replace("/", "\\")
        eval_out = item["eval_output"].replace("/", "\\")
        lines.extend(
            [
                f"echo [{name}] eval",
                f"python run_evaluation.py \"{ckpt_dir}\" --output \"{eval_out}\" --batch_size 2",
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
            "echo Latent-anchor stress test 60 eval finished.",
            "echo Status log: %STATUS_LOG%",
            "echo Total failures: !FAIL_COUNT!",
            "if not \"!FAIL_COUNT!\"==\"0\" exit /b 1",
            "exit /b 0",
        ]
    )
    EVAL_BAT_PATH.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def main() -> None:
    manifests = generate_configs()
    generate_train_bat(manifests)
    generate_eval_bat(manifests)
    print(f"Generated {len(manifests)} configs under: {CONFIG_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Train runner: {TRAIN_BAT_PATH}")
    print(f"Eval runner: {EVAL_BAT_PATH}")


if __name__ == "__main__":
    main()
