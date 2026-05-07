from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "config.json"
OUT_ROOT = ROOT / "experiments" / "omf_ablation_matrix"
CONFIG_DIR = OUT_ROOT / "configs"
BAT_PATH = ROOT / "run_omf_ablation_matrix.bat"


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
    balanced_bridge = {
        "objective_mode": "omf",
        "loss_type": "omf",
        "w_kinetic": 2.0,
        "terminal_swd_weight": 25.0,
        "w_color": 15.0,
        "w_repulsive": 0.1,
    }
    common_training = {
        "num_epochs": 100,
        "save_interval": 20,
        "resume_checkpoint": "",
    }

    return [
        {
            "name": "01_strict_anchor",
            "notes": "strong kinetic anchor, no skip routing",
            "patch": {
                "bridge": {**balanced_bridge, "w_kinetic": 5.0, "terminal_swd_weight": 15.0, "w_color": 10.0, "w_repulsive": 0.1},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "02_balanced_omf",
            "notes": "balanced omf baseline",
            "patch": {
                "bridge": balanced_bridge,
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "03_aggressive_style",
            "notes": "maximize swd texture pressure",
            "patch": {
                "bridge": {**balanced_bridge, "w_kinetic": 1.0, "terminal_swd_weight": 40.0},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "04_arch_skip",
            "notes": "balanced omf with normalized skip routing and add_proj fusion",
            "patch": {
                "bridge": balanced_bridge,
                "training": common_training,
                "model": {
                    "skip_routing_mode": "normalized",
                    "skip_fusion_mode": "add_proj",
                },
            },
        },
        {
            "name": "05_color_05",
            "notes": "no contextual color branch",
            "patch": {
                "bridge": {**balanced_bridge, "w_color": 0.0},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "06_high_color",
            "notes": "strong contextual color branch",
            "patch": {
                "bridge": {**balanced_bridge, "w_color": 30.0},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "07_pure_physics",
            "notes": "pure kinetic plus swd without color or repulsive",
            "patch": {
                "bridge": {**balanced_bridge, "w_kinetic": 4.0, "terminal_swd_weight": 30.0, "w_color": 0.0, "w_repulsive": 0.0},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "08_heavy_repulsive",
            "notes": "balanced omf with heavier repulsive force",
            "patch": {
                "bridge": {**balanced_bridge, "w_repulsive": 0.5},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
    ]


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
        cfg["checkpoint"]["save_dir"] = f"./experiments/omf_ablation_matrix/artifacts/{exp_name}"
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
                "config_path": cfg_path.relative_to(ROOT).as_posix(),
                "checkpoint_path": f"experiments/omf_ablation_matrix/artifacts/{exp_name}/epoch_0100.pt",
                "eval_output": f"experiments/omf_ablation_matrix/full_eval/{exp_name}",
            }
        )

    write_json(OUT_ROOT / "manifest.json", {"experiments": manifests})
    return manifests


def generate_bat(manifests: list[dict]) -> None:
    lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\"",
        "",
        "set \"STATUS_DIR=experiments\\omf_ablation_matrix\"",
        "if not exist \"%STATUS_DIR%\" mkdir \"%STATUS_DIR%\"",
        "set \"STATUS_LOG=%STATUS_DIR%\\run_status.csv\"",
        "echo name,train_status,train_rc,eval_status,eval_rc,checkpoint_exists>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "set /a TRAIN_FAIL_COUNT=0",
        "set /a EVAL_FAIL_COUNT=0",
        "set /a SKIP_EVAL_COUNT=0",
        "",
        "echo Running OMF ablation matrix...",
        "echo.",
        "",
    ]
    for item in manifests:
        name = item["name"]
        cfg = item["config_path"].replace("/", "\\")
        ckpt = item["checkpoint_path"].replace("/", "\\")
        eval_out = item["eval_output"].replace("/", "\\")
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
                f"if exist \"{ckpt}\" (",
                f"  echo [{name}] eval",
                f"  python run_evaluation.py \"{ckpt}\" --output \"{eval_out}\" --batch_size 4",
                "  set \"EVAL_RC=!ERRORLEVEL!\"",
                "  if not \"!EVAL_RC!\"==\"0\" (",
                "    set /a FAIL_COUNT+=1",
                "    set /a EVAL_FAIL_COUNT+=1",
                "    set \"EVAL_STATUS=FAIL\"",
                "  ) else (",
                "    set \"EVAL_STATUS=OK\"",
                "  )",
                "  set \"CKPT_STATUS=YES\"",
                ") else (",
                f"  echo [{name}] checkpoint missing, skip eval",
                "  set /a FAIL_COUNT+=1",
                "  set /a SKIP_EVAL_COUNT+=1",
                "  set \"EVAL_RC=NA\"",
                "  set \"EVAL_STATUS=SKIP\"",
                "  set \"CKPT_STATUS=NO\"",
                ")",
                "echo "
                + f"{name}"
                + ",!TRAIN_STATUS!,!TRAIN_RC!,!EVAL_STATUS!,!EVAL_RC!,!CKPT_STATUS!>>\"%STATUS_LOG%\"",
                "echo.",
            ]
        )
    lines.extend(
        [
            "echo.",
            "echo OMF ablation runs finished.",
            "echo Status log: %STATUS_LOG%",
            "echo Total failures: !FAIL_COUNT! ^| train: !TRAIN_FAIL_COUNT! ^| eval: !EVAL_FAIL_COUNT! ^| skipped eval: !SKIP_EVAL_COUNT!",
            "if not \"!FAIL_COUNT!\"==\"0\" exit /b 1",
            "exit /b 0",
        ]
    )
    BAT_PATH.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def main() -> None:
    manifests = generate_configs()
    generate_bat(manifests)
    print(f"Generated {len(manifests)} configs under: {CONFIG_DIR}")
    print(f"Batch runner: {BAT_PATH}")


if __name__ == "__main__":
    main()
