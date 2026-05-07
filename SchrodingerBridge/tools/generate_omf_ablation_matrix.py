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
    common_bridge = {
        "objective_mode": "omf",
        "loss_type": "mse",
        "w_kinetic": 1.0,
        "terminal_swd_weight": 20.0,
        "w_color": 8.0,
        "w_repulsive": 0.2,
    }
    common_training = {
        "num_epochs": 100,
        "save_interval": 20,
        "resume_checkpoint": "",
    }

    return [
        {
            "name": "01_gold_no_skip",
            "notes": "gold baseline with no skip routing",
            "patch": {
                "bridge": common_bridge,
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "02_gold_norm_skip",
            "notes": "gold baseline with normalized skip routing",
            "patch": {
                "bridge": common_bridge,
                "training": common_training,
                "model": {"skip_routing_mode": "normalized"},
            },
        },
        {
            "name": "03_no_repulsive",
            "notes": "remove repulsive term entirely",
            "patch": {
                "bridge": {**common_bridge, "w_repulsive": 0.0},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "04_repulsive_05",
            "notes": "slightly stronger repulsive term",
            "patch": {
                "bridge": {**common_bridge, "w_repulsive": 0.5},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "05_color_05",
            "notes": "lower contextual color pressure",
            "patch": {
                "bridge": {**common_bridge, "w_color": 5.0},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "06_color_12",
            "notes": "stronger contextual color pressure",
            "patch": {
                "bridge": {**common_bridge, "w_color": 12.0},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "07_swd_10",
            "notes": "lighter terminal swd texture force",
            "patch": {
                "bridge": {**common_bridge, "terminal_swd_weight": 10.0},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "08_swd_30",
            "notes": "stronger terminal swd texture force",
            "patch": {
                "bridge": {**common_bridge, "terminal_swd_weight": 30.0},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "09_kinetic_05",
            "notes": "lighter kinetic regularization",
            "patch": {
                "bridge": {**common_bridge, "w_kinetic": 0.5},
                "training": common_training,
                "model": {"skip_routing_mode": "none"},
            },
        },
        {
            "name": "10_kinetic_20_norm_skip",
            "notes": "stronger kinetic regularization plus normalized skip",
            "patch": {
                "bridge": {**common_bridge, "w_kinetic": 2.0},
                "training": common_training,
                "model": {"skip_routing_mode": "normalized"},
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
        "cd /d \"%~dp0\"",
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
                "if errorlevel 1 goto :error",
                f"echo [{name}] eval",
                f"python run_evaluation.py \"{ckpt}\" --output \"{eval_out}\" --batch_size 4",
                "if errorlevel 1 goto :error",
                "echo.",
            ]
        )
    lines.extend(
        [
            "echo All OMF ablation runs finished.",
            "exit /b 0",
            "",
            ":error",
            "echo.",
            "echo OMF ablation matrix aborted due to an error.",
            "exit /b 1",
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
