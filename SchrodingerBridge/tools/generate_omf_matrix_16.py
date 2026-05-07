from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "config.json"
OUT_ROOT = ROOT / "experiments"
CONFIG_DIR = OUT_ROOT
BAT_PATH = ROOT / "run_omf_matrix_16.bat"


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
        "num_epochs": 160,
        "save_interval": 20,
        "resume_checkpoint": "",
    }

    specs = [
        ("01_omf_swd_15", {"kin": 2.0, "swd": 15.0, "col": 15.0, "rep": 0.1, "skip": "none", "ot_mse": 0.0}),
        ("02_omf_swd_30", {"kin": 2.0, "swd": 30.0, "col": 15.0, "rep": 0.1, "skip": "none", "ot_mse": 0.0}),
        ("03_omf_swd_45", {"kin": 2.0, "swd": 45.0, "col": 15.0, "rep": 0.1, "skip": "none", "ot_mse": 0.0}),
        ("04_anchor_kin_only", {"kin": 4.0, "swd": 25.0, "col": 15.0, "rep": 0.1, "skip": "none", "ot_mse": 0.0}),
        ("05_anchor_ot_mse_only", {"kin": 0.0, "swd": 25.0, "col": 15.0, "rep": 0.1, "skip": "none", "ot_mse": 1.0}),
        ("06_anchor_skip_only", {"kin": 0.0, "swd": 25.0, "col": 15.0, "rep": 0.1, "skip": "add_proj", "ot_mse": 0.0}),
        ("07_anchor_hybrid_all", {"kin": 2.0, "swd": 25.0, "col": 15.0, "rep": 0.1, "skip": "add_proj", "ot_mse": 0.5}),
        ("08_color_00", {"kin": 2.0, "swd": 25.0, "col": 0.0, "rep": 0.1, "skip": "none", "ot_mse": 0.0}),
        ("09_color_25", {"kin": 2.0, "swd": 25.0, "col": 25.0, "rep": 0.1, "skip": "none", "ot_mse": 0.0}),
        ("10_color_50", {"kin": 2.0, "swd": 25.0, "col": 50.0, "rep": 0.1, "skip": "none", "ot_mse": 0.0}),
        ("11_repel_00", {"kin": 2.0, "swd": 25.0, "col": 15.0, "rep": 0.0, "skip": "none", "ot_mse": 0.0}),
        ("12_repel_05", {"kin": 2.0, "swd": 25.0, "col": 15.0, "rep": 0.5, "skip": "none", "ot_mse": 0.0}),
        ("13_repel_10", {"kin": 2.0, "swd": 25.0, "col": 15.0, "rep": 1.0, "skip": "none", "ot_mse": 0.0}),
        ("14_extreme_free_flow", {"kin": 0.0, "swd": 40.0, "col": 15.0, "rep": 0.2, "skip": "none", "ot_mse": 0.0}),
        ("15_extreme_stiff_ode", {"kin": 8.0, "swd": 10.0, "col": 10.0, "rep": 0.0, "skip": "add_proj", "ot_mse": 0.0}),
        ("16_the_god_weight", {"kin": 1.5, "swd": 30.0, "col": 15.0, "rep": 0.2, "skip": "add_proj", "ot_mse": 0.3}),
    ]

    experiments: list[dict] = []
    for name, params in specs:
        model_patch = {"skip_routing_mode": "none"}
        if params["skip"] == "add_proj":
            model_patch = {
                "skip_fusion_mode": "add_proj",
                "skip_routing_mode": "normalized",
            }
        bridge_patch = {
            "objective_mode": "omf",
            "loss_type": "omf" if params["ot_mse"] <= 0.0 else "mse",
            "w_kinetic": params["kin"],
            "terminal_swd_weight": params["swd"],
            "w_color": params["col"],
            "w_repulsive": params["rep"],
            "w_flow": params["ot_mse"],
        }
        experiments.append(
            {
                "name": name,
                "notes": f"16-run orthogonal matrix: {name}",
                "patch": {
                    "bridge": bridge_patch,
                    "training": common_training,
                    "model": model_patch,
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
        cfg["ablation"] = {"name": exp_name, "notes": exp["notes"]}
        cfg_path = CONFIG_DIR / f"{exp_name}.json"
        write_json(cfg_path, cfg)
        manifests.append(
            {
                "name": exp_name,
                "notes": exp["notes"],
                "config_path": f"experiments/{exp_name}.json",
                "checkpoint_path": f"experiments/{exp_name}/epoch_0160.pt",
            }
        )

    write_json(OUT_ROOT / "omf_matrix_16_manifest.json", {"experiments": manifests})
    return manifests


def generate_bat(manifests: list[dict]) -> None:
    lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\"",
        "",
        "if not exist \"experiments\" mkdir \"experiments\"",
        "set \"STATUS_LOG=experiments\\omf_matrix_16_run_status.csv\"",
        "echo name,train_status,train_rc,checkpoint_exists>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "set /a TRAIN_FAIL_COUNT=0",
        "",
        "echo Running OMF 16-run matrix...",
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
            "echo OMF 16-run training matrix finished.",
            "echo Status log: %STATUS_LOG%",
            "echo Total failures: !FAIL_COUNT! ^| train: !TRAIN_FAIL_COUNT!",
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
