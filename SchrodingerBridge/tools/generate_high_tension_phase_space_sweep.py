from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "config.json"
OUT_ROOT = ROOT / "experiments" / "high_tension_phase_space_sweep"
MANIFEST_PATH = OUT_ROOT / "manifest.json"
TRAIN_BAT_PATH = ROOT / "run_high_tension_phase_space_sweep.bat"
EVAL_BAT_PATH = ROOT / "eval_high_tension_phase_space_sweep.bat"


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
    locked_model = {
        "base_dim": 64,
        "num_res_blocks": 4,
    }
    locked_training = {
        "batch_size": 32,
        "learning_rate": 2e-4,
        "resume_checkpoint": "",
    }
    bridge_common = {
        "objective_mode": "omf",
        "ot_cost_mode": "swd",
        "terminal_num_steps": 4,
        "terminal_swd_on_identity": False,
        "w_low_freq": 1.0,
        "w_cycle": 1.0,
        "w_color": 0.0,
        "w_repulsive": 0.0,
        "w_nce": 0.0,
        "low_freq_kernel_size": 7,
        "semantic_swd_num_projections": 64,
        "swd_distance_mode": "cdf",
        "swd_use_high_freq": True,
    }
    specs = [
        (
            "g1_high_tension_base",
            "G1 [High-Tension Base]: kinetic 2.0 vs terminal SWD 5.0 with full priors.",
            {"w_kinetic": 2.0, "terminal_swd_weight": 5.0},
        ),
        (
            "g2_swd_nuke",
            "G2 [SWD Nuke]: kinetic 2.0 vs terminal SWD 12.0 with full priors.",
            {"w_kinetic": 2.0, "terminal_swd_weight": 12.0},
        ),
        (
            "g3_kinetic_vise",
            "G3 [Kinetic Vise]: kinetic 5.0 vs terminal SWD 5.0 with full priors.",
            {"w_kinetic": 5.0, "terminal_swd_weight": 5.0},
        ),
        (
            "g4_brittle_flow",
            "G4 [Brittle Flow]: kinetic 0.5 vs terminal SWD 8.0 with full priors.",
            {"w_kinetic": 0.5, "terminal_swd_weight": 8.0},
        ),
        (
            "g5_the_singularity",
            "G5 [The Singularity]: kinetic 3.0 vs terminal SWD 15.0 with full priors.",
            {"w_kinetic": 3.0, "terminal_swd_weight": 15.0},
        ),
        (
            "g6_cycle_ablation",
            "G6 [Cycle Ablation]: cycle prior removed under kinetic 2.0 and terminal SWD 8.0.",
            {"w_kinetic": 2.0, "terminal_swd_weight": 8.0, "w_cycle": 0.0},
        ),
        (
            "g7_freq_ablation",
            "G7 [Freq Ablation]: low-frequency prior removed under kinetic 2.0 and terminal SWD 8.0.",
            {"w_kinetic": 2.0, "terminal_swd_weight": 8.0, "w_low_freq": 0.0},
        ),
        (
            "g8_sweet_spot",
            "G8 [Sweet Spot]: kinetic 1.5 vs terminal SWD 8.0 with full priors.",
            {"w_kinetic": 1.5, "terminal_swd_weight": 8.0},
        ),
    ]

    experiments: list[dict] = []
    for name, notes, bridge_delta in specs:
        experiments.append(
            {
                "name": name,
                "notes": notes,
                "patch": {
                    "model": locked_model,
                    "training": locked_training,
                    "bridge": {**bridge_common, **bridge_delta},
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
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    num_epochs = int(base.get("training", {}).get("num_epochs", 80))
    checkpoint_name = f"epoch_{num_epochs:04d}.pt"

    for exp in experiments:
        cfg = copy.deepcopy(base)
        deep_update(cfg, exp["patch"])
        exp_name = exp["name"]
        cfg["checkpoint"]["save_dir"] = f"./experiments/high_tension_phase_space_sweep/{exp_name}"
        cfg["ablation"] = {
            "name": exp_name,
            "notes": exp["notes"],
        }
        cfg_path = OUT_ROOT / f"{exp_name}.json"
        write_json(cfg_path, cfg)
        manifests.append(
            {
                "name": exp_name,
                "notes": exp["notes"],
                "config_path": f"experiments/high_tension_phase_space_sweep/{exp_name}.json",
                "checkpoint_dir": f"experiments/high_tension_phase_space_sweep/{exp_name}",
                "checkpoint_path": f"experiments/high_tension_phase_space_sweep/{exp_name}/{checkpoint_name}",
                "eval_output": f"experiments/high_tension_phase_space_sweep/full_eval/{exp_name}",
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
        "if not exist \"experiments\\high_tension_phase_space_sweep\" mkdir \"experiments\\high_tension_phase_space_sweep\"",
        "set \"STATUS_LOG=experiments\\high_tension_phase_space_sweep\\train_status.csv\"",
        "echo name,train_status,train_rc,checkpoint_exists>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "set /a TRAIN_FAIL_COUNT=0",
        "",
        "echo Running high-tension phase space sweep...",
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
            "echo High-tension phase space sweep finished.",
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
        "if not exist \"experiments\\high_tension_phase_space_sweep\\full_eval\" mkdir \"experiments\\high_tension_phase_space_sweep\\full_eval\"",
        "set \"STATUS_LOG=experiments\\high_tension_phase_space_sweep\\eval_status.csv\"",
        "echo name,eval_status,eval_rc>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "",
        "echo Evaluating high-tension phase space sweep...",
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
            "echo High-tension phase space sweep eval finished.",
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
    print(f"Generated {len(manifests)} configs under: {OUT_ROOT}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Train runner: {TRAIN_BAT_PATH}")
    print(f"Eval runner: {EVAL_BAT_PATH}")


if __name__ == "__main__":
    main()
