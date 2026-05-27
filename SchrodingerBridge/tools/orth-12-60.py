from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "config.json"
OUT_ROOT = ROOT / "orthogonal_phase_space_sweep_60"
MANIFEST_PATH = OUT_ROOT / "manifest.json"
TRAIN_BAT_PATH = ROOT / "run_orthogonal_phase_space_sweep_60.bat"
EVAL_BAT_PATH = ROOT / "eval_orthogonal_phase_space_sweep_60.bat"


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
        "semantic_attn_temperature": 0.08,
    }
    locked_training = {
        "batch_size": 108,
        "learning_rate": 2e-4,
        "num_epochs": 60,
        "save_interval": 10,
        "resume_checkpoint": "",
    }
    bridge_common = {
        "objective_mode": "omf",
        "ot_cost_mode": "swd",
        "terminal_num_steps": 4,
        "terminal_swd_on_identity": False,
        "w_kinetic": 1.5,
        "terminal_swd_weight": 8.0,
        "w_low_freq": 1.0,
        "w_cycle": 1.0,
        "w_color": 0.0,
        "w_repulsive": 0.0,
        "w_nce": 0.0,
        "low_freq_kernel_size": 5,
        "semantic_swd_num_projections": 64,
        "swd_distance_mode": "cdf",
        "swd_use_high_freq": True,
    }
    specs = [
        (
            "g0_universe_center",
            "G0 [Universe Center]: kinetic 1.5, terminal SWD 8.0, kernel 5, cycle 1.0, temp 0.08, proj 64.",
            "center",
            {},
        ),
        (
            "g1_absolute_release",
            "G1 [Absolute Release]: reduced kinetic damping to probe manifold boundary.",
            "friction",
            {"bridge": {"w_kinetic": 0.5}},
        ),
        (
            "g2_absolute_freeze",
            "G2 [Absolute Freeze]: elevated kinetic damping to probe optimization lockup.",
            "friction",
            {"bridge": {"w_kinetic": 4.0}},
        ),
        (
            "g3_gravity_black_hole",
            "G3 [Gravity Black Hole]: extreme terminal SWD tension to probe gradient ceiling.",
            "tension",
            {"bridge": {"terminal_swd_weight": 20.0}},
        ),
        (
            "g4_gravity_vacuum",
            "G4 [Gravity Vacuum]: weakened terminal SWD tension to probe gradient floor.",
            "tension",
            {"bridge": {"terminal_swd_weight": 2.0}},
        ),
        (
            "g5_midfreq_strangulation",
            "G5 [Mid-Frequency Strangulation]: tighten low-frequency lock with kernel 3.",
            "cutoff",
            {"bridge": {"low_freq_kernel_size": 3}},
        ),
        (
            "g6_structure_amnesty",
            "G6 [Structure Amnesty]: relax low-frequency lock with kernel 9.",
            "cutoff",
            {"bridge": {"low_freq_kernel_size": 9}},
        ),
        (
            "g7_flesh_stripping",
            "G7 [Flesh Stripping]: remove cycle prior to probe topological stiffness floor.",
            "topology",
            {"bridge": {"w_cycle": 0.0}},
        ),
        (
            "g8_absolute_nailgun",
            "G8 [Absolute Nailgun]: overconstrain cycle prior to probe topological stiffness ceiling.",
            "topology",
            {"bridge": {"w_cycle": 5.0}},
        ),
        (
            "g9_cryogenic_hard_match",
            "G9 [Cryogenic Hard Match]: lower semantic attention temperature for hard OT alignment.",
            "entropy",
            {"model": {"semantic_attn_temperature": 0.02}},
        ),
        (
            "g10_thermal_soft_collapse",
            "G10 [Thermal Soft Collapse]: raise semantic attention temperature toward soft-collapse.",
            "entropy",
            {"model": {"semantic_attn_temperature": 0.20}},
        ),
        (
            "g11_blind_men_slicing",
            "G11 [Blind Men Slicing]: reduce semantic SWD projections to 16.",
            "radon",
            {"bridge": {"semantic_swd_num_projections": 16}},
        ),
        (
            "g12_limit_approximation",
            "G12 [Limit Approximation]: increase semantic SWD projections to 128.",
            "radon",
            {"bridge": {"semantic_swd_num_projections": 128}},
        ),
    ]

    experiments: list[dict] = []
    for name, notes, axis, patch in specs:
        merged_patch = {
            "model": copy.deepcopy(locked_model),
            "training": copy.deepcopy(locked_training),
            "bridge": copy.deepcopy(bridge_common),
        }
        deep_update(merged_patch, patch)
        experiments.append(
            {
                "name": name,
                "axis": axis,
                "notes": notes,
                "patch": merged_patch,
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

    for exp in experiments:
        cfg = copy.deepcopy(base)
        deep_update(cfg, exp["patch"])
        exp_name = exp["name"]
        cfg["checkpoint"]["save_dir"] = f"./orthogonal_phase_space_sweep_60/{exp_name}"
        cfg["ablation"] = {
            "name": exp_name,
            "axis": exp["axis"],
            "notes": exp["notes"],
        }
        cfg_path = OUT_ROOT / f"{exp_name}.json"
        write_json(cfg_path, cfg)
        manifests.append(
            {
                "name": exp_name,
                "axis": exp["axis"],
                "notes": exp["notes"],
                "config_path": f"orthogonal_phase_space_sweep_60/{exp_name}.json",
                "checkpoint_dir": f"orthogonal_phase_space_sweep_60/{exp_name}",
                "checkpoint_path": f"orthogonal_phase_space_sweep_60/{exp_name}/epoch_0060.pt",
                "eval_output": f"orthogonal_phase_space_sweep_60/full_eval/{exp_name}",
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
        "if not exist \"orthogonal_phase_space_sweep_60\" mkdir \"orthogonal_phase_space_sweep_60\"",
        "set \"STATUS_LOG=orthogonal_phase_space_sweep_60\\train_status.csv\"",
        "echo name,train_status,train_rc,checkpoint_exists>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "set /a TRAIN_FAIL_COUNT=0",
        "",
        "echo Running orthogonal phase space sweep 60...",
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
            "echo Orthogonal phase space sweep 60 finished.",
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
        "if not exist \"orthogonal_phase_space_sweep_60\\full_eval\" mkdir \"orthogonal_phase_space_sweep_60\\full_eval\"",
        "set \"STATUS_LOG=orthogonal_phase_space_sweep_60\\eval_status.csv\"",
        "echo name,eval_status,eval_rc>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "",
        "echo Evaluating orthogonal phase space sweep 60...",
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
            "echo Orthogonal phase space sweep 60 eval finished.",
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



  