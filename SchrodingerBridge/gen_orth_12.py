from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUITE_ROOT = ROOT / "full_dimensional_orthogonal_sweep_20"
BASE_CONFIG_PATH = SUITE_ROOT / "_suite_base.json"
MANIFEST_PATH = SUITE_ROOT / "manifest.json"
PLAN_CSV_PATH = SUITE_ROOT / "plan.csv"
RUN_BAT_PATH = SUITE_ROOT / "run_all.bat"
EVAL_BAT_PATH = SUITE_ROOT / "eval_all.bat"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_dir(name: str) -> str:
    return f"./exp/runs/fd20_{name}"


def _config_path(name: str) -> str:
    return f"full_dimensional_orthogonal_sweep_20/{name}.json"


def _write_suite_base() -> None:
    payload = {
        "_base": "../config.json",
        "model": {
            "base_dim": 64,
            "num_res_blocks": 4,
            "semantic_attn_temperature": 0.12,
        },
        "bridge": {
            "objective_mode": "omf",
            "ot_cost_mode": "swd",
            "terminal_num_steps": 4,
            "terminal_swd_on_identity": False,
            "terminal_swd_weight": 10.0,
            "w_kinetic": 0.45,
            "w_low_freq": 1.0,
            "w_cycle": 0.20,
            "w_color": 0.0,
            "w_repulsive": 0.0,
            "w_nce": 0.0,
            "low_freq_kernel_size": 5,
            "semantic_swd_num_projections": 64,
            "swd_num_projections": 64,
            "swd_patch_sizes": [3, 5, 7, 15],
            "swd_distance_mode": "cdf",
            "swd_use_high_freq": True,
        },
        "training": {
            "batch_size": 108,
            "learning_rate": 2e-4,
            "num_epochs": 60,
            "save_interval": 10,
            "resume_checkpoint": "",
        },
    }
    _write_json(BASE_CONFIG_PATH, payload)


def _specs() -> list[dict]:
    return [
        {
            "name": "g0_golden_pedestal",
            "axis": "center",
            "title": "G0 Golden Pedestal",
            "notes": "Universe center: damped M2-style base with AdaIN low-frequency anchor and multi-scale [3,5,7,15] SWD pyramid.",
            "patch": {},
        },
        {
            "name": "g1_micro_only",
            "axis": "patch_scales",
            "title": "G1 Micro-Only",
            "notes": "Micro-only patch bank to probe the lower bound of SWD receptive field.",
            "patch": {"bridge": {"swd_patch_sizes": [1, 3, 5]}},
        },
        {
            "name": "g2_macro_only",
            "axis": "patch_scales",
            "title": "G2 Macro-Only",
            "notes": "Macro-only patch bank to probe the upper bound of SWD receptive field.",
            "patch": {"bridge": {"swd_patch_sizes": [9, 15, 21]}},
        },
        {
            "name": "g3_bimodal_split",
            "axis": "patch_scales",
            "title": "G3 Bimodal Split",
            "notes": "Bimodal patch bank with a hollowed-out mid band to test scale discontinuity.",
            "patch": {"bridge": {"swd_patch_sizes": [3, 25]}},
        },
        {
            "name": "g4_high_tension",
            "axis": "dynamics",
            "title": "G4 High Tension",
            "notes": "Raise SWD tension and kinetic damping together to test the high-pressure operating ceiling.",
            "patch": {"bridge": {"terminal_swd_weight": 15.0, "w_kinetic": 0.60}},
        },
        {
            "name": "g5_low_tension",
            "axis": "dynamics",
            "title": "G5 Low Tension",
            "notes": "Relax both pull and damping to probe a gentler drift regime.",
            "patch": {"bridge": {"terminal_swd_weight": 6.0, "w_kinetic": 0.30}},
        },
        {
            "name": "g6_zero_friction",
            "axis": "dynamics",
            "title": "G6 Zero Friction",
            "notes": "Keep G0 SWD pressure but collapse kinetic damping toward zero.",
            "patch": {"bridge": {"w_kinetic": 0.10}},
        },
        {
            "name": "g7_sharp_ot",
            "axis": "temperature",
            "title": "G7 Sharp OT",
            "notes": "Lower semantic attention temperature to 0.08 and test the sharp-routing edge.",
            "patch": {"model": {"semantic_attn_temperature": 0.08}},
        },
        {
            "name": "g8_soft_ot",
            "axis": "temperature",
            "title": "G8 Soft OT",
            "notes": "Raise semantic attention temperature to 0.16 and test the over-smoothing edge.",
            "patch": {"model": {"semantic_attn_temperature": 0.16}},
        },
        {
            "name": "g9_strict_l1",
            "axis": "low_freq_kernel",
            "title": "G9 Strict L1",
            "notes": "Shrink the low-frequency kernel to 3 and lock more of the mid band to the AdaIN anchor.",
            "patch": {"bridge": {"low_freq_kernel_size": 3}},
        },
        {
            "name": "g10_loose_l1",
            "axis": "low_freq_kernel",
            "title": "G10 Loose L1",
            "notes": "Expand the low-frequency kernel to 7 and let more mid-scale structure flow through SWD.",
            "patch": {"bridge": {"low_freq_kernel_size": 7}},
        },
        {
            "name": "g11_cycle_drop",
            "axis": "cycle",
            "title": "G11 Cycle Drop",
            "notes": "Remove the cycle penalty and measure whether forward-only damping plus AdaIN is enough.",
            "patch": {"bridge": {"w_cycle": 0.0}},
        },
    ]


def _write_config(spec: dict) -> dict:
    name = spec["name"]
    payload = {
        "_base": "./_suite_base.json",
        "checkpoint": {"save_dir": _run_dir(name)},
        "ablation": {
            "name": name,
            "axis": spec["axis"],
            "notes": spec["notes"],
        },
    }
    for key, value in spec["patch"].items():
        payload[key] = value
    cfg_path = SUITE_ROOT / f"{name}.json"
    _write_json(cfg_path, payload)
    return {
        "name": name,
        "title": spec["title"],
        "axis": spec["axis"],
        "notes": spec["notes"],
        "config_path": _config_path(name),
        "run_dir": _run_dir(name),
        "eval_dir": f"{_run_dir(name)}/full_eval",
        "checkpoint_epoch_10": f"{_run_dir(name)}/epoch_0010.pt",
        "checkpoint_epoch_20": f"{_run_dir(name)}/epoch_0020.pt",
        "patch_overrides": spec["patch"],
    }


def _write_manifest(manifest: list[dict]) -> None:
    _write_json(MANIFEST_PATH, {"suite": "full_dimensional_orthogonal_sweep_20", "experiments": manifest})


def _write_plan_csv(manifest: list[dict]) -> None:
    rows = []
    for item in manifest:
        bridge = item["patch_overrides"].get("bridge", {})
        model = item["patch_overrides"].get("model", {})
        rows.append(
            {
                "name": item["name"],
                "title": item["title"],
                "axis": item["axis"],
                "config_path": item["config_path"],
                "run_dir": item["run_dir"],
                "eval_dir": item["eval_dir"],
                "terminal_swd_weight": bridge.get("terminal_swd_weight", 10.0),
                "w_kinetic": bridge.get("w_kinetic", 0.45),
                "semantic_attn_temperature": model.get("semantic_attn_temperature", 0.12),
                "w_cycle": bridge.get("w_cycle", 0.20),
                "w_low_freq": bridge.get("w_low_freq", 1.0),
                "low_freq_kernel_size": bridge.get("low_freq_kernel_size", 5),
                "swd_patch_sizes": json.dumps(bridge.get("swd_patch_sizes", [3, 5, 7, 15])),
                "notes": item["notes"],
            }
        )
    with PLAN_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "title",
                "axis",
                "config_path",
                "run_dir",
                "eval_dir",
                "terminal_swd_weight",
                "w_kinetic",
                "semantic_attn_temperature",
                "w_cycle",
                "w_low_freq",
                "low_freq_kernel_size",
                "swd_patch_sizes",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_bat_scripts(manifest: list[dict]) -> None:
    run_lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\\..\"",
        "",
        "set \"STATUS_LOG=full_dimensional_orthogonal_sweep_20\\train_status.csv\"",
        "echo name,train_status,train_rc,checkpoint_exists>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "",
    ]
    eval_lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\\..\"",
        "",
        "set \"STATUS_LOG=full_dimensional_orthogonal_sweep_20\\eval_status.csv\"",
        "echo name,eval_status,eval_rc>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "",
    ]

    for item in manifest:
        cfg = item["config_path"].replace("/", "\\")
        run_dir = item["run_dir"].replace("/", "\\").replace("./", "")
        eval_dir = item["eval_dir"].replace("/", "\\").replace("./", "")
        run_lines.extend(
            [
                f"echo [{item['name']}] train",
                f"python run.py --config \"{cfg}\"",
                "set \"TRAIN_RC=!ERRORLEVEL!\"",
                "if exist \"" + run_dir + "\\epoch_0020.pt\" (set \"CKPT_STATUS=YES\") else (set \"CKPT_STATUS=NO\")",
                "if not \"!TRAIN_RC!\"==\"0\" (",
                "  set /a FAIL_COUNT+=1",
                "  set \"TRAIN_STATUS=FAIL\"",
                ") else (",
                "  set \"TRAIN_STATUS=OK\"",
                ")",
                f"echo {item['name']},!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>\"%STATUS_LOG%\"",
                "echo.",
            ]
        )
        eval_lines.extend(
            [
                f"echo [{item['name']}] eval",
                f"python run_evaluation.py \"{run_dir}\" --output \"{eval_dir}\" --batch_size 2",
                "set \"EVAL_RC=!ERRORLEVEL!\"",
                "if not \"!EVAL_RC!\"==\"0\" (",
                "  set /a FAIL_COUNT+=1",
                "  set \"EVAL_STATUS=FAIL\"",
                ") else (",
                "  set \"EVAL_STATUS=OK\"",
                ")",
                f"echo {item['name']},!EVAL_STATUS!,!EVAL_RC!>>\"%STATUS_LOG%\"",
                "echo.",
            ]
        )

    run_lines.extend(
        [
            "echo Full-dimensional orthogonal sweep training finished.",
            "echo Status log: %STATUS_LOG%",
            "if not \"!FAIL_COUNT!\"==\"0\" exit /b 1",
            "exit /b 0",
        ]
    )
    eval_lines.extend(
        [
            "echo Full-dimensional orthogonal sweep evaluation finished.",
            "echo Status log: %STATUS_LOG%",
            "if not \"!FAIL_COUNT!\"==\"0\" exit /b 1",
            "exit /b 0",
        ]
    )
    RUN_BAT_PATH.write_text("\r\n".join(run_lines) + "\r\n", encoding="utf-8")
    EVAL_BAT_PATH.write_text("\r\n".join(eval_lines) + "\r\n", encoding="utf-8")


def main() -> None:
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_suite_base()
    manifest = [_write_config(spec) for spec in _specs()]
    _write_manifest(manifest)
    _write_plan_csv(manifest)
    _write_bat_scripts(manifest)
    print(SUITE_ROOT)


if __name__ == "__main__":
    main()
