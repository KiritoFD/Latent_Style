from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "exp" / "orth12" / "fd20_g0_golden_pedestal_e15s1_b32" / "config.json"
SUITE_NAME = "pareto_probe_4"
SUITE_ROOT = ROOT / SUITE_NAME
SUITE_BASE = SUITE_ROOT / "_suite_base.json"
MANIFEST_PATH = SUITE_ROOT / "manifest.json"
PLAN_CSV_PATH = SUITE_ROOT / "plan.csv"
RUN_BAT_PATH = SUITE_ROOT / "run_all.bat"
EVAL_BAT_PATH = SUITE_ROOT / "eval_all.bat"
BUILD_CSV_BAT_PATH = SUITE_ROOT / "build_csv.bat"
GRID_ROOT = "./exp/pareto_probe_4"
PYTHON_EXE = r"C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_base_config() -> dict:
    return json.loads(BASE_CONFIG.read_text(encoding="utf-8"))


def _experiments() -> list[dict]:
    return [
        {"skip": "add_proj", "kin": 3.0, "cyc": 2.0, "swd": 10.0, "col": 15.0},
        {"skip": "normalized", "kin": 3.0, "cyc": 2.0, "swd": 10.0, "col": 15.0},
        {"skip": "add_proj", "kin": 5.0, "cyc": 5.0, "swd": 15.0, "col": 15.0},
        {"skip": "normalized", "kin": 5.0, "cyc": 5.0, "swd": 15.0, "col": 15.0},
    ]


def _exp_name(params: dict[str, object]) -> str:
    skip = str(params["skip"])
    kin = str(params["kin"]).replace(".0", "")
    cyc = str(params["cyc"]).replace(".0", "")
    swd = str(params["swd"]).replace(".0", "")
    col = str(params["col"]).replace(".0", "")
    return f"S-{skip[:4]}_K-{kin}_C-{cyc}_W-{swd}_Col-{col}"


def _run_dir(name: str) -> str:
    return f"{GRID_ROOT}/{name}"


def _config_rel(name: str) -> str:
    return f"{SUITE_NAME}/{name}.json"


def _suite_base_payload() -> dict:
    cfg = _load_base_config()
    cfg["training"]["learning_rate"] = 5e-5
    cfg["training"]["min_learning_rate"] = 1e-5
    cfg["training"]["num_epochs"] = 20
    cfg["training"]["save_interval"] = 1
    cfg["training"]["resume_checkpoint"] = ""
    cfg["training"]["batch_size"] = 32
    cfg["training"]["full_eval_batch_size"] = 6
    cfg["bridge"]["objective_mode"] = "omf"
    cfg["bridge"]["loss_type"] = "omf"
    cfg["bridge"]["swd_use_high_freq"] = False
    cfg["bridge"]["w_low_freq"] = 0.0
    cfg["data"]["virtual_length_multiplier"] = 0.5
    cfg["checkpoint"]["save_dir"] = f"{GRID_ROOT}/__placeholder__"
    cfg["ablation"] = {
        "name": SUITE_NAME,
        "axis": "pareto_probe",
        "notes": "4-run high-frequency Pareto probe with half-epoch sampling.",
    }
    return cfg


def _build_manifest() -> list[dict]:
    manifest: list[dict] = []
    for idx, params in enumerate(_experiments()):
        name = _exp_name(params)
        manifest.append(
            {
                "index": idx,
                "name": name,
                "config_path": _config_rel(name),
                "run_dir": _run_dir(name),
                "eval_dir": f"{_run_dir(name)}/full_eval",
                "params": params,
            }
        )
    return manifest


def _write_configs(manifest: list[dict]) -> None:
    for item in manifest:
        params = item["params"]
        payload = {
            "_base": "./_suite_base.json",
            "model": {
                "skip_routing_mode": params["skip"],
            },
            "bridge": {
                "w_kinetic": params["kin"],
                "w_cycle": params["cyc"],
                "terminal_swd_weight": params["swd"],
                "w_color": params["col"],
            },
            "checkpoint": {
                "save_dir": item["run_dir"],
            },
            "ablation": {
                "name": item["name"],
                "axis": "pareto_probe",
                "notes": (
                    f"Probe {item['index'] + 1}/4 | skip={params['skip']} "
                    f"kin={params['kin']} cyc={params['cyc']} swd={params['swd']} color={params['col']}"
                ),
            },
        }
        _write_json(SUITE_ROOT / f"{item['name']}.json", payload)


def _write_manifest(manifest: list[dict]) -> None:
    payload = {
        "suite": SUITE_NAME,
        "base_config": str(BASE_CONFIG),
        "grid_root": GRID_ROOT,
        "num_experiments": len(manifest),
        "experiments": manifest,
    }
    _write_json(MANIFEST_PATH, payload)


def _write_plan_csv(manifest: list[dict]) -> None:
    rows = []
    for item in manifest:
        p = item["params"]
        rows.append(
            {
                "index": item["index"],
                "name": item["name"],
                "config_path": item["config_path"],
                "run_dir": item["run_dir"],
                "eval_dir": item["eval_dir"],
                "learning_rate": 5e-5,
                "num_epochs": 20,
                "save_interval": 1,
                "virtual_length_multiplier": 0.5,
                "skip_routing_mode": p["skip"],
                "w_kinetic": p["kin"],
                "w_cycle": p["cyc"],
                "terminal_swd_weight": p["swd"],
                "w_color": p["col"],
                "swd_use_high_freq": False,
                "w_low_freq": 0.0,
            }
        )
    with PLAN_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "name",
                "config_path",
                "run_dir",
                "eval_dir",
                "learning_rate",
                "num_epochs",
                "save_interval",
                "virtual_length_multiplier",
                "skip_routing_mode",
                "w_kinetic",
                "w_cycle",
                "terminal_swd_weight",
                "w_color",
                "swd_use_high_freq",
                "w_low_freq",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_bats(manifest: list[dict]) -> None:
    run_lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\\..\"",
        f"set \"PYTHON_EXE={PYTHON_EXE}\"",
        "set \"PYTHONHOME=\"",
        "",
        f"set \"STATUS_LOG={SUITE_NAME}\\train_eval_status.csv\"",
        "echo name,train_status,train_rc,checkpoint_epoch_0020,eval_status,eval_rc,batch_summary_exists>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "",
    ]
    eval_lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\\..\"",
        f"set \"PYTHON_EXE={PYTHON_EXE}\"",
        "set \"PYTHONHOME=\"",
        "",
        f"set \"STATUS_LOG={SUITE_NAME}\\eval_status.csv\"",
        "echo name,eval_status,eval_rc,batch_summary_exists>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "",
    ]
    for item in manifest:
        name = item["name"]
        cfg = item["config_path"]
        run_dir = item["run_dir"]
        eval_dir = item["eval_dir"]
        run_lines.extend(
            [
                f"echo [{name}] train",
                f"\"%PYTHON_EXE%\" run.py --config \"{cfg}\"",
                "set \"TRAIN_RC=!ERRORLEVEL!\"",
                f"if exist \"{run_dir}\\epoch_0020.pt\" (set \"CKPT_STATUS=YES\") else (set \"CKPT_STATUS=NO\")",
                "if not \"!TRAIN_RC!\"==\"0\" (",
                "  set /a FAIL_COUNT+=1",
                "  set \"TRAIN_STATUS=FAIL\"",
                "  set \"EVAL_STATUS=SKIPPED\"",
                "  set \"EVAL_RC=NA\"",
                "  set \"BATCH_STATUS=NO\"",
                ") else (",
                "  set \"TRAIN_STATUS=OK\"",
                f"  echo [{name}] eval",
                f"  \"%PYTHON_EXE%\" run_evaluation.py \"{run_dir}\" --output \"{eval_dir}\" --batch_size 2",
                "  set \"EVAL_RC=!ERRORLEVEL!\"",
                f"  if exist \"{eval_dir}\\batch_summary.csv\" (set \"BATCH_STATUS=YES\") else (set \"BATCH_STATUS=NO\")",
                "  if not \"!EVAL_RC!\"==\"0\" (",
                "    set /a FAIL_COUNT+=1",
                "    set \"EVAL_STATUS=FAIL\"",
                "  ) else (",
                "    set \"EVAL_STATUS=OK\"",
                "  )",
                ")",
                f"echo {name},!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>\"%STATUS_LOG%\"",
                "echo.",
                "",
            ]
        )
        eval_lines.extend(
            [
                f"echo [{name}] eval",
                f"\"%PYTHON_EXE%\" run_evaluation.py \"{run_dir}\" --output \"{eval_dir}\" --batch_size 2",
                "set \"EVAL_RC=!ERRORLEVEL!\"",
                f"if exist \"{eval_dir}\\batch_summary.csv\" (set \"BATCH_STATUS=YES\") else (set \"BATCH_STATUS=NO\")",
                "if not \"!EVAL_RC!\"==\"0\" (",
                "  set /a FAIL_COUNT+=1",
                "  set \"EVAL_STATUS=FAIL\"",
                ") else (",
                "  set \"EVAL_STATUS=OK\"",
                ")",
                f"echo {name},!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>\"%STATUS_LOG%\"",
                "echo.",
                "",
            ]
        )

    run_lines.extend(["echo Training done. Failures: %FAIL_COUNT%", "exit /b %FAIL_COUNT%", ""])
    eval_lines.extend(["echo Eval done. Failures: %FAIL_COUNT%", "exit /b %FAIL_COUNT%", ""])
    build_csv_lines = [
        "@echo off",
        "setlocal",
        "cd /d \"%~dp0\\..\"",
        f"set \"PYTHON_EXE={PYTHON_EXE}\"",
        "set \"PYTHONHOME=\"",
        f"\"%PYTHON_EXE%\" build_csv.py \"{GRID_ROOT}\"",
    ]

    RUN_BAT_PATH.write_text("\n".join(run_lines), encoding="utf-8")
    EVAL_BAT_PATH.write_text("\n".join(eval_lines), encoding="utf-8")
    BUILD_CSV_BAT_PATH.write_text("\n".join(build_csv_lines) + "\n", encoding="utf-8")


def main() -> int:
    manifest = _build_manifest()
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(SUITE_BASE, _suite_base_payload())
    _write_configs(manifest)
    _write_manifest(manifest)
    _write_plan_csv(manifest)
    _write_bats(manifest)

    print(f"Generated {len(manifest)} configs under: {SUITE_ROOT}")
    print(f"Grid root: {GRID_ROOT}")
    print(f"Run script: {RUN_BAT_PATH}")
    print(f"Eval script: {EVAL_BAT_PATH}")
    print(f"CSV script: {BUILD_CSV_BAT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
