from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "exp" / "orth12" / "fd20_g0_golden_pedestal_e15s1_b32" / "config.json"
SUITE_ROOT = ROOT / "screening_grid_3epoch_108"
SUITE_BASE = SUITE_ROOT / "_suite_base.json"
MANIFEST_PATH = SUITE_ROOT / "manifest.json"
PLAN_CSV_PATH = SUITE_ROOT / "plan.csv"
RUN_BAT_PATH = SUITE_ROOT / "run_all.bat"
EVAL_BAT_PATH = SUITE_ROOT / "eval_all.bat"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_base_config() -> dict:
    return json.loads(BASE_CONFIG.read_text(encoding="utf-8"))


def _run_dir(name: str) -> str:
    return f"./exp/grid_search_3epoch/{name}"


def _config_rel(name: str) -> str:
    return f"screening_grid_3epoch_108/{name}.json"


def _suite_base_payload(batch_size: int, eval_batch_size: int) -> dict:
    cfg = _load_base_config()
    cfg["training"]["num_epochs"] = 3
    cfg["training"]["save_interval"] = 1
    cfg["training"]["batch_size"] = int(batch_size)
    cfg["training"]["full_eval_batch_size"] = int(eval_batch_size)
    cfg["training"]["resume_checkpoint"] = ""
    cfg["bridge"]["objective_mode"] = "omf"
    cfg["bridge"]["loss_type"] = "omf"
    cfg["bridge"]["swd_use_high_freq"] = False
    cfg["bridge"]["w_low_freq"] = 0.0
    cfg["bridge"]["w_kinetic"] = 0.0
    cfg["bridge"]["w_cycle"] = 0.0
    cfg["bridge"]["terminal_swd_weight"] = 0.0
    cfg["bridge"]["w_color"] = 0.0
    cfg["checkpoint"]["save_dir"] = "./exp/grid_search_3epoch/__placeholder__"
    cfg["ablation"] = {
        "name": "screening_base",
        "axis": "3epoch_screening",
        "notes": "Base poison-cleared config for the 3-epoch screening grid.",
    }
    return cfg


def _grid_space() -> dict[str, list]:
    return {
        "skip": ["none", "add_proj", "normalized"],
        "kin": [1.0, 2.0, 4.0],
        "cyc": [0.0, 2.0, 5.0],
        "swd": [10.0, 20.0],
        "col": [0.0, 15.0],
    }


def _exp_name(params: dict[str, object]) -> str:
    skip = str(params["skip"])
    kin = str(params["kin"]).replace(".0", "")
    cyc = str(params["cyc"]).replace(".0", "")
    swd = str(params["swd"]).replace(".0", "")
    col = str(params["col"]).replace(".0", "")
    return f"S-{skip[:4]}_K-{kin}_C-{cyc}_W-{swd}_Col-{col}"


def _build_manifest() -> list[dict]:
    space = _grid_space()
    keys = list(space.keys())
    combos = list(itertools.product(*(space[k] for k in keys)))
    manifest: list[dict] = []
    for idx, values in enumerate(combos):
        params = dict(zip(keys, values))
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
                "axis": "3epoch_screening",
                "notes": f"Grid {item['index'] + 1}/108 | skip={params['skip']} kin={params['kin']} cyc={params['cyc']} swd={params['swd']} color={params['col']}",
            },
        }
        _write_json(SUITE_ROOT / f"{item['name']}.json", payload)


def _write_manifest(manifest: list[dict], batch_size: int, eval_batch_size: int) -> None:
    payload = {
        "suite": "screening_grid_3epoch_108",
        "base_config": str(BASE_CONFIG),
        "batch_size": int(batch_size),
        "eval_batch_size": int(eval_batch_size),
        "num_experiments": len(manifest),
        "experiments": manifest,
    }
    _write_json(MANIFEST_PATH, payload)


def _write_plan_csv(manifest: list[dict], batch_size: int, eval_batch_size: int) -> None:
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
                "batch_size": batch_size,
                "eval_batch_size": eval_batch_size,
                "num_epochs": 3,
                "save_interval": 1,
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
                "batch_size",
                "eval_batch_size",
                "num_epochs",
                "save_interval",
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
        "",
        "set \"STATUS_LOG=screening_grid_3epoch_108\\train_eval_status.csv\"",
        "echo name,train_status,train_rc,checkpoint_epoch_0003,eval_status,eval_rc,batch_summary_exists>\"%STATUS_LOG%\"",
        "set /a FAIL_COUNT=0",
        "",
    ]
    eval_lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        "cd /d \"%~dp0\\..\"",
        "",
        "set \"STATUS_LOG=screening_grid_3epoch_108\\eval_status.csv\"",
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
                f"python run.py --config \"{cfg}\"",
                "set \"TRAIN_RC=!ERRORLEVEL!\"",
                f"if exist \"{run_dir}\\epoch_0003.pt\" (set \"CKPT_STATUS=YES\") else (set \"CKPT_STATUS=NO\")",
                "if not \"!TRAIN_RC!\"==\"0\" (",
                "  set /a FAIL_COUNT+=1",
                "  set \"TRAIN_STATUS=FAIL\"",
                "  set \"EVAL_STATUS=SKIPPED\"",
                "  set \"EVAL_RC=NA\"",
                "  set \"BATCH_STATUS=NO\"",
                ") else (",
                "  set \"TRAIN_STATUS=OK\"",
                f"  echo [{name}] eval",
                f"  python run_evaluation.py \"{run_dir}\" --output \"{eval_dir}\" --batch_size 2",
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
                f"python run_evaluation.py \"{run_dir}\" --output \"{eval_dir}\" --batch_size 2",
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
    run_lines.extend(
        [
            "echo Training done. Failures: %FAIL_COUNT%",
            "exit /b %FAIL_COUNT%",
            "",
        ]
    )
    eval_lines.extend(
        [
            "echo Eval done. Failures: %FAIL_COUNT%",
            "exit /b %FAIL_COUNT%",
            "",
        ]
    )
    RUN_BAT_PATH.write_text("\n".join(run_lines), encoding="utf-8")
    EVAL_BAT_PATH.write_text("\n".join(eval_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the 108-run 3-epoch screening suite.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size injected into the suite base config.")
    parser.add_argument("--eval-batch-size", type=int, default=6, help="Eval generation batch size stored in the suite base config.")
    args = parser.parse_args()

    manifest = _build_manifest()
    if len(manifest) > 256:
        raise SystemExit(f"Refuse to generate {len(manifest)} configs; limit is 256.")

    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(SUITE_BASE, _suite_base_payload(batch_size=args.batch_size, eval_batch_size=args.eval_batch_size))
    _write_configs(manifest)
    _write_manifest(manifest, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size)
    _write_plan_csv(manifest, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size)
    _write_bats(manifest)

    print(f"Generated {len(manifest)} screening configs under: {SUITE_ROOT}")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Train batch size: {args.batch_size}")
    print(f"Eval generation batch size: {args.eval_batch_size}")
    print(f"Run script: {RUN_BAT_PATH}")
    print(f"Eval script: {EVAL_BAT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
