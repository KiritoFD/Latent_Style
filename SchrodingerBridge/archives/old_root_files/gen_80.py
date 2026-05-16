"""Generate the 80-experiment next-round suite.  Fully self-contained, no external dependencies.

Usage:
    python gen_80.py                    # default settings
    python gen_80.py --batch-size 16    # override training batch size
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "next_experiment_candidates_80.csv"
SUITE_ROOT = ROOT / "next_round_80"
SUITE_BASE = SUITE_ROOT / "_suite_base.json"
MANIFEST_PATH = SUITE_ROOT / "manifest.json"
PLAN_CSV_PATH = SUITE_ROOT / "plan.csv"
RUN_BAT_PATH = SUITE_ROOT / "run_all.bat"
EVAL_BAT_PATH = SUITE_ROOT / "eval_all.bat"


# ---------------------------------------------------------------------------
# Base config: embedded from golden pedestal (fd20_g0_golden_pedestal_e15s1_b32)
# ---------------------------------------------------------------------------
def _base_config_template() -> dict:
    return {
        "model": {
            "latent_channels": 4,
            "base_dim": 64,
            "style_dim": 160,
            "time_dim": 256,
            "num_styles": 5,
            "lift_channels": 128,
            "num_hires_blocks": 2,
            "num_res_blocks": 4,
            "num_decoder_blocks": 2,
            "num_groups": 4,
            "latent_scale_factor": 0.18215,
            "residual_gain": 1.0,
            "style_spatial_pre_gain_16": 0.35,
            "style_strength_default": 1.0,
            "style_strength_step_curve": "linear",
            "upsample_mode": "nearest",
            "style_id_spatial_jitter_px": 0,
            "upsample_blur": True,
            "upsample_blur_kernel": "box3",
            "style_attn_num_tokens": 128,
            "style_attn_num_heads": 4,
            "style_attn_sharpen_scale": 2.5,
            "style_attn_temperature": 0.08,
            "hires_block_type": "conv",
            "body_block_type": "global_attn",
            "decoder_block_type": "conv",
            "semantic_attn_temperature": 0.12,
            "feature_attn_num_heads": 4,
            "window_attn_window_size": 8,
            "skip_fusion_mode": "add_proj",
            "skip_routing_mode": "none",
            "skip_naive_gain": 1.0,
            "style_skip_content_retention_boost": 0.0,
            "input_anchor_noise_std": 0.0,
            "input_anchor_noise_eval": False,
            "ablation_skip_clean": True,
            "ablation_skip_blur": True,
            "ablation_no_residual": False,
            "ablation_no_residual_gain": 1.0,
            "ablation_disable_spatial_prior": False,
            "output_moment_match": False,
            "output_moment_match_eps": 1e-06,
            "output_moment_match_train_only": False,
            "use_style_blender": False,
        },
        "bridge": {
            "objective_mode": "omf",
            "ot_cost_mode": "swd",
            "terminal_num_steps": 4,
            "terminal_swd_on_identity": False,
            "w_kinetic": 0.45,
            "w_low_freq": 1.0,
            "w_cycle": 0.2,
            "terminal_swd_weight": 10.0,
            "w_color": 0.0,
            "w_repulsive": 0.0,
            "w_nce": 0.0,
            "low_freq_kernel_size": 5,
            "semantic_swd_num_projections": 64,
            "swd_distance_mode": "cdf",
            "swd_use_high_freq": True,
            "swd_num_projections": 64,
            "swd_patch_sizes": [3, 5, 7, 15],
        },
        "training": {
            "seed": 42,
            "batch_size": 32,
            "accumulation_steps": 1,
            "num_workers": 2,
            "shuffle": False,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "pin_memory": True,
            "cpu_threads": 4,
            "cpu_interop_threads": 2,
            "learning_rate": 0.0002,
            "min_learning_rate": 1e-05,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "grad_clip_norm": 1.0,
            "num_epochs": 15,
            "save_interval": 1,
            "log_interval": 20,
            "use_tqdm": True,
            "use_amp": True,
            "amp_dtype": "bf16",
            "allow_tf32": True,
            "cudnn_benchmark": True,
            "channels_last": True,
            "use_gradient_checkpointing": True,
            "fused_adamw": True,
            "resume_checkpoint": "",
            "full_eval_batch_size": 6,
            "test_image_dir": "../style_data/overfit50",
            "full_eval_cache_dir": "../Cycle-NCE/eval_cache",
            "full_eval_image_classifier_path": "../Cycle-NCE/eval_cache/eval_style_image_classifier.pt",
            "full_eval_clip_hf_cache_dir": "../Cycle-NCE/eval_cache/hf",
            "full_eval_clip_backend": "hf",
            "full_eval_classifier_only": False,
            "full_eval_disable_lpips": False,
            "full_eval_enable_art_fid": False,
            "full_eval_enable_kid": False,
        },
        "data": {
            "data_root": "../latent-256",
            "style_subdirs": ["photo", "Hayao", "monet", "vangogh", "cezanne"],
            "allow_hflip": True,
            "balance_target_styles_per_batch": True,
            "preload_to_gpu": False,
            "preload_max_vram_gb": 6.0,
            "preload_reserve_ratio": 0.4,
            "virtual_length_multiplier": 1,
        },
        "checkpoint": {
            "save_dir": "./exp/next_round_80/__placeholder__",
        },
        "ablation": {
            "name": "next_round_80_base",
            "axis": "next_round_80",
            "notes": "Base config for the 80-experiment next-round suite.",
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_dir(exp_name: str) -> str:
    return f"./exp/next_round_80/{exp_name}"


def _config_rel(exp_name: str) -> str:
    return f"next_round_80/{exp_name}.json"


def _suite_base_payload(batch_size: int, eval_batch_size: int, num_epochs: int) -> dict:
    cfg = _base_config_template()
    cfg["training"]["num_epochs"] = num_epochs
    cfg["training"]["save_interval"] = 1
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["full_eval_batch_size"] = eval_batch_size
    cfg["training"]["resume_checkpoint"] = ""
    cfg["bridge"]["objective_mode"] = "omf"
    cfg["bridge"]["loss_type"] = "omf"
    cfg["bridge"]["w_low_freq"] = 0.0
    cfg["bridge"]["w_kinetic"] = 0.0
    cfg["bridge"]["w_cycle"] = 0.0
    cfg["bridge"]["terminal_swd_weight"] = 0.0
    cfg["bridge"]["w_color"] = 0.0
    cfg["bridge"]["w_repulsive"] = 0.0
    cfg["bridge"]["w_nce"] = 0.0
    cfg["bridge"]["swd_use_high_freq"] = False
    cfg["bridge"]["swd_hf_weight_ratio"] = 1.0
    cfg["checkpoint"]["save_dir"] = "./exp/next_round_80/__placeholder__"
    cfg["ablation"] = {
        "name": "next_round_80_base",
        "axis": "next_round_80",
        "notes": "Base config for the 80-experiment next-round suite.",
    }
    return cfg


def _parse_patch_sizes(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split("|") if x.strip()]


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def _load_csv(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("candidate_id", "").strip():
                continue
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Manifest & configs
# ---------------------------------------------------------------------------
def _build_manifest(rows: list[dict]) -> list[dict]:
    manifest = []
    for idx, row in enumerate(rows):
        cid = row["candidate_id"].strip()
        short_name = row["short_name"].strip()
        exp_name = cid
        manifest.append({
            "index": idx,
            "candidate_id": cid,
            "block": row["block"].strip(),
            "short_name": short_name,
            "exp_name": exp_name,
            "risk_level": row.get("risk_level", "").strip(),
            "config_path": _config_rel(exp_name),
            "run_dir": _run_dir(exp_name),
            "eval_dir": f"{_run_dir(exp_name)}/full_eval",
            "csv_row": row,
        })
    return manifest


def _row_to_config(row: dict, item: dict, num_epochs: int) -> dict:
    patch_sizes = _parse_patch_sizes(row.get("swd_patch_sizes", "3|5|7|15"))
    learning_rate = float(row.get("learning_rate", 0.0002))
    vlm = float(row.get("virtual_length_multiplier", 1.0))

    return {
        "_base": "./_suite_base.json",
        "model": {
            "skip_routing_mode": row["skip_routing_mode"].strip(),
            "skip_fusion_mode": row.get("skip_fusion_mode", "add_proj").strip(),
            "style_attn_temperature": float(row.get("style_attn_temperature", 0.08)),
            "semantic_attn_temperature": float(row.get("semantic_attn_temperature", 0.12)),
            "base_dim": int(row.get("base_dim", 64)),
            "num_res_blocks": int(row.get("num_res_blocks", 4)),
        },
        "bridge": {
            "w_kinetic": float(row["w_kinetic"]),
            "w_cycle": float(row["w_cycle"]),
            "terminal_swd_weight": float(row["terminal_swd_weight"]),
            "w_color": float(row.get("w_color", 0)),
            "w_low_freq": float(row.get("w_low_freq", 0)),
            "w_nce": float(row.get("w_nce", 0)),
            "w_repulsive": float(row.get("w_repulsive", 0)),
            "low_freq_kernel_size": int(row.get("low_freq_kernel_size", 7)),
            "swd_use_high_freq": row.get("swd_use_high_freq", "False").strip().lower() == "true",
            "swd_hf_weight_ratio": float(row.get("swd_hf_weight_ratio", 1.0)),
            "swd_patch_sizes": patch_sizes,
            "swd_num_projections": int(row.get("swd_num_projections", 64)),
            "semantic_swd_num_projections": int(row.get("semantic_swd_num_projections", 64)),
        },
        "training": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "save_interval": int(row.get("save_interval", 1)),
        },
        "data": {
            "virtual_length_multiplier": vlm,
        },
        "checkpoint": {
            "save_dir": item["run_dir"],
        },
        "ablation": {
            "name": item["exp_name"],
            "axis": item["block"],
            "notes": f"Grid {item['index'] + 1}/80 | {item['candidate_id']} | block={item['block']} risk={item['risk_level']} | {row.get('priority_note', '')}",
        },
    }


def _write_configs(manifest: list[dict], num_epochs: int) -> None:
    for item in manifest:
        payload = _row_to_config(item["csv_row"], item, num_epochs)
        _write_json(SUITE_ROOT / f"{item['exp_name']}.json", payload)


def _write_manifest(manifest: list[dict], batch_size: int, eval_batch_size: int, num_epochs: int) -> None:
    clean = [{k: v for k, v in item.items() if k != "csv_row"} for item in manifest]
    payload = {
        "suite": "next_round_80",
        "base_config": "embedded_in_gen_80.py",
        "csv_source": "next_experiment_candidates_80.csv",
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "num_epochs": num_epochs,
        "num_experiments": len(manifest),
        "experiments": clean,
    }
    _write_json(MANIFEST_PATH, payload)


def _write_plan_csv(manifest: list[dict]) -> None:
    fieldnames = [
        "index", "candidate_id", "block", "short_name", "exp_name", "risk_level",
        "config_path", "run_dir", "eval_dir",
        "skip_routing_mode", "w_kinetic", "w_cycle", "terminal_swd_weight",
        "w_color", "w_low_freq", "w_nce", "swd_use_high_freq",
        "swd_hf_weight_ratio", "swd_patch_sizes", "base_dim", "num_res_blocks",
        "learning_rate", "num_epochs",
    ]
    rows = []
    for item in manifest:
        r = item["csv_row"]
        rows.append({
            "index": item["index"],
            "candidate_id": item["candidate_id"],
            "block": item["block"],
            "short_name": item["short_name"],
            "exp_name": item["exp_name"],
            "risk_level": item["risk_level"],
            "config_path": item["config_path"],
            "run_dir": item["run_dir"],
            "eval_dir": item["eval_dir"],
            "skip_routing_mode": r["skip_routing_mode"],
            "w_kinetic": r["w_kinetic"],
            "w_cycle": r["w_cycle"],
            "terminal_swd_weight": r["terminal_swd_weight"],
            "w_color": r.get("w_color", 0),
            "w_low_freq": r.get("w_low_freq", 0),
            "w_nce": r.get("w_nce", 0),
            "swd_use_high_freq": r.get("swd_use_high_freq", "False"),
            "swd_hf_weight_ratio": r.get("swd_hf_weight_ratio", 1.0),
            "swd_patch_sizes": r.get("swd_patch_sizes", "3|5|7|15"),
            "base_dim": r.get("base_dim", 64),
            "num_res_blocks": r.get("num_res_blocks", 4),
            "learning_rate": r.get("learning_rate", 0.0002),
            "num_epochs": r.get("num_epochs", 8),
        })
    with PLAN_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Batch scripts
# ---------------------------------------------------------------------------
def _write_bats(manifest: list[dict], num_epochs: int) -> None:
    ckpt_tag = f"{num_epochs:04d}"
    run_lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        'cd /d "%~dp0\\.."',
        "",
        'set "STATUS_LOG=next_round_80\\train_eval_status.csv"',
        f'echo name,train_status,train_rc,checkpoint_epoch_{ckpt_tag},eval_status,eval_rc,batch_summary_exists>"%STATUS_LOG%"',
        "set /a FAIL_COUNT=0",
        "",
    ]
    eval_lines = [
        "@echo off",
        "setlocal",
        "setlocal EnableDelayedExpansion",
        'cd /d "%~dp0\\.."',
        "",
        'set "STATUS_LOG=next_round_80\\eval_status.csv"',
        'echo name,eval_status,eval_rc,batch_summary_exists>"%STATUS_LOG%"',
        "set /a FAIL_COUNT=0",
        "",
    ]
    for item in manifest:
        name = item["exp_name"]
        cfg = item["config_path"]
        run_dir = item["run_dir"]
        eval_dir = item["eval_dir"]
        run_lines.extend([
            f"echo [{name}] train",
            f'python run.py --config "{cfg}"',
            'set "TRAIN_RC=!ERRORLEVEL!"',
            f'if exist "{run_dir}\\epoch_{ckpt_tag}.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")',
            'if not "!TRAIN_RC!"=="0" (',
            "  set /a FAIL_COUNT+=1",
            '  set "TRAIN_STATUS=FAIL"',
            '  set "EVAL_STATUS=SKIPPED"',
            '  set "EVAL_RC=NA"',
            '  set "BATCH_STATUS=NO"',
            ") else (",
            '  set "TRAIN_STATUS=OK"',
            f"  echo [{name}] eval",
            f'  python run_evaluation.py "{run_dir}" --output "{eval_dir}" --batch_size 2',
            '  set "EVAL_RC=!ERRORLEVEL!"',
            f'  if exist "{eval_dir}\\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")',
            '  if not "!EVAL_RC!"=="0" (',
            "    set /a FAIL_COUNT+=1",
            '    set "EVAL_STATUS=FAIL"',
            "  ) else (",
            '    set "EVAL_STATUS=OK"',
            "  )",
            ")",
            f'echo {name},!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"',
            "echo.",
            "",
        ])
        eval_lines.extend([
            f"echo [{name}] eval",
            f'python run_evaluation.py "{run_dir}" --output "{eval_dir}" --batch_size 2',
            'set "EVAL_RC=!ERRORLEVEL!"',
            f'if exist "{eval_dir}\\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")',
            'if not "!EVAL_RC!"=="0" (',
            "  set /a FAIL_COUNT+=1",
            '  set "EVAL_STATUS=FAIL"',
            ") else (",
            '  set "EVAL_STATUS=OK"',
            ")",
            f'echo {name},!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"',
            "echo.",
            "",
        ])
    run_lines.extend([
        "echo Training done. Failures: %FAIL_COUNT%",
        "exit /b %FAIL_COUNT%",
        "",
    ])
    eval_lines.extend([
        "echo Eval done. Failures: %FAIL_COUNT%",
        "exit /b %FAIL_COUNT%",
        "",
    ])
    RUN_BAT_PATH.write_text("\n".join(run_lines), encoding="utf-8")
    EVAL_BAT_PATH.write_text("\n".join(eval_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the 80-run next-round experiment suite.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=6, help="Eval generation batch size.")
    parser.add_argument("--num-epochs", type=int, default=8, help="Number of training epochs.")
    args = parser.parse_args()

    if not DEFAULT_CSV.exists():
        raise SystemExit(f"CSV not found: {DEFAULT_CSV}\nPlace next_experiment_candidates_80.csv next to this script.")

    rows = _load_csv(DEFAULT_CSV)
    print(f"Loaded {len(rows)} experiments from {DEFAULT_CSV.name}")

    manifest = _build_manifest(rows)
    if len(manifest) > 256:
        raise SystemExit(f"Refuse to generate {len(manifest)} configs; limit is 256.")

    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(SUITE_BASE, _suite_base_payload(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
    ))
    _write_configs(manifest, args.num_epochs)
    _write_manifest(manifest, args.batch_size, args.eval_batch_size, args.num_epochs)
    _write_plan_csv(manifest)
    _write_bats(manifest, args.num_epochs)

    print(f"Generated {len(manifest)} experiment configs under: {SUITE_ROOT.name}/")
    print(f"Batch size: {args.batch_size}  |  Epochs: {args.num_epochs}")
    print(f"Run script: {RUN_BAT_PATH.name}")
    print(f"Eval script: {EVAL_BAT_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
