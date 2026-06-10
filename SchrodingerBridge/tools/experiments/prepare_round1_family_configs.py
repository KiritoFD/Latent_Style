from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config
from dino_cache_utils import default_dino_cache_output
from round1_registry import COMMON_PARENT_CONFIG, ROUND1_CONFIG_DIR, ROUND1_DOC_DIR, ROUND1_FAMILY_SPECS


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = deepcopy(value)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Round-1 family configs from the common parent.")
    parser.add_argument("--parent-config", default=COMMON_PARENT_CONFIG)
    parser.add_argument("--output-dir", default=ROUND1_CONFIG_DIR)
    parser.add_argument("--dino-cache-path", default="")
    parser.add_argument("--num-epochs", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--manifest-csv", default="")
    args = parser.parse_args()

    parent_path = (SB_ROOT.parent / args.parent_config).resolve()
    output_dir = (SB_ROOT.parent / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = (SB_ROOT.parent / ROUND1_DOC_DIR).resolve()
    doc_dir.mkdir(parents=True, exist_ok=True)
    payload = load_config(parent_path)
    manifest_rows: list[dict[str, object]] = []

    for spec in ROUND1_FAMILY_SPECS:
        cfg = deepcopy(payload)
        cfg.pop("_base", None)
        _deep_update(cfg.setdefault("model", {}), spec.model_overrides)
        _deep_update(cfg.setdefault("bridge", {}), spec.bridge_overrides)
        cfg.setdefault("training", {})
        cfg["training"]["num_epochs"] = int(args.num_epochs)
        cfg["training"]["save_interval"] = int(args.save_interval)
        cfg["training"]["full_eval_each_epoch"] = False
        cfg["training"]["full_eval_defer_until_training_end"] = True
        cfg["training"]["full_eval_only_lpips_clip_style"] = True
        cfg["training"]["full_eval_save_generated_images"] = False
        cfg["training"]["full_eval_save_summary_grid"] = False
        _deep_update(cfg["training"], spec.training_overrides)
        cfg.setdefault("data", {})
        _deep_update(cfg["data"], spec.data_overrides)
        if spec.axis == "tokenizer":
            cfg["training"]["freeze_mode"] = "style_branch"
        elif spec.axis == "backbone":
            cfg["training"]["freeze_mode"] = "attention_only"
        else:
            cfg["training"]["freeze_mode"] = "executor_only"
        needs_dino_cache = spec.axis == "tokenizer" or spec.bridge_overrides.get("semantic_supervision_family") == "dino_masked_swd"
        if needs_dino_cache and str(args.dino_cache_path).strip():
            cfg["data"]["dino_cache_path"] = str(args.dino_cache_path)
            cfg["data"]["dino_cache_required"] = True
        elif needs_dino_cache:
            latent_root = Path(str(cfg["data"].get("data_root", "")).strip())
            cfg["data"]["dino_cache_path"] = str(default_dino_cache_output(latent_root, workspace_root=SB_ROOT.parent))
            cfg["data"]["dino_cache_required"] = True
        else:
            cfg["data"].pop("dino_cache_path", None)
            cfg["data"]["dino_cache_required"] = False
        run_name = f"aaai2027_round1_{spec.family_id}_seed42_b8a2"
        cfg.setdefault("checkpoint", {})
        cfg["checkpoint"]["save_dir"] = f"./exp/inmortal-exp/{run_name}"
        cfg.setdefault("ablation", {})
        cfg["ablation"]["name"] = run_name
        cfg["ablation"]["axis"] = "aaai2027_round1_full_sweep"
        cfg["ablation"]["stage"] = spec.wave
        cfg["ablation"]["notes"] = spec.notes
        out_path = output_dir / f"{run_name}.json"
        out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        family_doc_dir = doc_dir / spec.family_id
        family_doc_dir.mkdir(parents=True, exist_ok=True)
        doc_templates = {
            "plan.md": f"# {spec.family_id} Plan\n\n- Wave: `{spec.wave}`\n- Axis: `{spec.axis}`\n- Notes: {spec.notes}\n",
            "remote_run.md": f"# {spec.family_id} Remote Run Log\n\n- Run dir: `{cfg['checkpoint']['save_dir']}`\n",
            "fast_curve_read.md": f"# {spec.family_id} Fast Curve Read\n\n- Curve CSV: `clip_lpips_curve.csv`\n",
            "local_deep_review.md": f"# {spec.family_id} Local Deep Review\n\n- Expected: `IntroStyle + DINO + frozen VLM`\n",
            "closure.md": f"# {spec.family_id} Closure\n\n- Status: pending\n",
        }
        for name, text in doc_templates.items():
            path = family_doc_dir / name
            if not path.exists():
                path.write_text(text, encoding="utf-8")
        manifest_rows.append(
            {
                "family_id": spec.family_id,
                "wave": spec.wave,
                "axis": spec.axis,
                "config_path": str(out_path),
                "run_name": run_name,
                "run_dir": cfg["checkpoint"]["save_dir"],
                "freeze_mode": cfg["training"]["freeze_mode"],
                "batch_size": cfg["training"].get("batch_size"),
                "accumulation_steps": cfg["training"].get("accumulation_steps"),
                "num_epochs": cfg["training"].get("num_epochs"),
                "patience": spec.patience,
                "notes": spec.notes,
                "virtual_length_multiplier": cfg["data"].get("virtual_length_multiplier"),
                "warmstart_config": "",
                "reconstruction_pretrain_config": "",
                "decision_status": "planned",
            }
        )
        print(out_path)
    manifest_path = Path(args.manifest_csv).resolve() if str(args.manifest_csv).strip() else (doc_dir / "round1_family_manifest.csv")
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family_id",
                "wave",
                "axis",
                "config_path",
                "run_name",
                "run_dir",
                "freeze_mode",
                "batch_size",
                "accumulation_steps",
                "num_epochs",
                "patience",
                "notes",
                "parent_config",
                "tokenizer_family",
                "backbone_attention_family",
                "solver_family",
                "semantic_supervision_family",
                "virtual_length_multiplier",
                "warmstart_config",
                "reconstruction_pretrain_config",
                "local_fast_root",
                "local_review_root",
                "switch_smoke_status",
                "switch_smoke_artifact",
                "switch_smoke_row_count",
                "best_ckpt",
                "best_transfer_lpips_ckpt",
                "best_allpairs_clip_style_ckpt",
                "latest_ckpt",
                "fast_converged",
                "convergence_reason",
                "decision_status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
