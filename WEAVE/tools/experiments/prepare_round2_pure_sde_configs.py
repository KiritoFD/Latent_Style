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
from round2_registry import COMMON_PARENT_CONFIG, ROUND2_CONFIG_DIR, ROUND2_DOC_DIR, ROUND2_PURE_SDE_SPECS


DEFAULT_TRAIN_DATA_ROOT = "/mnt/i/wikiarts_5_full_notest_latents_ema/train"
DEFAULT_STYLE_SUBDIRS = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]
DEFAULT_LATENT_CACHE_DIR = f"{DEFAULT_TRAIN_DATA_ROOT}/.latent_cache"
DEFAULT_PAIRING_CACHE_PATH = f"{DEFAULT_LATENT_CACHE_DIR}/prototype_pairing_top8.pt"
DEFAULT_TEST_IMAGE_DIR = "/mnt/i/wikiart_distinct5_samam_512_classview/test"
DEFAULT_FULL_EVAL_CACHE_DIR = "/mnt/i/Github/Latent_Style/eval_cache"
DEFAULT_FULL_EVAL_CLIP_HF_CACHE_DIR = "/mnt/i/Github/Latent_Style/eval_cache/hf"


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = deepcopy(value)
    return dst


def _freeze_mode_for_axis(axis: str) -> str:
    axis = str(axis).strip().lower()
    if axis == "tokenizer":
        return "style_branch"
    return "none"


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Round-2 pure-latent/I2SB configs from the common parent.")
    parser.add_argument("--parent-config", default=COMMON_PARENT_CONFIG)
    parser.add_argument("--output-dir", default=ROUND2_CONFIG_DIR)
    parser.add_argument("--manifest-csv", default="")
    parser.add_argument("--num-epochs", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=1)
    args = parser.parse_args()

    parent_path = (SB_ROOT.parent / args.parent_config).resolve()
    output_dir = (SB_ROOT.parent / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = (SB_ROOT.parent / ROUND2_DOC_DIR).resolve()
    doc_dir.mkdir(parents=True, exist_ok=True)
    payload = load_config(parent_path)
    manifest_rows: list[dict[str, object]] = []

    for spec in ROUND2_PURE_SDE_SPECS:
        cfg = deepcopy(payload)
        cfg.pop("_base", None)
        _deep_update(cfg.setdefault("model", {}), spec.model_overrides)
        _deep_update(cfg.setdefault("bridge", {}), spec.bridge_overrides)
        cfg.setdefault("training", {})
        cfg["training"]["num_epochs"] = int(args.num_epochs)
        cfg["training"]["save_interval"] = int(args.save_interval)
        cfg["training"]["full_eval_each_epoch"] = True
        cfg["training"]["full_eval_defer_until_training_end"] = False
        cfg["training"]["full_eval_only_lpips_clip_style"] = True
        cfg["training"]["full_eval_save_generated_images"] = False
        cfg["training"]["full_eval_save_summary_grid"] = False
        cfg["training"]["full_eval_batch_size"] = 1
        cfg["training"]["full_eval_vae_decode_batch_size"] = 4
        cfg["training"]["test_image_dir"] = DEFAULT_TEST_IMAGE_DIR
        cfg["training"]["full_eval_cache_dir"] = DEFAULT_FULL_EVAL_CACHE_DIR
        cfg["training"]["full_eval_clip_hf_cache_dir"] = DEFAULT_FULL_EVAL_CLIP_HF_CACHE_DIR
        cfg["training"]["freeze_mode"] = _freeze_mode_for_axis(spec.axis)
        cfg["training"]["resume_model_strict"] = True
        cfg["training"]["resume_ignore_prefixes"] = []
        cfg["training"]["resume_include_prefixes"] = []
        _deep_update(cfg["training"], spec.training_overrides)
        cfg.setdefault("data", {})
        cfg["data"]["data_root"] = DEFAULT_TRAIN_DATA_ROOT
        cfg["data"]["style_subdirs"] = list(DEFAULT_STYLE_SUBDIRS)
        cfg["data"]["latent_cache_dir"] = DEFAULT_LATENT_CACHE_DIR
        cfg["data"]["pairing_cache_path"] = DEFAULT_PAIRING_CACHE_PATH
        _deep_update(cfg["data"], spec.data_overrides)
        cfg["data"].pop("dino_cache_path", None)
        cfg["data"]["dino_cache_required"] = False
        run_name = f"aaai2027_round2_{spec.family_id}_seed42_b8a2"
        cfg.setdefault("checkpoint", {})
        cfg["checkpoint"]["save_dir"] = f"./exp/inmortal-exp/{run_name}"
        cfg.setdefault("ablation", {})
        cfg["ablation"]["name"] = run_name
        cfg["ablation"]["axis"] = "aaai2027_round2_pure_sde"
        cfg["ablation"]["stage"] = spec.wave
        cfg["ablation"]["notes"] = spec.notes
        out_path = output_dir / f"{run_name}.json"
        out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        family_doc_dir = doc_dir / spec.family_id
        family_doc_dir.mkdir(parents=True, exist_ok=True)
        doc_templates = {
            "plan.md": (
                f"# {spec.family_id} Plan\n\n"
                f"- Wave: `{spec.wave}`\n"
                f"- Axis: `{spec.axis}`\n"
                f"- Notes: {spec.notes}\n"
                "- DINO policy: archived-only unless overwhelming gain appears.\n"
            ),
            "remote_run.md": f"# {spec.family_id} Remote Run Log\n\n- Run dir: `{cfg['checkpoint']['save_dir']}`\n",
            "fast_curve_read.md": f"# {spec.family_id} Fast Curve Read\n\n- Curve CSV: `clip_lpips_curve.csv`\n",
            "local_deep_review.md": f"# {spec.family_id} Local Deep Review\n\n- Expected: `IntroStyle + frozen VLM shortlist`\n",
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
                "launch_health_min_runtime_memory_mib": spec.launch_health_min_runtime_memory_mib,
                "notes": spec.notes,
                "decision_status": "planned",
            }
        )
        print(out_path)

    manifest_path = Path(args.manifest_csv).resolve() if str(args.manifest_csv).strip() else (doc_dir / "round2_family_manifest.csv")
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
                "launch_health_min_runtime_memory_mib",
                "notes",
                "decision_status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
