"""628 Comprehensive Ablation: Inference-side new mechanisms (I5-I10).

Based on T5 ep7 checkpoint. Runs eval-only with various inference parameters.

Usage (WSL):
    python 628_infer_ablation.py <exp_name> <overrides_json>

Example:
    python 628_infer_ablation.py I5_cfg10 '{"fiber_cfg_scale": 1.0}'
    python 628_infer_ablation.py I6_vel15 '{"fiber_velocity_scale": 1.5}'
    python 628_infer_ablation.py I8_triband03 '{"tri_band_inference_lock": true, "tri_band_edge_lock_alpha": 0.3}'
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib

run_module = importlib.import_module("run")
config_schema = importlib.import_module("config_schema")

ExperimentConfig = config_schema.ExperimentConfig
load_config = config_schema.load_config

CKPT_PATH = Path(os.environ.get(
    "628_CKPT_PATH",
    str(ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "epoch_0007.pt")
))
CONFIG_PATH = Path(os.environ.get(
    "628_CONFIG_PATH",
    str(ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "config.json")
))
BASELINE_CLIP = float(os.environ.get("628_BASELINE_CLIP", "0.7307"))
BASELINE_LPIPS = float(os.environ.get("628_BASELINE_LPIPS", "0.3403"))

OUTPUT_DIR = ROOT / "exp" / "628_ablation" / "infer_ablation"


def _fix_paths(config, raw: dict = None, overrides: dict = None) -> None:
    pass


def _apply_overrides(config, raw: dict, overrides: dict) -> None:
    model_keys = {
        "fiber_cfg_scale", "fiber_velocity_scale", "fiber_source_repulse_scale",
        "tri_band_inference_lock", "tri_band_edge_lock_alpha", "fiber_only_endpoint",
        "lowpass_mode", "endpoint_adain_scale", "style_extrap_alpha",
        "patch_adain_kernel", "multiband_adain_mode", "mid_adain_scale",
        "hh_adain_scale", "endpoint_adain_mode", "fiber_cfg_null_style_id",
    }
    bridge_keys = set()

    for key, value in overrides.items():
        if key in model_keys:
            setattr(config.model, key, value)
            if not hasattr(config.model, 'extra'):
                config.model.extra = {}
            config.model.extra[key] = value
            raw.setdefault("model", {})[key] = value
        elif key in bridge_keys:
            setattr(config.bridge, key, value)
            raw.setdefault("bridge", {})[key] = value
        else:
            setattr(config.model, key, value)
            if not hasattr(config.model, 'extra'):
                config.model.extra = {}
            config.model.extra[key] = value
            raw.setdefault("model", {})[key] = value


def _extract_metrics(checkpoint_path: Path, config) -> dict:
    eval_subdir = str(
        getattr(config.training, "full_eval_output_subdir", "full_eval") or "full_eval"
    )
    out_dir = checkpoint_path.parent / eval_subdir / checkpoint_path.stem
    summary_path = out_dir / "summary.json"
    metrics = {
        "summary_path": str(summary_path),
        "transfer_clip_style": None,
        "transfer_content_lpips": None,
        "allpairs_clip_style": None,
        "allpairs_content_lpips": None,
    }
    if not summary_path.is_file():
        return metrics
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return metrics

    analysis = dict((summary.get("analysis") or {}))
    transfer = dict((analysis.get("style_transfer_ability") or {}))
    allpairs = dict((analysis.get("all_pairs_overview") or {}))
    metrics["transfer_clip_style"] = transfer.get("clip_style")
    metrics["transfer_content_lpips"] = transfer.get("content_lpips")
    metrics["allpairs_clip_style"] = allpairs.get("clip_style")
    metrics["allpairs_content_lpips"] = allpairs.get("content_lpips")
    return metrics


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python 628_infer_ablation.py <exp_name> <overrides_json>")
        sys.exit(2)

    exp_name = sys.argv[1]
    overrides = json.loads(sys.argv[2])

    print(f"[628_infer] exp={exp_name} overrides={overrides}")
    print(f"[628_infer] ckpt={CKPT_PATH}")
    print(f"[628_infer] config={CONFIG_PATH}")

    if not CKPT_PATH.exists():
        print(f"[628_infer] ERROR: checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    raw = load_config(str(CONFIG_PATH))
    config = ExperimentConfig.from_mapping(raw)
    for section_name in ("model", "bridge", "training", "data", "checkpoint", "full_eval"):
        raw_section = raw.get(section_name, {})
        section_obj = getattr(config, section_name, None)
        if not isinstance(raw_section, dict) or section_obj is None:
            continue
        for key, value in raw_section.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)

    _apply_overrides(config, raw, overrides)
    _fix_paths(config, raw, overrides)

    config.training.full_eval_output_subdir = f"full_eval_628_{exp_name}"
    print(f"[628_infer] eval_subdir={config.training.full_eval_output_subdir}")

    override_payload = {
        "model": dict(raw.get("model", {})),
        "bridge": dict(raw.get("bridge", {})),
    }
    for k, v in overrides.items():
        if k in override_payload.get("model", {}):
            override_payload["model"][k] = v
        else:
            override_payload.setdefault("model", {})[k] = v

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    override_path = OUTPUT_DIR / f"{exp_name}_override.json"
    with override_path.open("w", encoding="utf-8") as f:
        json.dump(override_payload, f, indent=2, ensure_ascii=False)
    config.training.full_eval_config_override = str(override_path)

    print(f"[628_infer] invoking full eval...")
    eval_result = run_module._run_full_eval_for_checkpoint(config, CKPT_PATH)
    print(f"[628_infer] eval done.")

    metrics = _extract_metrics(CKPT_PATH, config)

    record = {
        "exp_name": exp_name,
        "checkpoint": str(CKPT_PATH),
        "overrides": overrides,
        "baseline_reference": {"clip": BASELINE_CLIP, "lpips": BASELINE_LPIPS},
        "metrics": metrics,
        "convergence_payload": eval_result,
    }

    output_path = OUTPUT_DIR / f"{exp_name}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[628_infer] wrote {output_path}")

    c = metrics.get("allpairs_clip_style") or metrics.get("transfer_clip_style")
    l = metrics.get("allpairs_content_lpips") or metrics.get("transfer_content_lpips")
    print(f"[628_infer] {exp_name} | clip={c} lpips={l}")


if __name__ == "__main__":
    main()
