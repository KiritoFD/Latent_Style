"""628 Inference-time ablation batch runner.

Runs inference-side ablations #9-#12 (and reuses #1-#8 parameters) on T5 ep7 checkpoint.
Each ablation only changes inference-time parameters (no retraining).

Ablation matrix:
  #9  bridge_path_mode      : linear / spherical_vp / vertical+tri_band_lock
  #10 swd_distance_mode     : squared (vs baseline cdf)
  #11 full_eval_num_steps   : 4 / 16 / 32 (vs baseline 8)
  #12 full_eval_style_strength : 0.5 / 1.5 / 2.0 (vs baseline 1.0)

Each ablation writes results to exp/628_ablation/infer_ablation/<name>.json.

Usage:
    python _628_infer_ablation_batch.py                 # run all
    python _628_infer_ablation_batch.py <name>          # run single
    python _628_infer_ablation_batch.py --list          # list all
"""
from __future__ import annotations

import json
import os
import sys
import time
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

# T5 ep7 checkpoint (baseline: clip=0.7307, lpips=0.3403)
CKPT_PATH = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "epoch_0007.pt"
CONFIG_PATH = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "config.json"
BASELINE_CLIP = 0.7307
BASELINE_LPIPS = 0.3403

OUTPUT_DIR = ROOT / "exp" / "628_ablation" / "infer_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Ablation matrix: (name, desc, model_overrides, bridge_overrides, training_overrides)
ABLATIONS = [
    # ===== #9: bridge_path_mode 推理切换 =====
    {
        "name": "I9a_bridge_path_linear",
        "desc": "bridge_path_mode=linear (vs baseline vertical)",
        "model": {},
        "bridge": {"bridge_path_mode": "linear"},
        "training": {},
    },
    {
        "name": "I9b_bridge_path_spherical_vp",
        "desc": "bridge_path_mode=spherical_vp (constant variance path)",
        "model": {},
        "bridge": {"bridge_path_mode": "spherical_vp"},
        "training": {},
    },
    {
        "name": "I9c_bridge_path_vertical_triband",
        "desc": "bridge_path_mode=vertical + tri_band_inference_lock=True (tri-band locking)",
        "model": {"tri_band_inference_lock": True},
        "bridge": {"bridge_path_mode": "vertical"},
        "training": {},
    },
    # ===== #10: swd_distance_mode 推理切换 =====
    {
        "name": "I10a_swd_squared",
        "desc": "swd_distance_mode=squared (vs baseline cdf)",
        "model": {},
        "bridge": {"swd_distance_mode": "squared"},
        "training": {},
    },
    # ===== #11: full_eval_num_steps 扫描 =====
    {
        "name": "I11a_num_steps_4",
        "desc": "num_steps=4 (vs baseline 8) - fewer ODE steps",
        "model": {},
        "bridge": {},
        "training": {"full_eval_num_steps": 4},
    },
    {
        "name": "I11b_num_steps_16",
        "desc": "num_steps=16 - more ODE steps",
        "model": {},
        "bridge": {},
        "training": {"full_eval_num_steps": 16},
    },
    {
        "name": "I11c_num_steps_32",
        "desc": "num_steps=32 - many ODE steps (saturation test)",
        "model": {},
        "bridge": {},
        "training": {"full_eval_num_steps": 32},
    },
    # ===== #12: full_eval_style_strength 扫描 =====
    {
        "name": "I12a_style_strength_05",
        "desc": "style_strength=0.5 (weaker style transfer)",
        "model": {},
        "bridge": {},
        "training": {"full_eval_style_strength": 0.5},
    },
    {
        "name": "I12b_style_strength_15",
        "desc": "style_strength=1.5 (stronger style transfer)",
        "model": {},
        "bridge": {},
        "training": {"full_eval_style_strength": 1.5},
    },
    {
        "name": "I12c_style_strength_20",
        "desc": "style_strength=2.0 (extreme style transfer)",
        "model": {},
        "bridge": {},
        "training": {"full_eval_style_strength": 2.0},
    },
]


def _fix_windows_paths(config) -> None:
    """Config uses /mnt/i/ WSL paths; runtime is Windows native."""
    if hasattr(config.training, "test_image_dir"):
        config.training.test_image_dir = "I:/wikiart_distinct5_samam_512_classview/test"
    if hasattr(config.training, "full_eval_cache_dir"):
        config.training.full_eval_cache_dir = "I:/Github/Latent_Style/eval_cache"
    if hasattr(config.training, "full_eval_clip_hf_cache_dir"):
        config.training.full_eval_clip_hf_cache_dir = "I:/Github/Latent_Style/eval_cache/hf"
    if hasattr(config.data, "data_root"):
        config.data.data_root = "I:/wikiart_distinct5_samam_512_latents_ema/train"
    if hasattr(config.data, "latent_cache_dir"):
        config.data.latent_cache_dir = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
    if hasattr(config.data, "pairing_cache_path"):
        config.data.pairing_cache_path = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt"


def _apply_overrides(config, raw: dict, ablation: dict) -> None:
    """Apply ablation overrides to dataclass config and raw payload."""
    # model overrides
    for k, v in ablation.get("model", {}).items():
        setattr(config.model, k, v)
        if hasattr(config.model, "extra"):
            config.model.extra[k] = v
        raw.setdefault("model", {})[k] = v
    # bridge overrides
    for k, v in ablation.get("bridge", {}).items():
        setattr(config.bridge, k, v)
        raw.setdefault("bridge", {})[k] = v
    # training overrides
    for k, v in ablation.get("training", {}).items():
        setattr(config.training, k, v)
        raw.setdefault("training", {})[k] = v


def _extract_metrics(checkpoint_path: Path, config) -> dict:
    """Read summary.json written by _run_full_eval_for_checkpoint."""
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
        "wfi_score": None,
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

    wfi = summary.get("wfi_benchmark") or {}
    gen_wfi = dict((wfi.get("generated_wfi") or {}))
    wfi_score = gen_wfi.get("wfi_score")
    if isinstance(wfi_score, dict):
        metrics["wfi_score"] = wfi_score.get("mean")
    else:
        metrics["wfi_score"] = wfi_score
    return metrics


def run_single(ablation: dict) -> dict:
    """Run a single inference ablation."""
    name = ablation["name"]
    print(f"\n{'='*60}")
    print(f"[628_infer_ablation] {name}")
    print(f"  desc: {ablation['desc']}")
    print(f"  model: {ablation.get('model', {})}")
    print(f"  bridge: {ablation.get('bridge', {})}")
    print(f"  training: {ablation.get('training', {})}")
    print(f"{'='*60}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"checkpoint not found: {CKPT_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config not found: {CONFIG_PATH}")

    raw = load_config(str(CONFIG_PATH))
    config = ExperimentConfig.from_mapping(raw)
    # Merge raw sections into config (align with run.py behavior)
    for section_name in ("model", "bridge", "training", "data", "checkpoint", "full_eval"):
        raw_section = raw.get(section_name, {})
        section_obj = getattr(config, section_name, None)
        if not isinstance(raw_section, dict) or section_obj is None:
            continue
        for key, value in raw_section.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)

    _apply_overrides(config, raw, ablation)
    _fix_windows_paths(config)

    # Each ablation uses a unique eval output subdir to avoid summary.json being overwritten
    config.training.full_eval_output_subdir = f"full_eval_628_infer_{name}"
    print(f"[628_infer_ablation] output_subdir={config.training.full_eval_output_subdir}")

    # Build override payload for run_evaluation.py (model + bridge + training)
    override_payload = {
        "model": dict(raw.get("model", {})),
        "bridge": dict(raw.get("bridge", {})),
        "training": dict(raw.get("training", {})),
    }
    override_path = OUTPUT_DIR / f"{name}_override.json"
    with override_path.open("w", encoding="utf-8") as f:
        json.dump(override_payload, f, indent=2, ensure_ascii=False)
    config.training.full_eval_config_override = str(override_path)
    print(f"[628_infer_ablation] config_override={override_path}")

    output_path = OUTPUT_DIR / f"{name}.json"
    if output_path.exists():
        print(f"[628_infer_ablation] SKIP (already done): {output_path}")
        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[628_infer_ablation] invoking full eval...")
    t0 = time.time()
    eval_result = run_module._run_full_eval_for_checkpoint(config, CKPT_PATH)
    elapsed = time.time() - t0
    print(f"[628_infer_ablation] eval done in {elapsed:.0f}s. convergence_payload={eval_result}")

    metrics = _extract_metrics(CKPT_PATH, config)

    record = {
        "exp_name": name,
        "desc": ablation["desc"],
        "checkpoint": str(CKPT_PATH),
        "config_path": str(CONFIG_PATH),
        "overrides": {
            "model": ablation.get("model", {}),
            "bridge": ablation.get("bridge", {}),
            "training": ablation.get("training", {}),
        },
        "baseline_reference": {"clip": BASELINE_CLIP, "lpips": BASELINE_LPIPS},
        "metrics": metrics,
        "convergence_payload": eval_result,
        "elapsed_sec": round(elapsed, 1),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[628_infer_ablation] wrote {output_path}")

    print(
        f"[628_infer_ablation] {name} | "
        f"clip_style={metrics['transfer_clip_style']} "
        f"lpips={metrics['transfer_content_lpips']} "
        f"wfi={metrics['wfi_score']}"
    )
    return record


def main() -> None:
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            print("Available ablations:")
            for a in ABLATIONS:
                print(f"  {a['name']}: {a['desc']}")
            return
        name = sys.argv[1]
        ablation = next((a for a in ABLATIONS if a["name"] == name), None)
        if not ablation:
            print(f"Unknown ablation: {name}. Use --list to see available.")
            sys.exit(2)
        run_single(ablation)
        return

    # Run all
    print(f"[628_infer_ablation] Running {len(ABLATIONS)} ablations on T5 ep7")
    print(f"[628_infer_ablation] checkpoint: {CKPT_PATH}")
    print(f"[628_infer_ablation] baseline: clip={BASELINE_CLIP} lpips={BASELINE_LPIPS}")

    results = []
    for i, ablation in enumerate(ABLATIONS, 1):
        print(f"\n[628_infer_ablation] === [{i}/{len(ABLATIONS)}] ===")
        try:
            record = run_single(ablation)
            results.append(record)
        except Exception as e:
            print(f"[628_infer_ablation] ERROR in {ablation['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"exp_name": ablation["name"], "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"[628_infer_ablation] SUMMARY ({len(results)} ablations)")
    print(f"{'='*60}")
    print(f"{'name':<35} {'clip':>8} {'lpips':>8} {'d_clip':>8} {'d_lpips':>8}")
    for r in results:
        name = r.get("exp_name", "?")
        m = r.get("metrics", {})
        clip = m.get("transfer_clip_style")
        lpips = m.get("transfer_content_lpips")
        if clip is None or lpips is None:
            print(f"{name:<35} {'ERR':>8} {'ERR':>8}")
            continue
        d_clip = clip - BASELINE_CLIP
        d_lpips = lpips - BASELINE_LPIPS
        print(f"{name:<35} {clip:>8.4f} {lpips:>8.4f} {d_clip:>+8.4f} {d_lpips:>+8.4f}")

    # Save summary CSV
    summary_path = OUTPUT_DIR / "summary.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("name,desc,clip,lpips,d_clip,d_lpips,wfi\n")
        for r in results:
            name = r.get("exp_name", "?")
            desc = r.get("desc", "").replace(",", ";")
            m = r.get("metrics", {})
            clip = m.get("transfer_clip_style")
            lpips = m.get("transfer_content_lpips")
            wfi = m.get("wfi_score")
            if clip is None:
                f.write(f"{name},{desc},ERR,ERR,ERR,ERR,ERR\n")
            else:
                d_clip = clip - BASELINE_CLIP
                d_lpips = lpips - BASELINE_LPIPS
                f.write(f"{name},{desc},{clip},{lpips},{d_clip},{d_lpips},{wfi}\n")
    print(f"[628_infer_ablation] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
