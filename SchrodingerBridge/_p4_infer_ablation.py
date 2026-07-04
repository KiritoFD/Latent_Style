"""FC-SB Phase 4 P4: Inference-time ablation over E4-long ep5 checkpoint.

Loads E4-long ep5 checkpoint (baseline: clip=0.727, lpips=0.581) and re-evaluates
with various inference-time ablation parameters (lowpass_mode, style_extrap_alpha,
patch_adain_kernel, multiband_adain_mode, tri_band_inference_lock, mid_adain_scale,
hh_adain_scale). Results are written to exp/p4_fusion_breakout/infer_ablation/<exp>.json.

Usage:
    python _p4_infer_ablation.py <exp_name> <lowpass_mode> [style_extrap_alpha] \
        [patch_adain_kernel] [multiband_adain_mode] [tri_band_inference_lock] \
        [mid_adain_scale] [hh_adain_scale]

Example:
    python _p4_infer_ablation.py D0_baseline avg_pool 0 0 single 0 0.3 0.3
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

# v5: 支持通过环境变量切换 checkpoint（E4-long ep5 或 B2 V2 ep1）
# E4-long ep5 (project memory: clip=0.727, lpips=0.581)
# B2 V2 ep1 (project memory: clip=0.6731, lpips=0.2781) — 路径 B 验证用
_DEFAULT_CKPT = ROOT / "exp" / "p3_remote_10h" / "e4_long_10ep" / "checkpoints" / "epoch_0005.pt"
_DEFAULT_CONFIG = ROOT / "exp" / "p3_remote_10h" / "e4_long_10ep" / "config.json"
CKPT_PATH = Path(os.environ.get("P4_CKPT_PATH", str(_DEFAULT_CKPT)))
CONFIG_PATH = Path(os.environ.get("P4_CONFIG_PATH", str(_DEFAULT_CONFIG)))
# v5: baseline 参考指标也可通过环境变量覆盖（B2 V2 用 0.6731/0.2781）
BASELINE_CLIP = float(os.environ.get("P4_BASELINE_CLIP", "0.727"))
BASELINE_LPIPS = float(os.environ.get("P4_BASELINE_LPIPS", "0.581"))

OUTPUT_DIR = ROOT / "exp" / "p4_fusion_breakout" / "infer_ablation"

# ModelConfig fields that are NOT declared as dataclass fields — must setattr
# AND mirror into the `extra` dict so they survive to_dict() / re-serialization.
EXTRA_FIELDS = (
    "style_extrap_alpha",
    "patch_adain_kernel",
    "multiband_adain_mode",
    "mid_adain_scale",
    "hh_adain_scale",
    # N5: 多级 style_fiber 放大参数 (推理侧, 通过环境变量注入)
    "style_extrap_levels",
    "style_extrap_hh_gain",
    "style_extrap_mid_gain",
)


def _parse_args(argv: list[str]) -> dict:
    if len(argv) < 3:
        print(
            "Usage: python _p4_infer_ablation.py <exp_name> <lowpass_mode> "
            "[style_extrap_alpha] [patch_adain_kernel] [multiband_adain_mode] "
            "[tri_band_inference_lock] [mid_adain_scale] [hh_adain_scale]"
        )
        sys.exit(2)

    return {
        "exp_name": argv[1],
        "lowpass_mode": argv[2],
        "style_extrap_alpha": float(argv[3]) if len(argv) > 3 else 0.0,
        "patch_adain_kernel": int(argv[4]) if len(argv) > 4 else 0,
        "multiband_adain_mode": argv[5] if len(argv) > 5 else "single",
        "tri_band_inference_lock": bool(int(argv[6])) if len(argv) > 6 else False,
        "mid_adain_scale": float(argv[7]) if len(argv) > 7 else 0.3,
        "hh_adain_scale": float(argv[8]) if len(argv) > 8 else 0.3,
    }


def _fix_windows_paths(config) -> None:
    # Config uses /mnt/i/ WSL paths; runtime is Windows native.
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


def _override_model_config(config, raw: dict, params: dict) -> None:
    """Apply ablation overrides to dataclass config, the `extra` dict, and the raw payload."""
    overrides = {
        "lowpass_mode": params["lowpass_mode"],
        "style_extrap_alpha": params["style_extrap_alpha"],
        "patch_adain_kernel": params["patch_adain_kernel"],
        "multiband_adain_mode": params["multiband_adain_mode"],
        "tri_band_inference_lock": params["tri_band_inference_lock"],
        "mid_adain_scale": params["mid_adain_scale"],
        "hh_adain_scale": params["hh_adain_scale"],
    }

    model_cfg = config.model
    raw_model = raw.setdefault("model", {})

    for key, value in overrides.items():
        # setattr works for both declared fields and ad-hoc attributes.
        setattr(model_cfg, key, value)
        # For non-declared fields, also mirror into `extra` so they survive
        # to_dict() / re-serialization (ModelConfig.to_dict merges `extra` back).
        if key in EXTRA_FIELDS:
            model_cfg.extra[key] = value
        # Also write into the raw payload in case the eval pipeline reloads
        # from raw (e.g. _run_full_eval_for_checkpoint re-reading config).
        raw_model[key] = value


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


def main() -> None:
    params = _parse_args(sys.argv)
    exp_name = params["exp_name"]

    print(f"[p4_infer_ablation] exp_name={exp_name}")
    print(f"[p4_infer_ablation] params={params}")
    print(f"[p4_infer_ablation] checkpoint={CKPT_PATH}")
    print(f"[p4_infer_ablation] config={CONFIG_PATH}")

    if not CKPT_PATH.exists():
        print(f"[p4_infer_ablation] ERROR: checkpoint not found: {CKPT_PATH}")
        sys.exit(1)
    if not CONFIG_PATH.exists():
        print(f"[p4_infer_ablation] ERROR: config not found: {CONFIG_PATH}")
        sys.exit(1)

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

    # Apply ablation overrides (config.model + extra dict + raw model section)
    _override_model_config(config, raw, params)
    # N5: 多级 style_fiber 放大参数也需要 setattr 到 config.model (环境变量注入)
    n5_levels_env = int(os.environ.get("P4_STYLE_EXTRAP_LEVELS", "1"))
    n5_hh_gain_env = float(os.environ.get("P4_STYLE_EXTRAP_HH_GAIN", "1.5"))
    n5_mid_gain_env = float(os.environ.get("P4_STYLE_EXTRAP_MID_GAIN", "1.0"))
    if n5_levels_env > 1:
        setattr(config.model, "style_extrap_levels", n5_levels_env)
        setattr(config.model, "style_extrap_hh_gain", n5_hh_gain_env)
        setattr(config.model, "style_extrap_mid_gain", n5_mid_gain_env)
        config.model.extra["style_extrap_levels"] = n5_levels_env
        config.model.extra["style_extrap_hh_gain"] = n5_hh_gain_env
        config.model.extra["style_extrap_mid_gain"] = n5_mid_gain_env
    print(f"[p4_infer_ablation] model.lowpass_mode={config.model.lowpass_mode}")
    print(f"[p4_infer_ablation] model.style_extrap_alpha={getattr(config.model, 'style_extrap_alpha', '?')}")
    print(f"[p4_infer_ablation] model.patch_adain_kernel={getattr(config.model, 'patch_adain_kernel', '?')}")
    print(f"[p4_infer_ablation] model.multiband_adain_mode={getattr(config.model, 'multiband_adain_mode', '?')}")
    print(f"[p4_infer_ablation] model.tri_band_inference_lock={config.model.tri_band_inference_lock}")
    print(f"[p4_infer_ablation] model.mid_adain_scale={getattr(config.model, 'mid_adain_scale', '?')}")
    print(f"[p4_infer_ablation] model.hh_adain_scale={getattr(config.model, 'hh_adain_scale', '?')}")

    _fix_windows_paths(config)

    # P4 修复: 每组消融用独立 eval output subdir, 避免 summary.json 被覆盖
    # (所有消融共用同一 checkpoint epoch_0005.pt)
    config.training.full_eval_output_subdir = f"full_eval_p4_{exp_name}"
    print(f"[p4_infer_ablation] training.full_eval_output_subdir={config.training.full_eval_output_subdir}")

    # P4 关键修复: 把修改后的 model+bridge 配置保存到临时 JSON, 通过 config_override 传给 run_evaluation.py
    # 原因: _run_full_eval_for_checkpoint 通过子进程/in-process 调用 run_evaluation.py,
    # 但只传 checkpoint 路径, run_evaluation.py 从 checkpoint 加载配置, setattr 修改丢失.
    # 解决: 用 --config_override 让 run_evaluation.py 合并修改后的配置.
    import tempfile
    override_payload = {
        "model": dict(raw.get("model", {})),
        "bridge": dict(raw.get("bridge", {})),
    }
    # 应用消融覆盖到 override payload
    overrides_map = {
        "lowpass_mode": params["lowpass_mode"],
        "style_extrap_alpha": params["style_extrap_alpha"],
        "patch_adain_kernel": params["patch_adain_kernel"],
        "multiband_adain_mode": params["multiband_adain_mode"],
        "tri_band_inference_lock": params["tri_band_inference_lock"],
        "mid_adain_scale": params["mid_adain_scale"],
        "hh_adain_scale": params["hh_adain_scale"],
    }
    for k, v in overrides_map.items():
        override_payload["model"][k] = v
    # P4 修复 A: 含 style_extrap_alpha > 0 的 U 类消融组必须同时设置 endpoint_adain_scale=1.0
    # 原因: model620.py L763 中 style_extrap_alpha 嵌套在 endpoint_adain_scale > 0.0 guard 内,
    # E4-long config 默认 endpoint_adain_scale=0.0, 会导致 style_extrap_alpha 被跳过.
    if params["style_extrap_alpha"] > 0:
        override_payload["model"]["endpoint_adain_scale"] = 1.0
    # N5: 多级 style_fiber 放大参数 (通过环境变量注入, 默认 1=单级=原始行为)
    n5_levels = int(os.environ.get("P4_STYLE_EXTRAP_LEVELS", "1"))
    n5_hh_gain = float(os.environ.get("P4_STYLE_EXTRAP_HH_GAIN", "1.5"))
    n5_mid_gain = float(os.environ.get("P4_STYLE_EXTRAP_MID_GAIN", "1.0"))
    if n5_levels > 1:
        override_payload["model"]["style_extrap_levels"] = n5_levels
        override_payload["model"]["style_extrap_hh_gain"] = n5_hh_gain
        override_payload["model"]["style_extrap_mid_gain"] = n5_mid_gain
        print(f"[p4_infer_ablation] N5 multi-level: levels={n5_levels} hh_gain={n5_hh_gain} mid_gain={n5_mid_gain}")
    override_path = OUTPUT_DIR / f"{exp_name}_override.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with override_path.open("w", encoding="utf-8") as f:
        json.dump(override_payload, f, indent=2, ensure_ascii=False)
    config.training.full_eval_config_override = str(override_path)
    print(f"[p4_infer_ablation] config_override={override_path}")

    output_path = OUTPUT_DIR / f"{exp_name}.json"

    print(f"[p4_infer_ablation] invoking full eval...")
    eval_result = run_module._run_full_eval_for_checkpoint(config, CKPT_PATH)
    print(f"[p4_infer_ablation] eval done. convergence_payload={eval_result}")

    metrics = _extract_metrics(CKPT_PATH, config)

    record = {
        "exp_name": exp_name,
        "checkpoint": str(CKPT_PATH),
        "config_path": str(CONFIG_PATH),
        "params": params,
        "baseline_reference": {"clip": BASELINE_CLIP, "lpips": BASELINE_LPIPS},
        "metrics": metrics,
        "convergence_payload": eval_result,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[p4_infer_ablation] wrote {output_path}")

    print(
        f"[p4_infer_ablation] {exp_name} | "
        f"clip_style={metrics['transfer_clip_style']} "
        f"lpips={metrics['transfer_content_lpips']} "
        f"wfi={metrics['wfi_score']}"
    )


if __name__ == "__main__":
    main()
