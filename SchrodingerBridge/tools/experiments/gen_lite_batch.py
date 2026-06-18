from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("SB_ROOT", "/mnt/i/Github/Latent_Style/SchrodingerBridge"))
BATCH = os.environ.get("PHASE618_BATCH", "exp/20250618_lite_ot_vertical")
BASE_CFG = Path(
    os.environ.get(
        "PHASE618_BASE_CFG",
        str(ROOT / "docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/remote_base_phase618_ot_rerun_lowrank.json"),
    )
)
ALLOW_LEGACY_BASE = str(os.environ.get("PHASE618_ALLOW_LEGACY", "") or "").strip().lower() in {"1", "true", "yes"}


EXPS: dict[str, dict[str, Any]] = {
    "h0_vertical_fm": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_cost_composition": "structure_only",
        "bridge.coupling_structure_cost_mode": "self_affinity_gw",
        "bridge.bridge_sigma": 0.0,
    },
    "h1_linear_fm": {
        "bridge.bridge_path_mode": "linear",
        "bridge.coupling_cost_composition": "structure_only",
        "bridge.coupling_structure_cost_mode": "self_affinity_gw",
        "bridge.bridge_sigma": 0.0,
    },
    "h2_euclidean_ot": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_cost_composition": "appearance_only",
        "bridge.bridge_sigma": 0.0,
    },
    "h3_sde_noise": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_cost_composition": "structure_only",
        "bridge.coupling_structure_cost_mode": "self_affinity_gw",
        "bridge.bridge_sigma": 0.02,
        "bridge.bridge_noise_schedule": "exact_brownian",
    },
    "h4_unbalanced_ot": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_cost_composition": "structure_only",
        "bridge.coupling_structure_cost_mode": "self_affinity_gw",
        "bridge.coupling_solver": "sinkhorn_unbalanced",
        "bridge.sinkhorn_unbalanced_tau_src": 0.5,
        "bridge.bridge_sigma": 0.0,
    },
    "h5_topogate_attention": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_cost_composition": "appearance_plus_structure",
        "bridge.coupling_structure_cost_mode": "topogate_attention_gw",
        "bridge.coupling_structure_cost_weight": 0.4,
        "bridge.bridge_sigma": 0.0,
    },
    "h6_combined_topogate": {
        "bridge.bridge_path_mode": "vertical",
        "bridge.coupling_solver": "sinkhorn_unbalanced",
        "bridge.sinkhorn_unbalanced_tau_src": 0.5,
        "bridge.coupling_cost_composition": "appearance_plus_structure",
        "bridge.coupling_structure_cost_mode": "topogate_attention_gw",
        "bridge.coupling_structure_cost_weight": 0.4,
        "bridge.bridge_sigma": 0.02,
        "bridge.bridge_noise_schedule": "exact_brownian",
    },
}


def _set_nested(payload: dict[str, Any], dotted: str, value: Any) -> None:
    target = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_phase618_base(cfg: dict[str, Any], *, allow_legacy: bool = False) -> None:
    if allow_legacy:
        return
    model = dict(cfg.get("model") or {})
    issues: list[str] = []
    tokenizer_family = str(model.get("tokenizer_family", "") or "").strip().lower()
    conditioning_mode = str(model.get("matched_target_conditioning_mode", "") or "").strip().lower()
    encoder_mode = str(model.get("matched_target_style_encoder_mode", "") or "").strip().lower()
    spatial_mode = str(model.get("style_code_spatial_mode", "") or "").strip().lower()
    try:
        spatial_scale = float(model.get("style_code_spatial_scale", 0.0) or 0.0)
    except (TypeError, ValueError):
        spatial_scale = 0.0

    if tokenizer_family != "pure_latent_spatial":
        issues.append(f"model.tokenizer_family={model.get('tokenizer_family')!r}")
    if conditioning_mode != "both":
        issues.append(f"model.matched_target_conditioning_mode={model.get('matched_target_conditioning_mode')!r}")
    if encoder_mode != "residual":
        issues.append(f"model.matched_target_style_encoder_mode={model.get('matched_target_style_encoder_mode')!r}")
    if spatial_mode != "lowrank":
        issues.append(f"model.style_code_spatial_mode={model.get('style_code_spatial_mode')!r}")
    if spatial_scale <= 0.0:
        issues.append(f"model.style_code_spatial_scale={model.get('style_code_spatial_scale')!r}")

    if issues:
        raise ValueError(
            "gen_lite_batch.py now defaults to the repaired phase618 lowrank carrier base. "
            "Refusing to silently regenerate old legacy-family runs.\n- " + "\n- ".join(issues)
        )


def build_run_config(base_cfg: dict[str, Any], *, save_dir: str, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    model = cfg.setdefault("model", {})
    data = cfg.setdefault("data", {})
    training = cfg.setdefault("training", {})
    checkpoint = cfg.setdefault("checkpoint", {})

    # Preserve an explicit base family. Only fill compatibility defaults if missing.
    model.setdefault("tokenizer_family", "legacy_factorized")
    model.setdefault("style_tokenizer", "factorized")
    model.setdefault("semantic_self_topology_gate", True)
    model.setdefault("semantic_self_topology_blend", 1.0)

    data["pairing_cache_path"] = ""
    data["virtual_length_multiplier"] = 0.1

    training["resume_checkpoint"] = ""
    training["resume_optimizer"] = False
    training["resume_training_state"] = False
    training["resume_prefer_local_checkpoint"] = False
    training["num_epochs"] = 60
    training["save_interval"] = 1
    training["batch_size"] = 20
    training["virtual_length_multiplier"] = 1.0
    training["full_eval_each_epoch"] = True
    training["full_eval_defer_until_training_end"] = False
    training["full_eval_only_lpips_clip_style"] = True
    training["full_eval_transfer_only"] = True
    training["full_eval_stop_on_convergence"] = True
    training["full_eval_convergence_patience"] = 4
    training["full_eval_convergence_min_epochs"] = 4
    training["full_eval_output_subdir"] = "full_eval_transfer"
    checkpoint["save_dir"] = save_dir

    for key, value in overrides.items():
        _set_nested(cfg, key, value)
    return cfg


def main() -> int:
    stage_root = ROOT / BATCH
    stage_root.mkdir(parents=True, exist_ok=True)
    base_cfg = _load_json(BASE_CFG)
    validate_phase618_base(base_cfg, allow_legacy=ALLOW_LEGACY_BASE)

    print(f"root={ROOT}")
    print(f"batch={BATCH}")
    print(f"base_cfg={BASE_CFG}")
    print(f"allow_legacy={ALLOW_LEGACY_BASE}")

    for name, overrides in EXPS.items():
        run_dir = stage_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_cfg = build_run_config(base_cfg, save_dir=str(run_dir), overrides=overrides)
        (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  {name}")

    print(f"\nDone. {len(EXPS)} experiments in {BATCH}")
    print(
        "preserved_base_family="
        f"{((base_cfg.get('model') or {}).get('tokenizer_family', 'missing'))}, "
        "train_from_scratch, eval_each_epoch"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
