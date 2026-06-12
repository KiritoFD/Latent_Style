from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
SB_SRC = SB_ROOT / "src"

import sys

if str(SB_SRC) not in sys.path:
    sys.path.insert(0, str(SB_SRC))

from config_schema import load_experiment_config
from style_families import (
    is_true_i2sb_training_contract,
    resolves_exact_brownian_schedule,
    runtime_conditioning_requires_dino,
)


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _row_for_config(path: Path) -> dict[str, Any]:
    cfg = load_experiment_config(path)
    model = cfg.model
    bridge = cfg.bridge
    training = cfg.training

    tokenizer_family = str(getattr(model, "tokenizer_family", "legacy_factorized"))
    solver_family = str(getattr(model, "solver_family", "euler_legacy"))
    transport_prediction_mode = str(getattr(model, "transport_prediction_mode", "velocity"))
    objective_mode = str(getattr(bridge, "objective_mode", ""))
    loss_type = str(getattr(bridge, "loss_type", ""))
    bridge_noise_schedule = str(getattr(bridge, "bridge_noise_schedule", "auto"))
    semantic_supervision_family = str(getattr(bridge, "semantic_supervision_family", "legacy_terminal_swd"))
    style_tokenizer = str(getattr(model, "style_tokenizer", ""))
    tokenizer_num_clusters = int(getattr(model, "tokenizer_num_clusters", 16))
    style_spatial_mode = str(getattr(model, "style_spatial_mode", ""))
    tokenizer_content_adaptive = bool(getattr(model, "tokenizer_content_adaptive", False))
    use_diffeomorphic_stroke = bool(getattr(model, "use_diffeomorphic_stroke", False))
    style_injection_mode = str(getattr(model, "style_injection_mode", ""))
    dino_masked_swd_weight = float(getattr(bridge, "dino_masked_swd_weight", 0.0))
    dino_runtime_required = runtime_conditioning_requires_dino(
        tokenizer_family=tokenizer_family,
        semantic_supervision_family=semantic_supervision_family,
    )
    resume_model_strict = bool(getattr(training, "resume_model_strict", True))

    pure_latent_contract = (
        tokenizer_family == "pure_latent_spatial"
        and style_tokenizer == "null"
        and tokenizer_num_clusters == 32
        and style_spatial_mode == "disabled"
        and (not tokenizer_content_adaptive)
        and semantic_supervision_family == "legacy_terminal_swd"
        and dino_masked_swd_weight == 0.0
        and (not dino_runtime_required)
    )
    resolved_exact_brownian_schedule = resolves_exact_brownian_schedule(
        bridge_noise_schedule=bridge_noise_schedule,
        objective_mode=objective_mode,
    )
    true_i2sb_contract = is_true_i2sb_training_contract(
        solver_family=solver_family,
        transport_prediction_mode=transport_prediction_mode,
        objective_mode=objective_mode,
        loss_type=loss_type,
        bridge_noise_schedule=bridge_noise_schedule,
    )
    pure_round2_mainline = (
        pure_latent_contract
        and style_injection_mode == "none"
        and (not use_diffeomorphic_stroke)
    )

    return {
        "config_path": str(path),
        "tokenizer_family": tokenizer_family,
        "solver_family": solver_family,
        "transport_prediction_mode": transport_prediction_mode,
        "objective_mode": objective_mode,
        "loss_type": loss_type,
        "bridge_noise_schedule": bridge_noise_schedule,
        "resolved_exact_brownian_schedule": resolved_exact_brownian_schedule,
        "semantic_supervision_family": semantic_supervision_family,
        "style_tokenizer": style_tokenizer,
        "tokenizer_num_clusters": tokenizer_num_clusters,
        "style_spatial_mode": style_spatial_mode,
        "tokenizer_content_adaptive": tokenizer_content_adaptive,
        "style_injection_mode": style_injection_mode,
        "use_diffeomorphic_stroke": use_diffeomorphic_stroke,
        "dino_masked_swd_weight": dino_masked_swd_weight,
        "dino_runtime_required": dino_runtime_required,
        "resume_model_strict": resume_model_strict,
        "pure_latent_contract": pure_latent_contract,
        "true_i2sb_contract": true_i2sb_contract,
        "pure_round2_mainline": pure_round2_mainline,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit round-2 config files for pure-latent / true-I2SB contract compliance.")
    parser.add_argument(
        "--config-root",
        default=str(SB_ROOT / "configs" / "aaai2027" / "round2_pure_sde"),
        help="Root directory containing round-2 json configs.",
    )
    parser.add_argument(
        "--json-out",
        default=str(SB_ROOT / "docs" / "experiments" / "round2_pure_sde" / "contract_audit.json"),
        help="Optional audit json output path.",
    )
    args = parser.parse_args()

    config_root = Path(args.config_root).expanduser()
    if not config_root.is_absolute():
        config_root = (WORKSPACE / config_root).resolve()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in sorted(config_root.rglob("*.json")):
        try:
            row = _row_for_config(path)
            rows.append(row)
        except Exception as exc:
            failures.append({"config_path": str(path), "error": repr(exc)})

    summary = {
        "config_root": str(config_root),
        "row_count": len(rows),
        "failure_count": len(failures),
        "rows": rows,
        "failures": failures,
        "pure_latent_contract_count": sum(1 for row in rows if bool(row["pure_latent_contract"])),
        "exact_brownian_schedule_count": sum(1 for row in rows if bool(row["resolved_exact_brownian_schedule"])),
        "true_i2sb_contract_count": sum(1 for row in rows if bool(row["true_i2sb_contract"])),
        "pure_round2_mainline_count": sum(1 for row in rows if bool(row["pure_round2_mainline"])),
    }

    json_out = str(args.json_out).strip()
    if json_out:
        out_path = Path(json_out).expanduser()
        if not out_path.is_absolute():
            out_path = (WORKSPACE / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(out_path)
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
