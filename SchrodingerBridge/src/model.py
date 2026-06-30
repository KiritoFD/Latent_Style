"""Model factory for FC-SB Phase 4 (620_spectral_ode active contract).

630 Phase 1C cleanup: removed TimeConditionedLANCETBridge (~2070 lines) and all
legacy imports (lancet_blocks, lancet_backbone, style_families, utils.diffeomorphic).
Only the active 620_spectral_ode / 620_spatial_bridge contracts are supported.
Legacy contracts raise ValueError with a clear migration message.
"""
from __future__ import annotations

import logging
from typing import Mapping

import torch.nn as nn

from config_schema import BridgeConfig, ModelConfig

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters (migrated from lancet_backbone.py)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _normalize_skip_routing_mode(config: ModelConfig) -> ModelConfig:
    """Normalize skip_routing_mode (retained for config compatibility)."""
    model_cfg = config.validated()
    skip_mode = str(model_cfg.skip_routing_mode).strip().lower()
    if skip_mode not in {"none", "naive", "adaptive", "normalized"}:
        if bool(model_cfg.extra.get("skip_frequency_gated", True)):
            skip_mode = "normalized"
        else:
            skip_mode = "naive"
    model_cfg.skip_routing_mode = skip_mode
    return model_cfg


def _attach_bridge_runtime_fields(
    model_cfg: ModelConfig,
    bridge_cfg: BridgeConfig | Mapping[str, object] | None,
) -> ModelConfig:
    """Attach bridge config fields to model config (retained for config compatibility)."""
    if bridge_cfg is None:
        return model_cfg
    bridge = bridge_cfg if isinstance(bridge_cfg, BridgeConfig) else BridgeConfig.from_mapping(bridge_cfg)
    bridge_fields = {
        "objective_mode": str(getattr(bridge, "objective_mode", "")),
        "loss_type": str(getattr(bridge, "loss_type", "")),
        "bridge_sigma": float(getattr(bridge, "bridge_sigma", 0.0)),
        "bridge_noise_schedule": str(getattr(bridge, "bridge_noise_schedule", "auto")),
        "i2sb_predictor_time_floor": float(getattr(bridge, "i2sb_predictor_time_floor", 0.0)),
        "i2sb_noise_family": str(getattr(bridge, "i2sb_noise_family", "gaussian")),
        "i2sb_style_noise_amplitude_power": float(getattr(bridge, "i2sb_style_noise_amplitude_power", 1.0)),
    }
    model_cfg.extra = dict(getattr(model_cfg, "extra", {}) or {})
    for key, value in bridge_fields.items():
        setattr(model_cfg, key, value)
        model_cfg.extra[key] = value
    return model_cfg


def build_model_from_config(
    model_cfg: ModelConfig | Mapping[str, object],
    *,
    bridge_cfg: BridgeConfig | Mapping[str, object] | None = None,
    use_checkpointing: bool = False,
) -> nn.Module:
    """Build model from config. Supports 620_spectral_ode (active) and 620_spatial_bridge.

    Legacy contracts (TimeConditionedLANCETBridge) were removed in 630 Phase 1C.
    """
    config = model_cfg if isinstance(model_cfg, ModelConfig) else ModelConfig.from_mapping(model_cfg)
    config = _attach_bridge_runtime_fields(config, bridge_cfg)
    config = _normalize_skip_routing_mode(config)
    config.use_checkpointing = bool(use_checkpointing)
    family = str(getattr(config, "contract_family", "legacy") or "legacy").strip().lower()
    if family == "620_spatial_bridge":
        from model620 import build_spatial_bridge620_from_config
        return build_spatial_bridge620_from_config(config, bridge_cfg=bridge_cfg, use_checkpointing=use_checkpointing)
    elif family == "620_spectral_ode":
        from spectral_bridge620 import build_spectral_ode_bridge_from_config
        return build_spectral_ode_bridge_from_config(config, bridge_cfg=bridge_cfg)
    raise ValueError(
        f"contract_family={family!r} is no longer supported. "
        "TimeConditionedLANCETBridge was removed in 630 Phase 1C. "
        "Use '620_spectral_ode' (active) or '620_spatial_bridge'."
    )


__all__ = [
    "build_model_from_config",
    "count_parameters",
]
