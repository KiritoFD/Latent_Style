from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import BridgeConfig, ExperimentConfig, ModelConfig, TrainingConfig  # noqa: E402
from losses import OTFlowMatchingObjective  # noqa: E402
from model import TimeConditionedLANCETBridge  # noqa: E402
from ot_cost import SWDTransportCost  # noqa: E402
from trainer import SBTrainer  # noqa: E402


def test_cpu_hungarian_requires_explicit_opt_in() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(),
        bridge=BridgeConfig(coupling_solver="hungarian", allow_cpu_hungarian=False),
    )

    with pytest.raises(ValueError, match="offloads OT matching to CPU"):
        OTFlowMatchingObjective(cfg)


def test_unbalanced_ot_can_route_to_source_local_dummies() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(),
        bridge=BridgeConfig(
            coupling_solver="sinkhorn_unbalanced",
            sinkhorn_unbalanced_dummy_cost=0.25,
            sinkhorn_unbalanced_dummy_offdiag_cost=9.0,
        ),
    )
    objective = OTFlowMatchingObjective(cfg)
    cost = torch.ones(3, 2)
    content = torch.randn(3, 4, 4, 4)
    target = torch.randn(2, 4, 4, 4)

    augmented_cost, augmented_targets, real_count = objective._augment_cost_with_source_dummies(
        cost,
        content,
        target,
    )

    assert real_count == 2
    assert augmented_cost.shape == (3, 5)
    assert augmented_targets.shape[0] == 5
    assert torch.allclose(torch.diagonal(augmented_cost[:, 2:]), torch.full((3,), 0.25))
    assert torch.equal(augmented_targets[2:], content)


def test_barycentric_projection_uses_source_row_shape() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(),
        bridge=BridgeConfig(coupling_target_mode="barycentric_full"),
    )
    objective = OTFlowMatchingObjective(cfg)
    plan = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]], dtype=torch.float32)
    target = torch.randn(3, 4, 5, 5)

    matched, _, _ = objective._sample_or_project_from_plan(plan, target)

    assert matched.shape == (2, 4, 5, 5)


def test_phase616_contract_rejects_proximal_trust() -> None:
    payload = {
        "model": {"contract_family": "phase616"},
        "bridge": {"proximal_trust_ratio": 0.5, "proximal_trust_weight": 0.5},
    }

    with pytest.raises(ValueError, match="proximal_trust"):
        ExperimentConfig.from_mapping(payload)


def test_model_config_drops_retired_legacy_style_spatial_keys() -> None:
    cfg = ModelConfig.from_mapping(
        {
            "style_spatial_pre_gain_16": 0.9,
            "style_spatial_mode": "vq",
            "style_spatial_num_prototypes": 8,
            "style_spatial_routing_temperature": 0.5,
            "style_spatial_content_hidden_dim": 48,
            "style_id_spatial_jitter_px": 2,
            "ablation_disable_spatial_prior": True,
            "custom_probe_flag": 7,
        }
    )

    payload = cfg.to_dict()
    for key in (
        "style_spatial_pre_gain_16",
        "style_spatial_mode",
        "style_spatial_num_prototypes",
        "style_spatial_routing_temperature",
        "style_spatial_content_hidden_dim",
        "style_id_spatial_jitter_px",
        "ablation_disable_spatial_prior",
    ):
        assert key not in payload
        assert not hasattr(cfg, key)
    assert payload["custom_probe_flag"] == 7


def test_swd_projection_cache_keys_include_spatial_shape() -> None:
    cfg = BridgeConfig(swd_patch_sizes=[3], swd_num_projections=4)
    cost = SWDTransportCost(cfg)

    cost._get_projection_bank(4, device=torch.device("cpu"), spatial_hw=(16, 16))
    cost._get_projection_bank(4, device=torch.device("cpu"), spatial_hw=(32, 32))

    keys = list(cost._projection_cache.keys())
    assert len(keys) == 2
    assert {key[-1] for key in keys} == {(16, 16), (32, 32)}


def test_channels_last_and_compile_are_mutually_exclusive() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(),
        bridge=BridgeConfig(),
        training=TrainingConfig(channels_last=True, torch_compile=True),
    )

    if not torch.cuda.is_available():
        # The trainer only activates channels_last on CUDA, so assert the guard
        # remains present without constructing the full model on CPU.
        assert any(
            isinstance(value, str) and "mutually exclusive" in value
            for value in SBTrainer.__init__.__code__.co_consts
        )
        return

    with pytest.raises(ValueError, match="mutually exclusive"):
        SBTrainer(cfg, torch.device("cuda"))


def test_solver_corrector_mode_is_explicit_not_fallback() -> None:
    assert TimeConditionedLANCETBridge._normalize_solver_corrector_mode("lowpass") == "lowpass_source_anchor"
    with pytest.raises(ValueError, match="Unsupported model.solver_corrector_mode"):
        TimeConditionedLANCETBridge._normalize_solver_corrector_mode("typo_lowpass")


def test_lowpass_corrector_is_opt_in() -> None:
    bridge = TimeConditionedLANCETBridge.__new__(TimeConditionedLANCETBridge)
    bridge.solver_corrector_steps = 1
    bridge.solver_corrector_step_size = 1.0
    bridge.solver_corrector_lowpass_kernel = 3
    bridge.solver_corrector_clamp = 0.0

    h = torch.ones(1, 1, 4, 4)
    source = torch.zeros_like(h)

    bridge.solver_corrector_mode = "none"
    assert torch.equal(bridge._correct_transport_state(h, source, dt=1.0), h)

    bridge.solver_corrector_mode = "lowpass_source_anchor"
    corrected = bridge._correct_transport_state(h, source, dt=1.0)
    assert corrected.abs().mean().item() < h.abs().mean().item()


def test_style_delta_basis_is_zero_init_compatible() -> None:
    bridge = TimeConditionedLANCETBridge.__new__(TimeConditionedLANCETBridge)
    nn.Module.__init__(bridge)
    bridge.style_delta_mode = "basis"
    bridge.style_delta_scale = 0.25
    bridge.style_delta_rank = 2
    bridge.latent_channels = 1
    bridge.style_delta_force_highpass = False
    bridge.style_delta_highpass_kernel = 3
    bridge.style_delta_basis_proj = nn.Conv2d(3, 2, kernel_size=1)
    bridge.style_delta_weight_head = nn.Linear(4, 2)
    bridge.last_style_delta_debug = {}
    nn.init.ones_(bridge.style_delta_basis_proj.weight)
    nn.init.zeros_(bridge.style_delta_basis_proj.bias)
    nn.init.zeros_(bridge.style_delta_weight_head.weight)
    nn.init.zeros_(bridge.style_delta_weight_head.bias)

    delta = torch.zeros(1, 1, 4, 4)
    h = torch.ones(1, 3, 4, 4)
    style_code = torch.ones(1, 4)

    out = bridge._apply_style_delta_basis(delta, h, style_code)
    assert torch.equal(out, delta)
    assert bridge.last_style_delta_debug["style_delta_basis_active"] == 1.0
    assert bridge.last_style_delta_debug["style_delta_side_abs"] == 0.0

    with torch.no_grad():
        bridge.style_delta_weight_head.bias.fill_(0.5)
    moved = bridge._apply_style_delta_basis(delta, h, style_code)
    assert moved.abs().mean().item() > 0.0


def test_crossattn_texture_legacy_factorized_has_no_legacy_style_spatial_path() -> None:
    bridge = TimeConditionedLANCETBridge(
        ModelConfig(
            num_styles=3,
            style_dim=32,
            base_dim=16,
            time_dim=32,
            tokenizer_identity_dim=8,
            tokenizer_texture_dim=8,
            tokenizer_geometry_dim=8,
            num_hires_blocks=1,
            num_res_blocks=1,
            num_decoder_blocks=1,
            style_attn_num_tokens=8,
            style_attn_num_heads=2,
            hires_block_type="conv",
            body_block_type="conv",
            decoder_block_type="conv",
            proximal_mode="crossattn_texture",
            proximal_hidden_channels=8,
        )
    )

    legacy_param_names = [name for name, _ in bridge.named_parameters() if name.startswith("style_spatial")]
    assert legacy_param_names == []
    assert not hasattr(bridge, "encode_style_spatial_id")

    z_base = torch.randn(2, 4, 16, 16)
    output = bridge.refine_endpoint(
        z_base,
        style_id=torch.tensor([0, 1]),
        source_latent=z_base,
    )

    assert output.shape == z_base.shape
    assert torch.allclose(output, z_base)
