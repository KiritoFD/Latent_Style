from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PHASE616_AUTO_PATH = ROOT / "tools" / "experiments" / "phase616_auto.py"
PHASE616_AUTO_SPEC = importlib.util.spec_from_file_location("phase616_auto", PHASE616_AUTO_PATH)
assert PHASE616_AUTO_SPEC is not None and PHASE616_AUTO_SPEC.loader is not None
PHASE616_AUTO = importlib.util.module_from_spec(PHASE616_AUTO_SPEC)
PHASE616_AUTO_SPEC.loader.exec_module(PHASE616_AUTO)

GEN_LITE_BATCH_PATH = ROOT / "tools" / "experiments" / "gen_lite_batch.py"
GEN_LITE_BATCH_SPEC = importlib.util.spec_from_file_location("gen_lite_batch", GEN_LITE_BATCH_PATH)
assert GEN_LITE_BATCH_SPEC is not None and GEN_LITE_BATCH_SPEC.loader is not None
GEN_LITE_BATCH = importlib.util.module_from_spec(GEN_LITE_BATCH_SPEC)
GEN_LITE_BATCH_SPEC.loader.exec_module(GEN_LITE_BATCH)

PROBE_CONDITIONING_PATH = ROOT / "tools" / "probe_conditioning_sensitivity.py"
PROBE_CONDITIONING_SPEC = importlib.util.spec_from_file_location(
    "probe_conditioning_sensitivity", PROBE_CONDITIONING_PATH
)
assert PROBE_CONDITIONING_SPEC is not None and PROBE_CONDITIONING_SPEC.loader is not None
PROBE_CONDITIONING = importlib.util.module_from_spec(PROBE_CONDITIONING_SPEC)
PROBE_CONDITIONING_SPEC.loader.exec_module(PROBE_CONDITIONING)

PROBE_CHECKPOINT_STYLE_RESPONSE_PATH = ROOT / "tools" / "probe_checkpoint_style_response.py"
PROBE_CHECKPOINT_STYLE_RESPONSE_SPEC = importlib.util.spec_from_file_location(
    "probe_checkpoint_style_response", PROBE_CHECKPOINT_STYLE_RESPONSE_PATH
)
assert PROBE_CHECKPOINT_STYLE_RESPONSE_SPEC is not None and PROBE_CHECKPOINT_STYLE_RESPONSE_SPEC.loader is not None
PROBE_CHECKPOINT_STYLE_RESPONSE = importlib.util.module_from_spec(PROBE_CHECKPOINT_STYLE_RESPONSE_SPEC)
PROBE_CHECKPOINT_STYLE_RESPONSE_SPEC.loader.exec_module(PROBE_CHECKPOINT_STYLE_RESPONSE)

from config_schema import BridgeConfig, ExperimentConfig, ModelConfig, TrainingConfig  # noqa: E402
from losses import OTFlowMatchingObjective  # noqa: E402
from model import TimeConditionedLANCETBridge, build_model_from_config  # noqa: E402
from ot_cost import SWDTransportCost  # noqa: E402
from trainer import SBTrainer  # noqa: E402
from run import _eval_convergence_requests_stop  # noqa: E402


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


def test_topogate_attention_gw_reuses_model_attention_not_tokenizer() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(
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
            body_block_type="global_attn",
            decoder_block_type="conv",
            semantic_self_topology_gate=True,
            semantic_self_topology_blend=1.0,
        ),
        bridge=BridgeConfig(
            coupling_cost_composition="appearance_plus_structure",
            coupling_structure_cost_mode="topogate_attention_gw",
            coupling_structure_cost_weight=0.4,
        ),
    )
    objective = OTFlowMatchingObjective(cfg)
    model = TimeConditionedLANCETBridge(cfg.model)
    content = torch.randn(2, 4, 16, 16)
    target = torch.randn(2, 4, 16, 16)

    cost, metrics = objective._structure_pairwise_cost(
        model,
        content,
        target,
        style_id=torch.tensor([0, 1]),
    )

    assert cost.shape == (2, 2)
    assert torch.isfinite(cost).all()
    assert metrics["ot_topogate_probe_active"].item() == 1.0
    assert metrics["ot_topogate_complexity_cost_mean"].item() >= 0.0
    assert metrics["ot_latent_affinity_cost_mean"].item() >= 0.0


def test_topogate_attention_gw_aggregates_all_body_blocks() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(
            num_styles=3,
            style_dim=32,
            base_dim=16,
            time_dim=32,
            tokenizer_identity_dim=8,
            tokenizer_texture_dim=8,
            tokenizer_geometry_dim=8,
            num_hires_blocks=1,
            num_res_blocks=4,
            num_decoder_blocks=1,
            style_attn_num_tokens=8,
            style_attn_num_heads=2,
            hires_block_type="conv",
            body_block_type="global_attn",
            decoder_block_type="conv",
            semantic_self_topology_gate=True,
            semantic_self_topology_blend=1.0,
        ),
        bridge=BridgeConfig(
            coupling_cost_composition="appearance_plus_structure",
            coupling_structure_cost_mode="topogate_attention_gw",
            coupling_structure_cost_weight=0.4,
        ),
    )
    objective = OTFlowMatchingObjective(cfg)
    model = TimeConditionedLANCETBridge(cfg.model)
    content = torch.randn(2, 4, 16, 16)
    target = torch.randn(2, 4, 16, 16)

    _, metrics = objective._structure_pairwise_cost(
        model,
        content,
        target,
        style_id=torch.tensor([0, 1]),
    )

    assert len(model.body_blocks) == 4
    assert metrics["ot_topogate_probe_active"].item() == 1.0
    assert metrics["ot_topogate_descriptor_blocks"].item() == pytest.approx(4.0)


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


def test_build_model_from_config_mirrors_bridge_runtime_noise_schedule() -> None:
    model = build_model_from_config(
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
        ),
        bridge_cfg=BridgeConfig(
            objective_mode="bridge_velocity",
            loss_type="mse",
            bridge_sigma=0.02,
            bridge_noise_schedule="exact_brownian",
            i2sb_predictor_time_floor=0.125,
            i2sb_noise_family="style_covariant",
            i2sb_style_noise_amplitude_power=1.7,
        ),
        use_checkpointing=False,
    )

    assert model.objective_mode == "bridge_velocity"
    assert model.loss_type == "mse"
    assert model.bridge_sigma == pytest.approx(0.02)
    assert model.bridge_noise_schedule == "exact_brownian"
    assert model.i2sb_predictor_time_floor == pytest.approx(0.125)
    assert model.i2sb_noise_family == "style_covariant"
    assert model.i2sb_style_noise_amplitude_power == pytest.approx(1.7)


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
    delta = (output - z_base).abs()
    # The legacy style-spatial path is gone, but tiny zero-init residuals in the
    # proximal branch can still produce a numerically non-zero near-identity map.
    assert delta.mean().item() < 5e-3
    assert delta.max().item() < 5e-2


@pytest.mark.parametrize(
    ("form", "site"),
    [
        ("mixed", "body"),
        ("carrier_gate", "body"),
        ("spatial_carrier_gate", "body"),
    ],
)
def test_style_injection_live_init_wakes_sleeping_branch(form: str, site: str) -> None:
    model_cfg = ModelConfig(
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
        style_injection_mode=site,
        style_injection_form=form,
        style_injection_scale=1.0,
        style_injection_live_init=False,
    )
    live_model_cfg = ModelConfig(**{**model_cfg.to_dict(), "style_injection_live_init": True, "style_injection_live_init_std": 0.02})

    bridge = TimeConditionedLANCETBridge(model_cfg)
    live_bridge = TimeConditionedLANCETBridge(live_model_cfg)
    feat = torch.randn(2, bridge.body_channels, 16, 16)
    x = torch.randn(2, bridge.latent_channels, 16, 16)
    style_code = torch.randn(2, bridge.bridge_style_dim)
    style_map = torch.randn(2, bridge.body_channels, 16, 16)

    out = bridge._apply_style_feature_injection(feat.clone(), x, style_code, site=site, style_map=style_map)
    live_out = live_bridge._apply_style_feature_injection(feat.clone(), x, style_code, site=site, style_map=style_map)

    assert torch.allclose(out, feat)
    assert (live_out - feat).abs().mean().item() > 0.0


def test_path_anatomy_probe_tracks_live_init_body_injection() -> None:
    def build_cfg(*, live_init: bool) -> ExperimentConfig:
        return ExperimentConfig(
            model=ModelConfig(
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
                style_injection_mode="body",
                style_injection_form="mixed",
                style_injection_scale=1.0,
                style_injection_live_init=live_init,
                style_injection_live_init_std=0.02,
            ),
            bridge=BridgeConfig(),
        )

    inputs = PROBE_CONDITIONING._random_inputs(
        batch_size=2,
        latent_channels=4,
        latent_size=16,
        style_id=1,
        seed=123,
        device=torch.device("cpu"),
    )
    dead_rows, dead_summary = PROBE_CONDITIONING._path_anatomy_rows(
        build_cfg(live_init=False),
        device=torch.device("cpu"),
        seed=0,
        checkpoint=None,
        inputs=inputs,
    )
    live_rows, live_summary = PROBE_CONDITIONING._path_anatomy_rows(
        build_cfg(live_init=True),
        device=torch.device("cpu"),
        seed=0,
        checkpoint=None,
        inputs=inputs,
    )

    dead_code = next(row for row in dead_rows if row["path_mode"] == "code_only_no_reference")
    live_code = next(row for row in live_rows if row["path_mode"] == "code_only_no_reference")

    assert dead_summary["anatomy_code_body_dead_spatial_body_live"] is True
    assert live_summary["anatomy_code_body_dead_spatial_body_live"] is False
    assert dead_code["h_body_a_vs_b_mean_abs"] == pytest.approx(0.0, abs=1e-12)
    assert live_code["h_body_a_vs_b_mean_abs"] > 0.0
    assert live_code["h_fused_a_vs_b_mean_abs"] > dead_code["h_fused_a_vs_b_mean_abs"]


def test_lowrank_path_anatomy_probe_tracks_live_runtime_deltas() -> None:
    cfg = PROBE_CONDITIONING.load_experiment_config(
        ROOT
        / "docs"
        / "experiments"
        / "2026-06-18-stage1-lowrank-rerun-audit"
        / "baseline_h1_lowrank_config.json"
    )
    inputs = PROBE_CONDITIONING._random_inputs(
        batch_size=2,
        latent_channels=int(cfg.model.latent_channels),
        latent_size=16,
        style_id=1,
        seed=123,
        device=torch.device("cpu"),
    )

    conditioning_rows, conditioning_summary = PROBE_CONDITIONING._conditioning_rows(
        cfg,
        device=torch.device("cpu"),
        seed=0,
        checkpoint=None,
        inputs=inputs,
    )
    anatomy_rows, _ = PROBE_CONDITIONING._path_anatomy_rows(
        cfg,
        device=torch.device("cpu"),
        seed=0,
        checkpoint=None,
        inputs=inputs,
    )

    code_row = next(row for row in anatomy_rows if row["path_mode"] == "code_only_no_reference")
    spatial_row = next(row for row in anatomy_rows if row["path_mode"] == "spatial_matched_target")

    assert code_row["style_map_a_vs_b_mean_abs"] > 0.0
    assert spatial_row["style_map_a_vs_b_mean_abs"] > 0.0
    assert code_row["delta_a_vs_b_mean_abs"] == pytest.approx(
        conditioning_summary["conditioning_code_forward_delta"],
        rel=1e-5,
        abs=1e-7,
    )
    assert spatial_row["delta_a_vs_b_mean_abs"] == pytest.approx(
        conditioning_summary["conditioning_spatial_forward_delta"],
        rel=1e-5,
        abs=1e-7,
    )


def test_eval_convergence_stop_respects_min_epoch() -> None:
    train_cfg = SimpleNamespace(
        full_eval_stop_on_convergence=True,
        full_eval_convergence_min_epochs=4,
    )
    payload = {"converged": True}

    assert _eval_convergence_requests_stop(train_cfg, payload, epoch=4) is True
    assert _eval_convergence_requests_stop(train_cfg, payload, epoch=3) is False
    assert _eval_convergence_requests_stop(train_cfg, {"converged": False}, epoch=10) is False


def test_eval_convergence_stop_accepts_objective_patience_signal() -> None:
    train_cfg = SimpleNamespace(
        full_eval_stop_on_convergence=True,
        full_eval_convergence_min_epochs=4,
    )
    payload = {
        "converged": False,
        "objective_patience_converged": True,
        "stop_ready": True,
        "stop_reason": "objective_gap_patience",
    }

    assert _eval_convergence_requests_stop(train_cfg, payload, epoch=4) is True
    assert _eval_convergence_requests_stop(train_cfg, payload, epoch=3) is False


def test_phase616_auto_patience_stop_prefers_convergence_stop_ready(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "full_eval_transfer"
    eval_dir.mkdir(parents=True)
    curve_csv = eval_dir / "clip_lpips_curve.csv"
    curve_csv.write_text(
        "epoch,epoch_int,transfer_clip_style,transfer_content_lpips\n"
        "epoch_0001,1,0.6700,0.3800\n"
        "epoch_0002,2,0.6620,0.3340\n"
        "epoch_0003,3,0.6610,0.3360\n"
        "epoch_0004,4,0.6605,0.3500\n"
        "epoch_0005,5,0.6617,0.3440\n"
        "epoch_0006,6,0.6611,0.3614\n",
        encoding="utf-8",
    )
    (eval_dir / "round2_convergence.json").write_text(
        json.dumps(
            {
                "row_count": 6,
                "best_epoch": "epoch_0001",
                "last_pareto_epoch": "epoch_0002",
                "converged": False,
                "objective_best_epoch": "epoch_0002",
                "objective_best_gap": 0.1121,
                "objective_epochs_since_best": 4,
                "objective_patience_converged": True,
                "stop_ready": True,
                "stop_reason": "objective_gap_patience",
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "training": {
            "full_eval_convergence_patience": 4,
            "full_eval_convergence_min_epochs": 4,
        }
    }

    should_stop, detail = PHASE616_AUTO._run_has_patience_proven_best(
        run_dir,
        eval_subdir="full_eval_transfer",
        cfg=cfg,
    )

    assert should_stop is True
    assert detail["reason"] == "objective_gap_patience"
    assert detail["best_epoch"] == 2
    assert detail["epochs_since_best"] == 4


def test_checkpoint_style_response_transition_labels() -> None:
    assert PROBE_CHECKPOINT_STYLE_RESPONSE._transition_label(0.03, 0.0009) == "trained_suppression"
    assert PROBE_CHECKPOINT_STYLE_RESPONSE._transition_label(0.0, 0.0) == "persistent_noop"
    assert PROBE_CHECKPOINT_STYLE_RESPONSE._transition_label(0.0, 0.004) == "trained_wakeup"
    assert PROBE_CHECKPOINT_STYLE_RESPONSE._transition_label(0.01, 0.02) == "trained_amplification"


def test_checkpoint_style_response_overall_reading_prefers_suppression() -> None:
    metrics = {
        "matched_target_spatial_forward_delta": {"transition": "trained_suppression"},
        "matched_target_both_forward_delta": {"transition": "trained_suppression"},
        "topology_gate1_blend_effect_delta": {"transition": "trained_suppression"},
        "styleid_max_forward_pair_delta": {"transition": "roughly_stable"},
        "styleid_mean_forward_pair_delta": {"transition": "roughly_stable"},
        "styleid_max_body_pair_delta": {"transition": "persistent_noop"},
        "styleid_max_delta_pair_delta": {"transition": "roughly_stable"},
    }

    assert PROBE_CHECKPOINT_STYLE_RESPONSE._overall_reading(metrics) == "trained_style_suppression"


def test_checkpoint_style_response_overall_reading_flags_mixed_decoder_only_case() -> None:
    metrics = {
        "matched_target_spatial_forward_delta": {"transition": "trained_suppression"},
        "matched_target_both_forward_delta": {"transition": "trained_suppression"},
        "topology_gate1_blend_effect_delta": {"transition": "trained_suppression"},
        "styleid_max_forward_pair_delta": {"transition": "trained_amplification"},
        "styleid_mean_forward_pair_delta": {"transition": "trained_amplification"},
        "styleid_max_body_pair_delta": {"transition": "persistent_noop"},
        "styleid_max_delta_pair_delta": {"transition": "trained_amplification"},
    }

    assert (
        PROBE_CHECKPOINT_STYLE_RESPONSE._overall_reading(metrics)
        == "matched_target_suppressed_styleid_amplified_body_dead"
    )


def test_sampled_bridge_plain_path_distill_is_active_and_preserves_conditioned_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(),
        bridge=BridgeConfig(
            objective_mode="bridge_velocity",
            w_flow=1.0,
            w_plain_path_distill=0.5,
        ),
    )
    objective = OTFlowMatchingObjective(cfg)
    zero = torch.tensor(0.0)

    monkeypatch.setattr(
        objective,
        "_ot_match_targets",
        lambda model, content, target_style, target_style_id, source_style_id: (
            target_style,
            zero.to(content.device),
            zero.to(content.device),
            zero.to(content.device),
            {},
        ),
    )
    monkeypatch.setattr(
        objective,
        "_project_training_target",
        lambda *, content, matched_target: (matched_target, {}),
    )
    monkeypatch.setattr(
        objective,
        "_sample_t",
        lambda content: torch.full((content.shape[0],), 0.5, device=content.device),
    )
    monkeypatch.setattr(
        objective,
        "_bridge_state_and_velocity",
        lambda *, content, matched_target, t: (content, torch.zeros_like(content), {}),
    )
    monkeypatch.setattr(
        objective,
        "_resolve_matched_target_conditioning",
        lambda model, *, matched_target, target_style_id: (None, torch.ones_like(matched_target[:, :1, :1, :1])),
    )
    monkeypatch.setattr(
        objective,
        "_terminal_swd",
        lambda *args, **kwargs: (None, torch.tensor(0.0),),
    )
    monkeypatch.setattr(
        objective,
        "_content_topology_anchor_loss",
        lambda *args, **kwargs: (
            torch.tensor(0.0),
            torch.tensor(0.0),
        ),
    )
    monkeypatch.setattr(
        objective,
        "_generated_delta_diversity_loss",
        lambda *args, **kwargs: (
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ),
    )
    monkeypatch.setattr(objective, "_cycle_consistency_loss", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(objective, "_fiber_probe_metrics", lambda **kwargs: {})
    monkeypatch.setattr(objective, "_profile_metrics", lambda content: {})
    monkeypatch.setattr(objective, "_model_profile_metrics", lambda model, content: {})

    class DummyBridge:
        transport_prediction_mode = "velocity"

        def __init__(self) -> None:
            self.last_semantic_attn = None
            self.last_semantic_k = None
            self.last_semantic_topology_attn = None
            self.last_style_path_debug = {}
            self.last_style_code_path_debug = {}

        def __call__(
            self,
            x: torch.Tensor,
            *,
            t: torch.Tensor,
            style_id: torch.Tensor,
            style_code_override: torch.Tensor | None = None,
            target_style_latent: torch.Tensor | None = None,
        ) -> torch.Tensor:
            conditioned = style_code_override is not None or target_style_latent is not None
            marker = 2.0 if conditioned else 0.5
            self.last_semantic_attn = x.new_full((1, 1, 1, 1), marker)
            self.last_semantic_k = x.new_full((1, 1, 1, 1), marker + 1.0)
            self.last_semantic_topology_attn = x.new_full((1, 1, 1, 1), 0.25 if conditioned else 0.75)
            self.last_style_path_debug = {
                "style_spatial_source_target_latent": 1.0 if conditioned else 0.0,
            }
            self.last_style_code_path_debug = {
                "style_code_override_active": 1.0 if conditioned else 0.0,
            }
            return x + marker

    model = DummyBridge()
    content = torch.randn(2, 4, 8, 8)
    target_style = torch.randn(2, 4, 8, 8)
    target_style_id = torch.tensor([0, 1], dtype=torch.long)

    metrics, components, _ = objective._compute_sampled_bridge_details(
        model,
        content=content,
        target_style=target_style,
        target_style_id=target_style_id,
    )

    assert metrics["plain_path_distill_active"].item() == 1.0
    assert metrics["plain_path_distill"].item() > 0.0
    assert metrics["plain_path_student_abs"].item() > 0.0
    assert components["plain_path_distill"].item() > 0.0
    assert metrics["style_code_override_active"].item() == 1.0
    assert metrics["style_spatial_source_target_latent"].item() == 1.0


def test_style_sweep_preflight_keeps_training_live_variants() -> None:
    specs = [
        {"name": "r15_vertical_blend_0p00", "overrides": {}},
        {"name": "dead_variant", "overrides": {}},
    ]
    config_preflight = {
        "variant_classification": {
            "r15_vertical_blend_0p00": {"classification": "train_graph_only"},
            "dead_variant": {"classification": "no_effect"},
        }
    }
    training_preflight = {
        "variant_classification": {
            "r15_vertical_blend_0p00": {"classification": "bridge_only_change"},
            "dead_variant": {"classification": "no_training_effect"},
        }
    }

    selected, skipped = PHASE616_AUTO._apply_style_sweep_preflight(
        specs=specs,
        preflight=config_preflight,
        training_preflight=training_preflight,
        include_train_graph_only=False,
        explicit_include_names=set(),
    )

    assert [spec["name"] for spec in selected] == ["r15_vertical_blend_0p00"]
    assert selected[0]["training_effect_preflight"]["classification"] == "bridge_only_change"
    assert [entry["name"] for entry in skipped] == ["dead_variant"]


def test_style_sweep_preflight_keeps_validity_metadata_on_skipped_rows() -> None:
    specs = [
        {
            "name": "dead_variant",
            "overrides": {},
            "validity_audit": {"artifact_status": "suspect", "effect_contract": "unknown"},
        }
    ]
    config_preflight = {
        "variant_classification": {
            "dead_variant": {"classification": "no_effect"},
        }
    }
    training_preflight = {
        "variant_classification": {
            "dead_variant": {"classification": "no_training_effect"},
        }
    }

    selected, skipped = PHASE616_AUTO._apply_style_sweep_preflight(
        specs=specs,
        preflight=config_preflight,
        training_preflight=training_preflight,
        include_train_graph_only=False,
        explicit_include_names=set(),
    )

    assert selected == []
    assert len(skipped) == 1
    assert skipped[0]["validity_audit"]["artifact_status"] == "suspect"


def test_close_result_diagnosis_marks_runtime_real_cluster_as_weak() -> None:
    entries = [
        {
            "name": "r11",
            "transfer_clip_style": 0.6680,
            "transfer_content_lpips": 0.3010,
            "objective_gap": 0.0730,
            "validity_audit": {
                "artifact_status": "valid",
                "effect_contract": "runtime_and_training_real",
            },
        },
        {
            "name": "r12",
            "transfer_clip_style": 0.6672,
            "transfer_content_lpips": 0.3040,
            "objective_gap": 0.0768,
            "validity_audit": {
                "artifact_status": "valid",
                "effect_contract": "runtime_and_training_real",
            },
        },
    ]
    best = {
        "name": "r11",
        "style": 0.6680,
        "lpips": 0.3010,
        "gap": 0.0730,
    }

    diagnosis = PHASE616_AUTO._summarize_close_result_diagnosis(entries, best)

    assert diagnosis["status"] == "close_cluster"
    assert diagnosis["interpretation"] == "runtime_real_but_weak"
    assert diagnosis["close_peer_count"] == 1


def test_close_result_diagnosis_marks_eval_inert_cluster_as_contract_gap() -> None:
    entries = [
        {
            "name": "h0",
            "transfer_clip_style": 0.6660,
            "transfer_content_lpips": 0.3000,
            "objective_gap": 0.0740,
            "validity_audit": {
                "artifact_status": "valid",
                "effect_contract": "training_real_eval_inert",
            },
        },
        {
            "name": "h2",
            "transfer_clip_style": 0.6658,
            "transfer_content_lpips": 0.3030,
            "objective_gap": 0.0772,
            "validity_audit": {
                "artifact_status": "valid",
                "effect_contract": "training_real_eval_inert",
            },
        },
    ]
    best = {
        "name": "h0",
        "style": 0.6660,
        "lpips": 0.3000,
        "gap": 0.0740,
    }

    diagnosis = PHASE616_AUTO._summarize_close_result_diagnosis(entries, best)

    assert diagnosis["status"] == "close_cluster"
    assert diagnosis["interpretation"] == "train_eval_contract_gap"


def test_phase616_auto_prepare_run_config_preserves_repaired_lowrank_family(tmp_path: Path) -> None:
    base_cfg = {
        "model": {
            "tokenizer_family": "pure_latent_spatial",
            "style_tokenizer": "factorized",
            "matched_target_conditioning_mode": "both",
            "matched_target_style_encoder_mode": "residual",
            "style_code_spatial_mode": "lowrank",
            "style_code_spatial_scale": 0.35,
        },
        "bridge": {},
        "training": {},
        "data": {},
    }

    cfg_path = PHASE616_AUTO._prepare_run_config(
        base_cfg,
        run_dir=tmp_path / "phase618_lowrank_run",
        name="phase618_lowrank_run",
        overrides={},
        num_epochs=3,
    )
    run_cfg = PHASE616_AUTO._load_json(cfg_path)

    assert run_cfg["model"]["tokenizer_family"] == "pure_latent_spatial"
    assert run_cfg["model"]["matched_target_conditioning_mode"] == "both"
    assert run_cfg["model"]["matched_target_style_encoder_mode"] == "residual"
    assert run_cfg["model"]["style_code_spatial_mode"] == "lowrank"
    assert run_cfg["model"]["style_code_spatial_scale"] == pytest.approx(0.35)


def test_gen_lite_batch_preserves_repaired_lowrank_family() -> None:
    base_cfg = {
        "model": {
            "tokenizer_family": "pure_latent_spatial",
            "style_tokenizer": "factorized",
            "matched_target_conditioning_mode": "both",
            "matched_target_style_encoder_mode": "residual",
            "style_code_spatial_mode": "lowrank",
            "style_code_spatial_scale": 0.35,
        },
        "bridge": {},
        "training": {},
        "data": {},
        "checkpoint": {},
    }

    run_cfg = GEN_LITE_BATCH.build_run_config(
        base_cfg,
        save_dir="/tmp/phase618_manual/h0_vertical_fm",
        overrides={"bridge.bridge_path_mode": "vertical"},
    )

    assert run_cfg["model"]["tokenizer_family"] == "pure_latent_spatial"
    assert run_cfg["model"]["matched_target_conditioning_mode"] == "both"
    assert run_cfg["model"]["matched_target_style_encoder_mode"] == "residual"
    assert run_cfg["model"]["style_code_spatial_mode"] == "lowrank"
    assert run_cfg["model"]["style_code_spatial_scale"] == pytest.approx(0.35)
    assert run_cfg["checkpoint"]["save_dir"] == "/tmp/phase618_manual/h0_vertical_fm"


def test_gen_lite_batch_rejects_old_legacy_base_by_default() -> None:
    legacy_cfg = {
        "model": {
            "tokenizer_family": "legacy_factorized",
            "matched_target_conditioning_mode": "auto",
            "matched_target_style_encoder_mode": "none",
            "style_code_spatial_mode": "none",
            "style_code_spatial_scale": 0.0,
        }
    }

    with pytest.raises(ValueError, match="repaired phase618 lowrank carrier base"):
        GEN_LITE_BATCH.validate_phase618_base(legacy_cfg, allow_legacy=False)


def test_phase616_validity_preflight_marks_old_base_style_sweep_confounded(tmp_path: Path) -> None:
    base_cfg = json.loads(
        (ROOT / "docs" / "experiments" / "2026-06-18-remote-h1-e18-diagnosis" / "remote_config.json").read_text(
            encoding="utf-8"
        )
    )
    specs = [
        {
            "name": "r8_linear_code_map_lowrank_both",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.matched_target_conditioning_mode": "both",
                "model.matched_target_style_encoder_mode": "residual",
                "model.matched_target_style_encoder_hidden_dim": 64,
                "model.style_code_spatial_mode": "lowrank",
                "model.style_code_spatial_hidden_dim": 64,
                "model.style_code_spatial_rank": 8,
                "model.style_code_spatial_base_hw": 16,
                "model.style_code_spatial_scale": 0.35,
            },
        }
    ]

    payload = PHASE616_AUTO._run_variant_validity_preflight(
        base_cfg=base_cfg,
        stage_root=tmp_path / "style_sweep_old_base",
        specs=specs,
    )

    meta = payload["variant_classification"]["r8_linear_code_map_lowrank_both"]
    assert meta["artifact_status"] == "confounded"
    assert meta["effect_contract"] == "runtime_real"
    assert meta["suite"] == "stage3_style_r1_r10_old_base"


def test_phase616_validity_preflight_marks_repaired_blend_runtime_real(tmp_path: Path) -> None:
    base_cfg = json.loads(
        (
            ROOT
            / "docs"
            / "experiments"
            / "2026-06-18-stage1-lowrank-rerun-audit"
            / "baseline_h1_lowrank_config.json"
        ).read_text(encoding="utf-8")
    )
    specs = [{"name": "r11_linear_blend_0p00", "overrides": {"model.semantic_self_topology_blend": 0.0}}]

    payload = PHASE616_AUTO._run_variant_validity_preflight(
        base_cfg=base_cfg,
        stage_root=tmp_path / "style_sweep_repaired",
        specs=specs,
    )

    meta = payload["variant_classification"]["r11_linear_blend_0p00"]
    assert meta["artifact_status"] == "valid"
    assert meta["effect_contract"] == "runtime_and_training_real"
    assert meta["suite"] == "bold_r11_r16_repaired_lowrank"


def test_run_stage1_writes_validity_preflight_to_stage_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_cfg_path = tmp_path / "base_stage1.json"
    base_cfg_path.write_text(json.dumps({"model": {}, "bridge": {}, "training": {}, "data": {}}), encoding="utf-8")
    stage_root = tmp_path / "stage1_auto"

    monkeypatch.setattr(PHASE616_AUTO, "_stage1_specs", lambda: [{"name": "h0_vertical_fm", "overrides": {}}])
    monkeypatch.setattr(
        PHASE616_AUTO,
        "_run_variant_validity_preflight",
        lambda **kwargs: {
            "variant_classification": {
                "h0_vertical_fm": {"artifact_status": "valid", "effect_contract": "training_real_eval_inert"}
            }
        },
    )
    monkeypatch.setattr(
        PHASE616_AUTO,
        "_run_style_sweep_config_effect_preflight",
        lambda **kwargs: {"variant_classification": {"h0_vertical_fm": {"classification": "no_effect"}}},
    )
    monkeypatch.setattr(
        PHASE616_AUTO,
        "_run_variant_training_effect_preflight",
        lambda **kwargs: {"variant_classification": {"h0_vertical_fm": {"classification": "bridge_only_change"}}},
    )

    def fake_run_manifest(**kwargs):
        specs = kwargs["specs"]
        assert specs[0]["validity_audit"]["artifact_status"] == "valid"
        assert specs[0]["config_effect_preflight"]["classification"] == "no_effect"
        assert specs[0]["training_effect_preflight"]["classification"] == "bridge_only_change"
        return {"best": {"name": "h0_vertical_fm"}}

    monkeypatch.setattr(PHASE616_AUTO, "_run_manifest", fake_run_manifest)

    args = SimpleNamespace(
        base_cfg=str(base_cfg_path),
        stage_root=str(stage_root),
        num_epochs=3,
        batch_candidates=[16],
        probe_steps=20,
        probe_timeout_sec=40,
        skip_probe=True,
        fixed_batch_size=16,
        skip_name=[],
        skip_config_effect_preflight=False,
        skip_training_effect_preflight=False,
    )

    assert PHASE616_AUTO.run_stage1(args) == 0
    payload = json.loads((stage_root / "stage_summary.json").read_text(encoding="utf-8"))
    assert payload["validity_preflight"]["variant_classification"]["h0_vertical_fm"]["artifact_status"] == "valid"
    assert payload["config_effect_preflight"]["variant_classification"]["h0_vertical_fm"]["classification"] == "no_effect"
    assert (
        payload["training_effect_preflight"]["variant_classification"]["h0_vertical_fm"]["classification"]
        == "bridge_only_change"
    )
    assert payload["close_result_diagnosis"]["status"] in {"insufficient_runs", "separated", "close_cluster"}


def test_run_style_sweep_writes_validity_preflight_and_skipped_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_cfg = {
        "model": {
            "tokenizer_family": "pure_latent_spatial",
            "matched_target_conditioning_mode": "both",
            "matched_target_style_encoder_mode": "residual",
            "style_code_spatial_mode": "lowrank",
            "style_code_spatial_scale": 0.35,
        },
        "bridge": {},
        "training": {},
        "data": {},
    }
    base_cfg_path = tmp_path / "base_style_sweep.json"
    base_cfg_path.write_text(json.dumps(base_cfg), encoding="utf-8")
    stage_root = tmp_path / "style_sweep_auto"

    monkeypatch.setattr(
        PHASE616_AUTO,
        "_style_sweep_specs",
        lambda: [
            {"name": "live_variant", "overrides": {}},
            {"name": "dead_variant", "overrides": {}},
        ],
    )
    monkeypatch.setattr(
        PHASE616_AUTO,
        "_run_variant_validity_preflight",
        lambda **kwargs: {
            "variant_classification": {
                "live_variant": {"artifact_status": "valid", "effect_contract": "runtime_and_training_real"},
                "dead_variant": {"artifact_status": "suspect", "effect_contract": "unknown"},
            }
        },
    )
    monkeypatch.setattr(
        PHASE616_AUTO,
        "_run_style_sweep_config_effect_preflight",
        lambda **kwargs: {
            "variant_classification": {
                "live_variant": {"classification": "plain_eval_change"},
                "dead_variant": {"classification": "no_effect"},
            }
        },
    )
    monkeypatch.setattr(
        PHASE616_AUTO,
        "_run_variant_training_effect_preflight",
        lambda **kwargs: {
            "variant_classification": {
                "live_variant": {"classification": "bridge_only_change"},
                "dead_variant": {"classification": "no_training_effect"},
            }
        },
    )
    monkeypatch.setattr(PHASE616_AUTO, "_best_curve_point", lambda *args, **kwargs: None)

    def fake_run_manifest(**kwargs):
        specs = kwargs["specs"]
        assert [spec["name"] for spec in specs] == ["live_variant"]
        assert specs[0]["validity_audit"]["artifact_status"] == "valid"
        return {
            "best": {
                "name": "live_variant",
                "run_dir": str(stage_root / "live_variant"),
                "style": 0.66,
                "lpips": 0.34,
                "gap": 0.12,
            }
        }

    monkeypatch.setattr(PHASE616_AUTO, "_run_manifest", fake_run_manifest)

    args = SimpleNamespace(
        base_cfg=str(base_cfg_path),
        stage_root=str(stage_root),
        include_name=[],
        skip_config_effect_preflight=False,
        skip_training_effect_preflight=False,
        include_train_graph_only=False,
        num_epochs=3,
        batch_candidates=[16],
        probe_steps=20,
        probe_timeout_sec=40,
        skip_probe=True,
        fixed_batch_size=16,
        skip_name=[],
        reference_h0_dir=str(tmp_path / "missing_h0"),
        reference_h1_dir=str(tmp_path / "missing_h1"),
    )

    assert PHASE616_AUTO.run_style_sweep(args) == 0
    payload = json.loads((stage_root / "stage_summary.json").read_text(encoding="utf-8"))
    assert payload["validity_preflight"]["variant_classification"]["live_variant"]["artifact_status"] == "valid"
    assert len(payload["skipped_by_preflight"]) == 1
    assert payload["skipped_by_preflight"][0]["name"] == "dead_variant"
    assert payload["skipped_by_preflight"][0]["validity_audit"]["artifact_status"] == "suspect"
    assert payload["close_result_diagnosis"]["status"] in {"insufficient_runs", "separated", "close_cluster"}


def test_run_plain_path_distill_writes_preflight_to_stage_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_cfg = {
        "model": {
            "tokenizer_family": "pure_latent_spatial",
            "matched_target_conditioning_mode": "both",
            "matched_target_style_encoder_mode": "residual",
            "style_code_spatial_mode": "lowrank",
            "style_code_spatial_scale": 0.35,
        },
        "bridge": {},
        "training": {},
        "data": {},
    }
    base_cfg_path = tmp_path / "base_plain_path_distill.json"
    base_cfg_path.write_text(json.dumps(base_cfg), encoding="utf-8")
    stage_root = tmp_path / "plain_path_distill_auto"

    monkeypatch.setattr(
        PHASE616_AUTO,
        "_plain_path_distill_specs",
        lambda: [{"name": "h1_plain_path_distill_0p50", "overrides": {"bridge.w_plain_path_distill": 0.5}}],
    )
    monkeypatch.setattr(
        PHASE616_AUTO,
        "_run_variant_validity_preflight",
        lambda **kwargs: {
            "variant_classification": {
                "h1_plain_path_distill_0p50": {
                    "artifact_status": "valid",
                    "effect_contract": "training_only_by_design",
                }
            }
        },
    )
    monkeypatch.setattr(
        PHASE616_AUTO,
        "_run_style_sweep_config_effect_preflight",
        lambda **kwargs: {
            "variant_classification": {
                "h1_plain_path_distill_0p50": {"classification": "no_effect"}
            }
        },
    )
    monkeypatch.setattr(
        PHASE616_AUTO,
        "_run_variant_training_effect_preflight",
        lambda **kwargs: {
            "variant_classification": {
                "h1_plain_path_distill_0p50": {"classification": "conditioning_or_loss_change"}
            }
        },
    )

    def fake_run_manifest(**kwargs):
        specs = kwargs["specs"]
        assert specs[0]["validity_audit"]["effect_contract"] == "training_only_by_design"
        assert specs[0]["config_effect_preflight"]["classification"] == "no_effect"
        assert specs[0]["training_effect_preflight"]["classification"] == "conditioning_or_loss_change"
        return {
            "best": {
                "name": "h1_plain_path_distill_0p50",
                "style": 0.66,
                "lpips": 0.33,
                "gap": 0.11,
            }
        }

    monkeypatch.setattr(PHASE616_AUTO, "_run_manifest", fake_run_manifest)

    args = SimpleNamespace(
        base_cfg=str(base_cfg_path),
        stage_root=str(stage_root),
        include_name=[],
        skip_config_effect_preflight=False,
        skip_training_effect_preflight=False,
        num_epochs=4,
        batch_candidates=[16],
        probe_steps=20,
        probe_timeout_sec=40,
        skip_probe=True,
        fixed_batch_size=16,
        skip_name=[],
    )

    assert PHASE616_AUTO.run_plain_path_distill(args) == 0
    payload = json.loads((stage_root / "stage_summary.json").read_text(encoding="utf-8"))
    assert (
        payload["validity_preflight"]["variant_classification"]["h1_plain_path_distill_0p50"]["effect_contract"]
        == "training_only_by_design"
    )
    assert (
        payload["config_effect_preflight"]["variant_classification"]["h1_plain_path_distill_0p50"]["classification"]
        == "no_effect"
    )
    assert (
        payload["training_effect_preflight"]["variant_classification"]["h1_plain_path_distill_0p50"]["classification"]
        == "conditioning_or_loss_change"
    )
    assert payload["close_result_diagnosis"]["status"] in {"insufficient_runs", "separated", "close_cluster"}
