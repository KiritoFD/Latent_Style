from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


COMMON_PARENT_CONFIG = "SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_seed42_b8a2.json"
ROUND2_CONFIG_DIR = "SchrodingerBridge/configs/aaai2027/round2_pure_sde"
ROUND2_DOC_DIR = "SchrodingerBridge/docs/experiments/round2_pure_sde"


@dataclass(frozen=True)
class Round2PureSDESpec:
    family_id: str
    wave: str
    axis: str
    model_overrides: dict[str, Any]
    bridge_overrides: dict[str, Any]
    training_overrides: dict[str, Any]
    notes: str
    patience: int
    data_overrides: dict[str, Any] = field(default_factory=dict)
    launch_health_min_runtime_memory_mib: int | None = None


def _pure_i2sb_bridge_overrides(
    *,
    bridge_sigma: float,
    terminal_num_steps: int | None = None,
    with_heuristics: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "objective_mode": "i2sb_endpoint",
        "loss_type": "mse",
        "bridge_sigma": float(bridge_sigma),
        "bridge_noise_mode": "gaussian",
        "bridge_noise_schedule": "exact_brownian",
        "sb_noise_epsilon": 0.0,
        "semantic_supervision_family": "legacy_terminal_swd",
        "dino_masked_swd_weight": 0.0,
        "w_flow": 1.0,
        "terminal_swd_weight": 18.0,
        "terminal_swd_aux_weight": 0.0,
        "w_kinetic": 0.0,
        "w_curvature": 0.0,
        "structure_penalty_mode": "off",
        "w_anisotropic_kinetic": 0.0,
        "w_stokes_viscous": 0.0,
        "w_phase_separation": 0.0,
        "w_style_energy_floor": 0.0,
        "w_lowfreq_velocity": 0.0,
        "w_style_contrastive": 0.0,
        "w_residual_style_direction": 0.0,
        "w_generated_delta_diversity": 0.0,
        "w_spectral_amplitude": 0.0,
        "target_teacher_mode": "off",
        "target_teacher_weight": 0.0,
        "cycle_consistency_weight": 0.0,
    }
    if terminal_num_steps is not None:
        payload["terminal_num_steps"] = int(terminal_num_steps)
    if with_heuristics:
        payload.update(
            {
                "structure_penalty_mode": "anisotropic_plus_stokes",
                "w_anisotropic_kinetic": 0.05,
                "w_stokes_viscous": 0.05,
                "w_curvature": 0.01,
            }
        )
    return payload


def _pure_latent_model_overrides(*, solver_family: str) -> dict[str, Any]:
    return {
        "style_tokenizer": "null",
        "tokenizer_family": "pure_latent_spatial",
        "tokenizer_num_clusters": 32,
        "transport_prediction_mode": "endpoint",
        "solver_family": str(solver_family),
        "proximal_mode": "off",
        # Round-2 pure mainline owns style context via the structured tokenizer only.
        "tokenizer_content_adaptive": False,
        "style_spatial_mode": "disabled",
        "style_id_spatial_jitter_px": 0,
        "use_diffeomorphic_stroke": False,
        "style_injection_mode": "none",
        "record_base_endpoint_metrics": False,
    }


ROUND2_PURE_SDE_SPECS: tuple[Round2PureSDESpec, ...] = (
    Round2PureSDESpec(
        family_id="tok_baseline_global",
        wave="wave1_tokenizer",
        axis="tokenizer",
        model_overrides={
            "tokenizer_family": "legacy_factorized",
            "transport_prediction_mode": "endpoint",
            "solver_family": "euler_legacy",
            "proximal_mode": "off",
            "ablation_disable_spatial_prior": True,
            "style_spatial_mode": "disabled",
            "style_id_spatial_jitter_px": 0,
        },
        bridge_overrides=_pure_i2sb_bridge_overrides(bridge_sigma=0.0),
        training_overrides={"batch_size": 46, "accumulation_steps": 1},
        notes="Wave-1 baseline with the legacy global code only. No DINO sidecar and no stochastic bridge.",
        patience=4,
    ),
    Round2PureSDESpec(
        family_id="tok_pure_latent_spatial",
        wave="wave1_tokenizer",
        axis="tokenizer",
        model_overrides=_pure_latent_model_overrides(solver_family="euler_legacy"),
        bridge_overrides=_pure_i2sb_bridge_overrides(bridge_sigma=0.0),
        training_overrides={"batch_size": 37, "accumulation_steps": 1},
        notes="Wave-1 proposed tokenizer: latent-native spatial routing from z0 with deterministic ODE transport.",
        patience=4,
        launch_health_min_runtime_memory_mib=6600,
    ),
    Round2PureSDESpec(
        family_id="sde_i2sb_sigma_0p25",
        wave="wave2_sde_noise",
        axis="solver",
        model_overrides=_pure_latent_model_overrides(solver_family="solver_i2sb"),
        bridge_overrides=_pure_i2sb_bridge_overrides(bridge_sigma=0.25),
        training_overrides={"batch_size": 40, "accumulation_steps": 1},
        notes="Wave-2 mild Brownian bridge noise for exact-posterior I2SB.",
        patience=4,
    ),
    Round2PureSDESpec(
        family_id="sde_i2sb_sigma_0p5",
        wave="wave2_sde_noise",
        axis="solver",
        model_overrides=_pure_latent_model_overrides(solver_family="solver_i2sb"),
        bridge_overrides=_pure_i2sb_bridge_overrides(bridge_sigma=0.5),
        training_overrides={"batch_size": 40, "accumulation_steps": 1},
        notes="Wave-2 mainline I2SB setting. Exact posterior with the recommended bridge sigma.",
        patience=4,
    ),
    Round2PureSDESpec(
        family_id="sde_i2sb_sigma_1p0",
        wave="wave2_sde_noise",
        axis="solver",
        model_overrides=_pure_latent_model_overrides(solver_family="solver_i2sb"),
        bridge_overrides=_pure_i2sb_bridge_overrides(bridge_sigma=1.0),
        training_overrides={"batch_size": 32, "accumulation_steps": 1},
        notes="Wave-2 high-noise stress test for structure robustness under true stochastic transport.",
        patience=4,
    ),
    Round2PureSDESpec(
        family_id="sde_optimal_with_heuristics",
        wave="wave3_ablation",
        axis="losses",
        model_overrides={
            **_pure_latent_model_overrides(solver_family="solver_i2sb"),
            "use_diffeomorphic_stroke": True,
            "style_injection_mode": "body_decoder",
            "style_injection_form": "spatial_carrier_gate",
            "record_base_endpoint_metrics": True,
        },
        bridge_overrides=_pure_i2sb_bridge_overrides(bridge_sigma=0.5, with_heuristics=True),
        training_overrides={"batch_size": 32, "accumulation_steps": 1},
        notes="Wave-3 ablation branch that keeps the old structure heuristics on top of I2SB.",
        patience=4,
    ),
    Round2PureSDESpec(
        family_id="sde_optimal_clean",
        wave="wave3_ablation",
        axis="losses",
        model_overrides=_pure_latent_model_overrides(solver_family="solver_i2sb"),
        bridge_overrides=_pure_i2sb_bridge_overrides(bridge_sigma=0.5),
        training_overrides={"batch_size": 40, "accumulation_steps": 1},
        notes="Wave-3 clean mainline: pure endpoint regression plus terminal SWD, with DINO retired.",
        patience=5,
    ),
    Round2PureSDESpec(
        family_id="sde_clean_nfe_4",
        wave="wave4_efficiency",
        axis="inference",
        model_overrides=_pure_latent_model_overrides(solver_family="solver_i2sb"),
        bridge_overrides=_pure_i2sb_bridge_overrides(bridge_sigma=0.5, terminal_num_steps=4),
        training_overrides={"batch_size": 40, "accumulation_steps": 1, "full_eval_num_steps": 4},
        notes="Wave-4 efficiency probe at 4 inference steps.",
        patience=2,
    ),
    Round2PureSDESpec(
        family_id="sde_clean_nfe_8",
        wave="wave4_efficiency",
        axis="inference",
        model_overrides=_pure_latent_model_overrides(solver_family="solver_i2sb"),
        bridge_overrides=_pure_i2sb_bridge_overrides(bridge_sigma=0.5, terminal_num_steps=8),
        training_overrides={"batch_size": 40, "accumulation_steps": 1, "full_eval_num_steps": 8},
        notes="Wave-4 recommended operational point at 8 inference steps.",
        patience=2,
    ),
)
