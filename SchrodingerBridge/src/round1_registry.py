from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


COMMON_PARENT_CONFIG = "SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_seed42_b8a2.json"
ROUND1_CONFIG_DIR = "SchrodingerBridge/configs/aaai2027/round1_full_sweep"
ROUND1_DOC_DIR = "SchrodingerBridge/docs/experiments/round1_full_sweep"


@dataclass(frozen=True)
class Round1FamilySpec:
    family_id: str
    wave: str
    axis: str
    model_overrides: dict[str, Any]
    bridge_overrides: dict[str, Any]
    training_overrides: dict[str, Any]
    notes: str
    patience: int
    data_overrides: dict[str, Any] = field(default_factory=dict)


ROUND1_FAMILY_SPECS: tuple[Round1FamilySpec, ...] = (
    Round1FamilySpec(
        family_id="tok_a_dino_dict",
        wave="wave1_tokenizer",
        axis="tokenizer",
        model_overrides={"tokenizer_family": "tok_a_dino_dict"},
        bridge_overrides={"semantic_supervision_family": "dino_masked_swd", "dino_masked_swd_weight": 1.0},
        training_overrides={"batch_size": 8, "accumulation_steps": 2},
        notes="Universal keys plus style-specific values with DINO-masked SWD.",
        patience=4,
    ),
    Round1FamilySpec(
        family_id="tok_b_cross_image",
        wave="wave1_tokenizer",
        axis="tokenizer",
        model_overrides={"tokenizer_family": "tok_b_cross_image"},
        bridge_overrides={"semantic_supervision_family": "dino_masked_swd", "dino_masked_swd_weight": 1.0},
        training_overrides={"batch_size": 8, "accumulation_steps": 2},
        notes="Cross-image routing over style-bank DINO tokens through the closed-set style_id wrapper.",
        patience=4,
    ),
    Round1FamilySpec(
        family_id="tok_c_residual_adapter",
        wave="wave1_tokenizer",
        axis="tokenizer",
        model_overrides={"tokenizer_family": "tok_c_residual_adapter"},
        bridge_overrides={"semantic_supervision_family": "dino_masked_swd", "dino_masked_swd_weight": 1.0},
        training_overrides={"batch_size": 8, "accumulation_steps": 2},
        notes="Residual semantic adapter that preserves the global style code and routes high-frequency detail only.",
        patience=4,
    ),
    Round1FamilySpec(
        family_id="tok_d_vlm_prompt",
        wave="wave1_tokenizer",
        axis="tokenizer",
        model_overrides={"tokenizer_family": "tok_d_vlm_prompt"},
        bridge_overrides={"semantic_supervision_family": "dino_masked_swd", "dino_masked_swd_weight": 0.5},
        training_overrides={"batch_size": 8, "accumulation_steps": 2},
        notes="Prompt-token style priors routed into global and spatial conditioning.",
        patience=4,
    ),
    Round1FamilySpec(
        family_id="attn_sa_mod",
        wave="wave2_backbone",
        axis="backbone",
        model_overrides={"backbone_attention_family": "attn_sa_mod"},
        bridge_overrides={},
        training_overrides={"batch_size": 13, "accumulation_steps": 2},
        notes="Spatially modulated self-attention with content topology preserved in self affinity.",
        patience=4,
    ),
    Round1FamilySpec(
        family_id="attn_gw_ot",
        wave="wave2_backbone",
        axis="backbone",
        model_overrides={"backbone_attention_family": "attn_gw_ot"},
        bridge_overrides={},
        training_overrides={"batch_size": 13, "accumulation_steps": 2},
        notes="GW/OT-inspired routing with spatial cost regularization and sinkhorn normalization.",
        patience=4,
    ),
    Round1FamilySpec(
        family_id="attn_gated_spade",
        wave="wave2_backbone",
        axis="backbone",
        model_overrides={"backbone_attention_family": "attn_gated_spade"},
        bridge_overrides={},
        training_overrides={"batch_size": 13, "accumulation_steps": 2},
        notes="Locally gated SPADE-like injection balancing structure and style fields.",
        patience=4,
    ),
    Round1FamilySpec(
        family_id="attn_pnp_selfinject",
        wave="wave2_backbone",
        axis="backbone",
        model_overrides={"backbone_attention_family": "attn_pnp_selfinject"},
        bridge_overrides={},
        training_overrides={"batch_size": 19, "accumulation_steps": 2},
        notes="PnP-style self-attention injection using content affinity with style values.",
        patience=4,
    ),
    Round1FamilySpec(
        family_id="solver_tangent_rk",
        wave="wave2_solver",
        axis="solver",
        model_overrides={"solver_family": "solver_tangent_rk", "solver_rk_order": 4},
        bridge_overrides={},
        training_overrides={"batch_size": 16, "accumulation_steps": 2, "num_epochs": 48},
        data_overrides={"virtual_length_multiplier": 0.5},
        notes="Tangent-projected RK transport solver.",
        patience=6,
    ),
    Round1FamilySpec(
        family_id="solver_pc",
        wave="wave2_solver",
        axis="solver",
        model_overrides={"solver_family": "solver_pc", "solver_corrector_steps": 2},
        bridge_overrides={},
        training_overrides={"batch_size": 8, "accumulation_steps": 2, "num_epochs": 48},
        data_overrides={"virtual_length_multiplier": 0.5},
        notes="Predictor-corrector solver with structure-aware correction.",
        patience=6,
    ),
    Round1FamilySpec(
        family_id="solver_unsb_cycle",
        wave="wave2_solver",
        axis="solver",
        model_overrides={"solver_family": "solver_unsb_cycle", "solver_stochastic_noise_scale": 0.01},
        bridge_overrides={"cycle_consistency_weight": 0.1, "cycle_consistency_num_steps": 4},
        training_overrides={"batch_size": 8, "accumulation_steps": 2, "num_epochs": 48},
        data_overrides={"virtual_length_multiplier": 0.5},
        notes="UNSB-inspired stochastic bridge solver with cycle-consistency support.",
        patience=6,
    ),
)
