from __future__ import annotations

TOKENIZER_FAMILIES = {
    "legacy_factorized",
    "tok_a_dino_dict",
    "tok_b_cross_image",
    "tok_c_residual_adapter",
    "tok_d_vlm_prompt",
}

BACKBONE_ATTENTION_FAMILIES = {
    "legacy_semantic_crossattn",
    "attn_sa_mod",
    "attn_gw_ot",
    "attn_gated_spade",
    "attn_pnp_selfinject",
}

SOLVER_FAMILIES = {
    "euler_legacy",
    "solver_tangent_rk",
    "solver_pc",
    "solver_unsb_cycle",
}

SEMANTIC_SUPERVISION_FAMILIES = {
    "legacy_terminal_swd",
    "dino_masked_swd",
}


def normalize_family(value: str, *, allowed: set[str], default: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in allowed:
        return candidate
    return default
