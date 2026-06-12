from __future__ import annotations

TOKENIZER_FAMILIES = {
    "legacy_factorized",
    "pure_latent_spatial",
    "tok_a_dino_dict",
    "tok_b_cross_image",
    "tok_c_residual_adapter",
    "tok_d_vlm_prompt",
}

PURE_LATENT_COMPAT_STRIP_PREFIXES = (
    "style_spatial_id_16",
    "style_spatial_proto_16",
    "style_spatial_atoms_16",
    "style_spatial_logits.",
    "style_spatial_content_router.",
)

PURE_LATENT_COMPAT_ONLY_TOKENIZER_PREFIXES = (
    "style_tokenizer.",
)

STYLE_INJECTION_PREFIXES = (
    "body_style_injector.",
    "decoder_style_injector.",
    "body_style_carrier.",
    "decoder_style_carrier.",
    "body_content_gate.",
    "decoder_content_gate.",
    "body_style_spatial_proj.",
    "decoder_style_spatial_proj.",
    "body_structure_gate.",
    "decoder_structure_gate.",
)

PROXIMAL_OPTIONAL_PREFIXES = (
    "proximal_attn_q.",
    "proximal_attn_k.",
    "proximal_attn_v.",
    "proximal_attn_out.",
    "proximal_style_tokens.",
)

BACKBONE_ATTENTION_FAMILIES = {
    "legacy_semantic_crossattn",
    "attn_sa_mod",
    "attn_gw_ot",
    "attn_gated_spade",
    "attn_pnp_selfinject",
}

SOLVER_FAMILIES = {
    "euler_legacy",
    "solver_i2sb",
    "solver_tangent_rk",
    "solver_pc",
    "solver_unsb_cycle",
}

SEMANTIC_SUPERVISION_FAMILIES = {
    "legacy_terminal_swd",
    "dino_masked_swd",
}

TRANSPORT_PREDICTION_MODES = {
    "velocity",
    "endpoint",
}

BRIDGE_NOISE_SCHEDULES = {
    "auto",
    "exact_brownian",
    "delayed_window",
}

I2SB_OBJECTIVE_MODES = {
    "i2sb",
    "i2sb_endpoint",
    "bridge_endpoint",
}

DINO_CONDITIONED_TOKENIZER_FAMILIES = {
    "tok_a_dino_dict",
    "tok_b_cross_image",
    "tok_c_residual_adapter",
    "tok_d_vlm_prompt",
}


def normalize_family(value: str, *, allowed: set[str], default: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in allowed:
        return candidate
    return default


def normalize_tokenizer_family(value: str, *, default: str = "legacy_factorized") -> str:
    return normalize_family(value, allowed=TOKENIZER_FAMILIES, default=default)


def tokenizer_family_requires_dino(value: str) -> bool:
    family = normalize_tokenizer_family(value)
    return family in DINO_CONDITIONED_TOKENIZER_FAMILIES


def semantic_supervision_requires_dino(value: str) -> bool:
    family = normalize_family(value, allowed=SEMANTIC_SUPERVISION_FAMILIES, default="legacy_terminal_swd")
    return family == "dino_masked_swd"


def runtime_conditioning_requires_dino(*, tokenizer_family: str, semantic_supervision_family: str) -> bool:
    return tokenizer_family_requires_dino(tokenizer_family) or semantic_supervision_requires_dino(semantic_supervision_family)


def validate_dino_retired_runtime(
    *,
    tokenizer_family: str,
    semantic_supervision_family: str,
    allow_dino: bool = False,
    context: str = "round2 pure-sde launcher",
) -> None:
    if allow_dino:
        return
    family = normalize_tokenizer_family(tokenizer_family)
    semantic = normalize_family(
        semantic_supervision_family,
        allowed=SEMANTIC_SUPERVISION_FAMILIES,
        default="legacy_terminal_swd",
    )
    if not runtime_conditioning_requires_dino(
        tokenizer_family=family,
        semantic_supervision_family=semantic,
    ):
        return
    raise ValueError(
        f"{context} blocks DINO-conditioned configs by default; "
        f"got tokenizer_family={family!r}, semantic_supervision_family={semantic!r}. "
        "Use the explicit allow-dino override only if a later board result justifies reviving the archived DINO route."
    )


def compat_state_strip_prefixes_for_tokenizer_family(value: str) -> tuple[str, ...]:
    family = normalize_tokenizer_family(value)
    if family == "pure_latent_spatial":
        return PURE_LATENT_COMPAT_STRIP_PREFIXES
    return ()


def compat_state_strip_prefixes_for_model_contract(
    *,
    tokenizer_family: str,
    style_injection_mode: str = "",
    proximal_mode: str = "",
) -> tuple[str, ...]:
    prefixes: list[str] = list(compat_state_strip_prefixes_for_tokenizer_family(tokenizer_family))
    family = normalize_tokenizer_family(tokenizer_family)
    if family == "pure_latent_spatial":
        prefixes.extend(PURE_LATENT_COMPAT_ONLY_TOKENIZER_PREFIXES)
    mode = str(style_injection_mode or "").strip().lower()
    if mode in {"", "none"}:
        prefixes.extend(STYLE_INJECTION_PREFIXES)
    proximal = str(proximal_mode or "").strip().lower()
    if proximal in {"", "off"}:
        prefixes.extend(PROXIMAL_OPTIONAL_PREFIXES)
    return tuple(dict.fromkeys(prefixes))


def validate_pure_latent_contract(
    *,
    tokenizer_family: str,
    semantic_supervision_family: str = "",
    dino_masked_swd_weight: float = 0.0,
    style_spatial_mode: str = "",
    tokenizer_content_adaptive: bool = False,
) -> None:
    family = normalize_tokenizer_family(tokenizer_family)
    if family != "pure_latent_spatial":
        return
    semantic = normalize_family(
        semantic_supervision_family,
        allowed=SEMANTIC_SUPERVISION_FAMILIES,
        default="legacy_terminal_swd",
    )
    if semantic != "legacy_terminal_swd":
        raise ValueError(
            "tokenizer_family='pure_latent_spatial' requires bridge.semantic_supervision_family='legacy_terminal_swd'."
        )
    if float(dino_masked_swd_weight) > 0.0:
        raise ValueError(
            "tokenizer_family='pure_latent_spatial' requires bridge.dino_masked_swd_weight=0.0."
        )
    mode = str(style_spatial_mode or "").strip().lower()
    if mode and mode != "disabled":
        raise ValueError(
            "tokenizer_family='pure_latent_spatial' requires model.style_spatial_mode='disabled'."
        )
    if bool(tokenizer_content_adaptive):
        raise ValueError(
            "tokenizer_family='pure_latent_spatial' requires model.tokenizer_content_adaptive=false."
        )


def prune_state_dict_for_tokenizer_family(
    state_dict: dict[str, object],
    *,
    tokenizer_family: str,
    style_injection_mode: str = "",
    proximal_mode: str = "",
) -> tuple[dict[str, object], list[str]]:
    prefixes = compat_state_strip_prefixes_for_model_contract(
        tokenizer_family=tokenizer_family,
        style_injection_mode=style_injection_mode,
        proximal_mode=proximal_mode,
    )
    if not prefixes:
        return dict(state_dict), []
    kept: dict[str, object] = {}
    removed: list[str] = []
    for key, value in state_dict.items():
        if any(str(key).startswith(prefix) for prefix in prefixes):
            removed.append(str(key))
            continue
        kept[str(key)] = value
    return kept, removed


def normalize_transport_prediction_mode(value: str, *, default: str = "velocity") -> str:
    candidate = str(value or "").strip().lower()
    if candidate in TRANSPORT_PREDICTION_MODES:
        return candidate
    return default


def normalize_bridge_noise_schedule(value: str, *, default: str = "auto") -> str:
    candidate = str(value or "").strip().lower()
    if candidate in BRIDGE_NOISE_SCHEDULES:
        return candidate
    return default


def is_i2sb_objective_mode(value: str) -> bool:
    return str(value or "").strip().lower() in I2SB_OBJECTIVE_MODES


def resolves_exact_brownian_schedule(
    *,
    bridge_noise_schedule: str,
    objective_mode: str = "",
) -> bool:
    schedule = normalize_bridge_noise_schedule(bridge_noise_schedule)
    if schedule == "exact_brownian":
        return True
    if schedule == "delayed_window":
        return False
    return is_i2sb_objective_mode(objective_mode)


def is_true_i2sb_training_contract(
    *,
    solver_family: str,
    transport_prediction_mode: str,
    objective_mode: str = "",
    loss_type: str = "",
    bridge_noise_schedule: str = "auto",
) -> bool:
    solver = normalize_family(solver_family, allowed=SOLVER_FAMILIES, default="euler_legacy")
    transport = normalize_transport_prediction_mode(transport_prediction_mode)
    objective = str(objective_mode or "").strip().lower()
    loss = str(loss_type or "").strip().lower()
    if solver != "solver_i2sb":
        return False
    if transport != "endpoint":
        return False
    if not is_i2sb_objective_mode(objective):
        return False
    if loss and loss != "mse":
        return False
    return resolves_exact_brownian_schedule(
        bridge_noise_schedule=bridge_noise_schedule,
        objective_mode=objective,
    )


def validate_i2sb_contract(
    *,
    solver_family: str,
    transport_prediction_mode: str,
    objective_mode: str = "",
    loss_type: str = "",
) -> None:
    solver = normalize_family(solver_family, allowed=SOLVER_FAMILIES, default="euler_legacy")
    transport = normalize_transport_prediction_mode(transport_prediction_mode)
    objective = str(objective_mode or "").strip().lower()
    loss = str(loss_type or "").strip().lower()
    if solver == "solver_i2sb" and transport != "endpoint":
        raise ValueError("solver_i2sb requires model.transport_prediction_mode='endpoint'.")
    if solver == "solver_i2sb" and not is_i2sb_objective_mode(objective):
        raise ValueError("solver_i2sb requires bridge.objective_mode='i2sb_endpoint'.")
    if is_i2sb_objective_mode(objective_mode) and transport != "endpoint":
        raise ValueError("bridge.objective_mode='i2sb_endpoint' requires model.transport_prediction_mode='endpoint'.")
    if (solver == "solver_i2sb" or is_i2sb_objective_mode(objective)) and loss and loss != "mse":
        raise ValueError("true I2SB requires bridge.loss_type='mse'.")
