from __future__ import annotations

TOKENIZER_FAMILIES = {
    "legacy_factorized",
    "pure_latent_spatial",
    "affine_connection_tokenizer",
    "tok_a_dino_dict",
    "tok_b_cross_image",
    "tok_c_residual_adapter",
    "tok_d_vlm_prompt",
    "smoe_translator",
}

CONTRACT_FAMILIES = {
    "legacy",
    "phase616",
    "620_spatial_bridge",
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

STYLE_DELTA_OPTIONAL_PREFIXES = (
    "style_delta_basis_proj.",
    "style_delta_weight_head.",
    "style_section_basis_proj.",
    "style_section_weight_head.",
    "style_section_out.",
    "style_head_adapter_in.",
    "style_head_adapter_film.",
    "style_head_adapter_out.",
)

OUTPUT_APPEARANCE_OPTIONAL_PREFIXES = (
    "output_appearance_head.",
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
    "fiberwise_swd",
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

LATENT_STRUCTURED_TOKENIZER_FAMILIES = {
    "pure_latent_spatial",
    "smoe_translator",
    "affine_connection_tokenizer",
}

PURE_PLACEHOLDER_STRUCTURED_TOKENIZER_FAMILIES = {
    "pure_latent_spatial",
    "affine_connection_tokenizer",
}


def normalize_family(value: str, *, allowed: set[str], default: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in allowed:
        return candidate
    return default


def normalize_tokenizer_family(value: str, *, default: str = "legacy_factorized") -> str:
    return normalize_family(value, allowed=TOKENIZER_FAMILIES, default=default)


def normalize_contract_family(value: str, *, default: str = "legacy") -> str:
    return normalize_family(value, allowed=CONTRACT_FAMILIES, default=default)


def compat_state_strip_prefixes_for_tokenizer_family(value: str) -> tuple[str, ...]:
    family = normalize_tokenizer_family(value)
    if family in LATENT_STRUCTURED_TOKENIZER_FAMILIES:
        return PURE_LATENT_COMPAT_STRIP_PREFIXES
    return ()


def compat_state_strip_prefixes_for_model_contract(
    *,
    tokenizer_family: str,
    contract_family: str = "legacy",
    style_injection_mode: str = "",
    proximal_mode: str = "",
    style_delta_mode: str = "",
    output_appearance_alignment_mode: str = "",
) -> tuple[str, ...]:
    prefixes: list[str] = list(compat_state_strip_prefixes_for_tokenizer_family(tokenizer_family))
    family = normalize_tokenizer_family(tokenizer_family)
    contract = normalize_contract_family(contract_family)
    if family in PURE_PLACEHOLDER_STRUCTURED_TOKENIZER_FAMILIES:
        prefixes.extend(PURE_LATENT_COMPAT_ONLY_TOKENIZER_PREFIXES)
    mode = str(style_injection_mode or "").strip().lower()
    if mode in {"", "none"}:
        prefixes.extend(STYLE_INJECTION_PREFIXES)
    proximal = str(proximal_mode or "").strip().lower()
    if proximal in {"", "off"}:
        prefixes.extend(PROXIMAL_OPTIONAL_PREFIXES)
    delta_mode = str(style_delta_mode or "").strip().lower()
    if delta_mode in {"", "none"}:
        prefixes.extend(STYLE_DELTA_OPTIONAL_PREFIXES)
    appearance_mode = str(output_appearance_alignment_mode or "").strip().lower()
    if appearance_mode in {"", "none"}:
        prefixes.extend(OUTPUT_APPEARANCE_OPTIONAL_PREFIXES)
    if contract == "phase616":
        prefixes.extend(OUTPUT_APPEARANCE_OPTIONAL_PREFIXES)
        prefixes.extend(PROXIMAL_OPTIONAL_PREFIXES)
        prefixes.extend(STYLE_DELTA_OPTIONAL_PREFIXES)
    return tuple(dict.fromkeys(prefixes))


def validate_pure_latent_contract(
    *,
    tokenizer_family: str,
    style_tokenizer: str = "",
    semantic_supervision_family: str = "",
    tokenizer_content_adaptive: bool = False,
) -> None:
    family = normalize_tokenizer_family(tokenizer_family)
    if family != "pure_latent_spatial":
        if family != "affine_connection_tokenizer":
            return
    tokenizer_kind = str(style_tokenizer or "").strip().lower()
    if tokenizer_kind not in {"", "null", "none", "pure_placeholder"}:
        raise ValueError(
            f"tokenizer_family={family!r} requires model.style_tokenizer "
            "to be an explicit null compatibility placeholder ('null'/'none'), "
            f"got style_tokenizer={style_tokenizer!r}."
        )
    semantic = normalize_family(
        semantic_supervision_family,
        allowed=SEMANTIC_SUPERVISION_FAMILIES,
        default="legacy_terminal_swd",
    )
    if semantic != "legacy_terminal_swd":
        raise ValueError(
            f"tokenizer_family={family!r} requires bridge.semantic_supervision_family='legacy_terminal_swd'."
        )
    if bool(tokenizer_content_adaptive):
        raise ValueError(
            f"tokenizer_family={family!r} requires model.tokenizer_content_adaptive=false."
        )


def prune_state_dict_for_tokenizer_family(
    state_dict: dict[str, object],
    *,
    tokenizer_family: str,
    contract_family: str = "legacy",
    style_injection_mode: str = "",
    proximal_mode: str = "",
    style_delta_mode: str = "",
    output_appearance_alignment_mode: str = "",
) -> tuple[dict[str, object], list[str]]:
    prefixes = compat_state_strip_prefixes_for_model_contract(
        tokenizer_family=tokenizer_family,
        contract_family=contract_family,
        style_injection_mode=style_injection_mode,
        proximal_mode=proximal_mode,
        style_delta_mode=style_delta_mode,
        output_appearance_alignment_mode=output_appearance_alignment_mode,
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
    bridge_noise_schedule: str = "auto",
) -> None:
    solver = normalize_family(solver_family, allowed=SOLVER_FAMILIES, default="euler_legacy")
    transport = normalize_transport_prediction_mode(transport_prediction_mode)
    objective = str(objective_mode or "").strip().lower()
    loss = str(loss_type or "").strip().lower()
    exact_schedule = resolves_exact_brownian_schedule(
        bridge_noise_schedule=bridge_noise_schedule,
        objective_mode=objective,
    )
    if solver == "solver_i2sb" and transport != "endpoint":
        raise ValueError("solver_i2sb requires model.transport_prediction_mode='endpoint'.")
    if solver == "solver_i2sb" and not is_i2sb_objective_mode(objective):
        raise ValueError("solver_i2sb requires bridge.objective_mode='i2sb_endpoint'.")
    if is_i2sb_objective_mode(objective_mode) and transport != "endpoint":
        raise ValueError("bridge.objective_mode='i2sb_endpoint' requires model.transport_prediction_mode='endpoint'.")
    if (solver == "solver_i2sb" or is_i2sb_objective_mode(objective)) and loss and loss != "mse":
        raise ValueError("true I2SB requires bridge.loss_type='mse'.")
    if (solver == "solver_i2sb" or is_i2sb_objective_mode(objective)) and not exact_schedule:
        schedule = normalize_bridge_noise_schedule(bridge_noise_schedule)
        raise ValueError(
            "true I2SB requires bridge.bridge_noise_schedule='exact_brownian' "
            "or 'auto' resolving to the exact Brownian bridge; "
            f"got bridge_noise_schedule={schedule!r} objective_mode={objective!r}."
        )


def validate_phase616_clean_contract(
    *,
    contract_family: str,
    output_appearance_alignment_mode: str = "",
    proximal_mode: str = "",
    style_delta_mode: str = "",
    solver_corrector_mode: str = "",
    cycle_consistency_weight: float = 0.0,
    w_content_lowpass_anchor: float = 0.0,
    w_content_edge_anchor: float = 0.0,
    proximal_trust_ratio: float = 0.0,
    proximal_trust_weight: float = 0.0,
    full_eval_postprocess_mode: str = "",
    full_eval_latent_postprocess_mode: str = "",
    pre_integrate_moment_match: bool = False,
    output_moment_match: bool = False,
) -> None:
    family = normalize_contract_family(contract_family)
    if family != "phase616":
        return
    appearance = str(output_appearance_alignment_mode or "").strip().lower()
    if appearance not in {"", "none"}:
        raise ValueError(
            "model.contract_family='phase616' requires model.output_appearance_alignment_mode='none'."
        )
    prox = str(proximal_mode or "").strip().lower()
    if prox not in {"", "off"}:
        raise ValueError(
            "model.contract_family='phase616' requires model.proximal_mode='off'."
        )
    delta = str(style_delta_mode or "").strip().lower()
    if delta not in {"", "none"}:
        raise ValueError(
            "model.contract_family='phase616' requires model.style_delta_mode='none'."
        )
    solver_corrector = str(solver_corrector_mode or "").strip().lower()
    if solver_corrector not in {"", "none"}:
        raise ValueError(
            "model.contract_family='phase616' requires model.solver_corrector_mode='none'."
        )
    if float(cycle_consistency_weight) > 0.0:
        raise ValueError(
            "model.contract_family='phase616' requires bridge.cycle_consistency_weight=0.0."
        )
    if float(w_content_lowpass_anchor) > 0.0 or float(w_content_edge_anchor) > 0.0:
        raise ValueError(
            "model.contract_family='phase616' requires content anchor losses to stay off."
        )
    if float(proximal_trust_ratio) > 0.0 or float(proximal_trust_weight) > 0.0:
        raise ValueError(
            "model.contract_family='phase616' requires bridge.proximal_trust_ratio/weight=0.0."
        )
    full_eval_post = str(full_eval_postprocess_mode or "").strip().lower()
    if full_eval_post not in {"", "none"}:
        raise ValueError(
            "model.contract_family='phase616' requires full-eval RGB postprocess to stay off."
        )
    full_eval_latent_post = str(full_eval_latent_postprocess_mode or "").strip().lower()
    if full_eval_latent_post not in {"", "none"}:
        raise ValueError(
            "model.contract_family='phase616' requires full-eval latent postprocess to stay off."
        )
    if bool(pre_integrate_moment_match):
        raise ValueError(
            "model.contract_family='phase616' requires model.pre_integrate_moment_match=false."
        )
    if bool(output_moment_match):
        raise ValueError(
            "model.contract_family='phase616' requires model.output_moment_match=false."
        )
