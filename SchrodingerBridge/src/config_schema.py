from __future__ import annotations

import copy
import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from style_families import validate_i2sb_contract, validate_pure_latent_contract


INFERENCE_DEFAULTS: dict[str, dict[str, Any]] = {
    "inference": {
        "num_steps": 12,
        "step_size": 1.0,
        "style_strength": 1.0,
    },
    "full_eval": {
        "num_steps": 12,
        "step_size": 1.0,
        "style_strength": 1.0,
        "batch_size": 8,
        "generation_batch_size": 0,
        "metric_batch_size": 0,
        "max_src_samples": 30,
        "max_ref_compare": 24,
        "max_ref_cache": 80,
        "ref_feature_batch_size": 8,
        "target_chunk_size": 2,
        "vae_decode_batch_size": 16,
        "vae_compile_decoder": False,
        "vae_compile_method": "pt2",
        "vae_compile_mode": "reduce-overhead",
        "vae_compile_fullgraph": False,
        "vae_compile_cache_dir": "",
        "only_lpips_clip_style": True,
        "transfer_only": False,
        "postprocess_mode": "none",
        "postprocess_strength": 0.0,
        "postprocess_mean_strength": 1.0,
        "postprocess_std_strength": 1.0,
        "postprocess_ref_limit": 64,
        "allow_metric_postprocess": False,
        "latent_postprocess_mode": "none",
        "latent_postprocess_strength": 0.0,
        "latent_postprocess_mean_strength": 1.0,
        "latent_postprocess_std_strength": 1.0,
        "latent_postprocess_ref_limit": 64,
        "enable_introstyle": False,
        "introstyle_style_bank_root": "",
        "introstyle_model_id": "",
        "introstyle_modelscope_id": "stabilityai/stable-diffusion-2-1-base",
        "introstyle_modelscope_cache_dir": "",
        "introstyle_allow_network": False,
        "introstyle_bank_limit_per_style": 64,
        "introstyle_batch_size": 4,
        "introstyle_topk": 8,
        "introstyle_t": 25,
        "introstyle_up_ft_index": 1,
        "introstyle_ensemble_size": 1,
        "save_generated_images": False,
        "save_summary_grid": False,
        "keep_generated_on_device": True,
        "delta_observability": False,
        "source_latent_cache": False,
        "lpips_chunk_size": 4,
        "in_process": False,
    },
}


def _section_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return dict(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                key: item
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
        except Exception:
            pass
    try:
        return dict(value)
    except Exception:
        return {}


def _split_known_fields(cls: type[Any], payload: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _section_dict(payload)
    known_names = {item.name for item in fields(cls) if item.init and item.name != "extra"}
    known = {key: data[key] for key in known_names if key in data}
    extra = {key: value for key, value in data.items() if key not in known_names}
    return known, extra


def _materialize_missing_dataclass_fields(obj: Any) -> None:
    for item in fields(obj):
        name = str(item.name)
        if hasattr(obj, name):
            continue
        if item.default is not MISSING:
            value = copy.deepcopy(item.default)
        elif item.default_factory is not MISSING:  # type: ignore[attr-defined]
            value = item.default_factory()  # type: ignore[misc]
        else:
            value = None
        setattr(obj, name, value)


def _rehydrate_extra_attributes(obj: Any) -> None:
    extra = getattr(obj, "extra", None)
    if not isinstance(extra, Mapping):
        return
    for key, value in extra.items():
        name = str(key)
        if hasattr(obj, name):
            continue
        setattr(obj, name, value)


def _normalize_model_contract_defaults(cfg: "ModelConfig") -> "ModelConfig":
    family = str(getattr(cfg, "tokenizer_family", "legacy_factorized") or "legacy_factorized").strip().lower()
    if family == "pure_latent_spatial":
        cfg.style_tokenizer = "null"
        cfg.tokenizer_content_adaptive = False
        cfg.style_spatial_mode = "disabled"
        cfg.style_id_spatial_jitter_px = 0
    if bool(getattr(cfg, "ablation_disable_spatial_prior", False)):
        cfg.style_spatial_mode = "disabled"
        cfg.style_id_spatial_jitter_px = 0
    return cfg


_RETIRED_BRIDGE_KEYS = {
    "w_color",
    "w_repulsive",
    "w_nce",
    "w_low_freq",
    "w_cycle",
    "nce_num_patches",
    "nce_temperature",
    "low_freq_kernel_size",
    "omf_color_patch_size",
    "color_transport_mode",
    "color_gumbel_tau",
    "kinetic_entropy_gate_weight",
    "repulsive_pool_size",
    "repulsive_temperature",
    "w_content_anchor",
    "w_edge_anchor",
    "w_semantic_entropy",
    "semantic_entropy_target",
    "w_divergence",
    "divergence_samples",
    "w_feature_riemannian",
    "w_kantorovich",
    "kantorovich_steps",
    "kantorovich_lr",
    "kantorovich_channels",
    "w_nonlocal_structure",
    "nonlocal_structure_pool",
    "w_phase_separation",
    "phase_gradient_weight",
    "w_fourier_phase_lock",
    "fourier_phase_lock_highpass",
    "w_head_color_tv",
    "w_head_color_energy",
    "w_head_amp_energy",
    "w_warp_curl_reward",
}


@dataclass
class ModelConfig:
    latent_channels: int = 4
    num_styles: int = 5
    style_dim: int = 160
    style_tokenizer: str = "factorized"
    tokenizer_family: str = "legacy_factorized"
    backbone_attention_family: str = "legacy_semantic_crossattn"
    solver_family: str = "euler_legacy"
    tokenizer_identity_dim: int = 24
    tokenizer_texture_dim: int = 32
    tokenizer_geometry_dim: int = 24
    tokenizer_projection_mode: str = "concat"
    tokenizer_residual_gain: float = 0.5
    tokenizer_init_std: float = 0.02
    tokenizer_num_atoms: int = 32
    tokenizer_num_prototypes: int = 4
    tokenizer_atom_temperature: float = 0.25
    tokenizer_field_dropout_p: float = 0.0
    tokenizer_code_l2_norm: bool = False
    tokenizer_code_scale: float = 1.0
    tokenizer_atom_topk: int = 0
    tokenizer_atom_hard_eval: bool = False
    tokenizer_content_adaptive: bool = False
    tokenizer_content_hidden_dim: int = 64
    tokenizer_content_gain: float = 0.5
    tokenizer_content_stopgrad: bool = True
    tokenizer_content_style_gate: bool = False
    tokenizer_content_style_gate_max: float = 2.0
    tokenizer_content_style_gate_init: float = 1.0
    tokenizer_latent_init_mode: str = "none"
    tokenizer_latent_init_cache_dir: str = ""
    tokenizer_latent_init_scale: float = 0.2
    tokenizer_latent_init_pool_size: int = 4
    tokenizer_latent_init_kmeans_iters: int = 8
    tokenizer_latent_init_sample_limit_per_style: int = 1000
    tokenizer_dino_dim: int = 384
    tokenizer_num_clusters: int = 16
    tokenizer_query_dim: int = 64
    tokenizer_query_num_blocks: int = 4
    tokenizer_spatial_dim: int = 0
    smoe_translation_rank: int = 0
    tokenizer_pe_temperature: float = 1.0
    tokenizer_global_gate_hidden_dim: int = 160
    tokenizer_global_gate_scale: float = 1.0
    tokenizer_prompt_dim: int = 256
    tokenizer_prompt_length: int = 8
    tokenizer_structured_temperature: float = 0.1
    time_dim: int = 256
    base_dim: int = 64
    lift_channels: int | None = None
    num_hires_blocks: int = 2
    num_res_blocks: int = 4
    num_decoder_blocks: int = 2
    num_groups: int = 4
    latent_scale_factor: float = 0.18215
    residual_gain: float = 1.0
    style_spatial_pre_gain_16: float = 0.35
    style_spatial_mode: str = "class"
    style_spatial_num_prototypes: int = 4
    style_spatial_routing_temperature: float = 0.25
    style_spatial_content_hidden_dim: int = 64
    style_strength_default: float = 1.0
    style_strength_max: float = 1.0
    allow_style_overdrive: bool = False
    style_strength_step_curve: str = "linear"
    upsample_mode: str = "nearest"
    style_id_spatial_jitter_px: int = 0
    upsample_blur: bool = True
    upsample_blur_kernel: str = "box3"
    style_attn_num_tokens: int = 128
    style_attn_num_heads: int = 4
    style_attn_sharpen_scale: float = 2.5
    style_attn_temperature: float = 0.08
    hires_block_type: str = "conv"
    body_block_type: str = "global_attn"
    decoder_block_type: str = "conv"
    semantic_attn_temperature: float = 0.08
    semantic_attn_routing_mode: str = "softmax"
    semantic_sinkhorn_iters: int = 3
    semantic_gumbel_tau: float = 1.0
    semantic_self_topology_gate: bool = False
    semantic_self_topology_blend: float = 1.0
    semantic_gw_spatial_lambda: float = 0.25
    velocity_head_mode: str = "identity"
    velocity_tanh_limit: float = 20.0
    transport_prediction_mode: str = "velocity"
    transport_endpoint_scale: float = 4.0
    endpoint_parameterization: str = "absolute"
    endpoint_residual_blend: float = 0.0
    endpoint_orthogonal_kernel: int = 5
    endpoint_orthogonal_high_scale: float = 1.0
    endpoint_orthogonal_low_anchor: float = 1.0
    endpoint_orthogonal_low_mode: str = "all"
    feature_attn_num_heads: int = 4
    window_attn_window_size: int = 8
    skip_fusion_mode: str = "add_proj"
    skip_routing_mode: str = "none"
    skip_naive_gain: float = 1.0
    skip_residual_weight: float = 0.1
    style_skip_content_retention_boost: float = 0.0
    input_anchor_noise_std: float = 0.0
    input_anchor_noise_eval: bool = False
    ablation_no_residual: bool = False
    ablation_no_residual_gain: float = 1.0
    ablation_disable_spatial_prior: bool = False
    ablation_direct_delta_blend: bool = False
    raw_latent_splat_highway: bool = False
    ablation_skip_clean: bool = True
    ablation_skip_blur: bool = True
    skip_bottleneck_channels: int = 16
    skip_spatial_dropout_p: float = 0.15
    ablation_decoder_highpass: bool = True
    color_highway_gain: float = 1.0
    use_diffeomorphic_stroke: bool = False
    dynamic_style_operator_head: bool = False
    dynamic_style_operator_hidden_mult: float = 1.0
    zero_init_output_head: bool = False
    diffeomorphic_head_mode: str = "standard"
    diffeomorphic_color_strength: float = 0.85
    diffeomorphic_warp_strength: float = 0.08
    diffeomorphic_texture_gate_strength: float = 8.0
    diffeomorphic_normal_leak: float = 0.0
    diffeomorphic_color_lowpass_kernel: int = 1
    diffeomorphic_color_edge_gamma: float = 0.0
    diffeomorphic_amp_strength: float = 0.5
    diffeomorphic_factorized_enable_color: bool = True
    diffeomorphic_factorized_enable_amp: bool = True
    diffeomorphic_joint_bilateral_kernel: int = 1
    diffeomorphic_joint_bilateral_range_sigma: float = 0.5
    diffeomorphic_divergence_free_warp: bool = False
    diffeomorphic_metric_mask_gamma: float = 0.0
    diffeomorphic_metric_mask_smooth_kernel: int = 3
    diffeomorphic_metric_mask_use_z0: bool = False
    latent_canvas_strength: float = 0.0
    latent_canvas_edge_gamma: float = 4.0
    latent_canvas_highpass_kernel: int = 5
    pre_integrate_moment_match: bool = False
    pre_integrate_moment_blend: float = 1.0
    output_moment_match: bool = False
    output_moment_match_eps: float = 1e-6
    output_moment_match_train_only: bool = False
    output_appearance_alignment_mode: str = "none"
    output_appearance_hidden_dim: int = 96
    output_appearance_log_scale_span: float = 0.22314355131420976
    output_appearance_shift_span: float = 0.35
    output_appearance_blend: float = 1.0
    output_appearance_use_spatial_stats: bool = True
    output_appearance_use_gate_mask_stats: bool = True
    proximal_mode: str = "off"
    proximal_hidden_channels: int = 64
    proximal_highpass_kernel: int = 5
    proximal_attn_routing_mode: str = "softmax"
    proximal_attn_sinkhorn_iters: int = 3
    proximal_attn_gumbel_tau: float = 1.0
    proximal_residual_energy_weight: float = 0.0
    proximal_trust_ratio: float = 0.0
    proximal_trust_weight: float = 0.0
    proximal_clamp_ratio: float = 0.0
    proximal_clamp_ratio_mid: float = 0.0
    proximal_clamp_ratio_end: float = 0.0
    proximal_clamp_schedule: str = "linear"
    proximal_clamp_hold_epochs: int = 0
    proximal_clamp_release_epochs: int = 0
    proximal_clamp_mid_hold_epochs: int = 0
    proximal_clamp_second_release_epochs: int = 0
    proximal_force_highpass: bool = True
    proximal_bind_terminal_losses: bool = True
    record_base_endpoint_metrics: bool = False
    endpoint_velocity_time_floor: float = 0.05
    execution_budget_mode: str = "none"
    execution_budget_hidden_dim: int = 64
    execution_budget_log_span: float = 0.22314355131420976
    style_injection_mode: str = "none"
    style_injection_form: str = "mixed"
    style_injection_hidden_dim: int = 64
    style_injection_scale: float = 1.0
    style_injection_gate_log_span: float = 0.4054651081081644
    style_injection_spatial_kernel: int = 5
    style_injection_force_highpass: bool = True
    style_delta_mode: str = "none"
    style_delta_rank: int = 4
    style_delta_hidden_dim: int = 64
    style_delta_scale: float = 0.15
    style_delta_highpass_kernel: int = 5
    style_delta_force_highpass: bool = True
    style_section_hidden_dim: int = 64
    style_section_scale: float = 0.10
    style_section_force_highpass: bool = True
    style_head_adapter_hidden_dim: int = 32
    style_head_adapter_scale: float = 0.10
    style_head_adapter_force_highpass: bool = False
    use_style_blender: bool = False
    solver_rk_order: int = 4
    solver_corrector_steps: int = 1
    solver_corrector_step_size: float = 0.1
    solver_corrector_mode: str = "none"
    solver_corrector_lowpass_kernel: int = 5
    solver_corrector_clamp: float = 0.0
    solver_tangent_projection_strength: float = 1.0
    solver_stochastic_noise_scale: float = 0.01
    solver_fiber_aligned: bool = False
    i2sb_fiber_aligned_noise: bool = False
    i2sb_fiber_noise_rms_normalize: bool = True
    i2sb_fiber_project_endpoint: bool = False
    i2sb_fiber_project_noise: bool = False
    i2sb_fiber_project_kernel: int = 5
    i2sb_fiber_project_use_gate: bool = False
    solver_dual_track_detach: bool = True
    use_checkpointing: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ModelConfig":
        known, extra = _split_known_fields(cls, payload)
        cfg = cls(**known)
        cfg.extra = extra
        cfg = _normalize_model_contract_defaults(cfg)
        _rehydrate_extra_attributes(cfg)
        return cfg

    def validated(self, *, use_checkpointing: bool | None = None) -> "ModelConfig":
        _materialize_missing_dataclass_fields(self)
        cfg = ModelConfig.from_mapping(self.to_dict())
        if use_checkpointing is not None:
            cfg.use_checkpointing = bool(use_checkpointing)
        if cfg.lift_channels is None:
            cfg.lift_channels = int(cfg.base_dim)
        _rehydrate_extra_attributes(cfg)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        _materialize_missing_dataclass_fields(self)
        payload = asdict(self)
        extra = payload.pop("extra", {})
        payload.update(extra)
        return payload


@dataclass
class BridgeConfig:
    objective_mode: str = "omf"
    loss_type: str = "omf"
    ot_cost_mode: str = "swd"
    t_min: float = 0.0
    t_max: float = 1.0
    identity_endpoint: bool = False
    eps: float = 1e-4
    coupling_solver: str = "sinkhorn"
    allow_cpu_hungarian: bool = False
    coupling_feature_mode: str = "latent"
    coupling_lowfreq_kernel: int = 9
    coupling_edge_weight: float = 0.0
    coupling_target_mode: str = "sample"
    coupling_barycentric_topk: int = 0
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iters: int = 60
    sinkhorn_stabilize: bool = True
    bridge_sigma: float = 0.05
    bridge_noise_mode: str = "gaussian"
    bridge_noise_schedule: str = "auto"
    bridge_path_mode: str = "linear"
    bridge_path_slerp_eps: float = 1e-4
    i2sb_predictor_time_floor: float = 0.0
    bridge_noise_window_start: float = 0.18
    bridge_noise_window_end: float = 0.82
    bridge_style_noise_kernel: int = 5
    bridge_style_noise_flat_gamma: float = 0.0
    terminal_swd_weight: float = 0.1
    terminal_swd_aux_weight: float = 0.0
    semantic_supervision_family: str = "legacy_terminal_swd"
    dino_masked_swd_weight: float = 0.0
    w_variance_penalty: float = 0.0
    w_style_energy_floor: float = 0.0
    w_lowfreq_velocity: float = 0.0
    w_content_lowpass_anchor: float = 0.0
    w_content_edge_anchor: float = 0.0
    content_anchor_lowpass_kernel: int = 9
    w_style_contrastive: float = 0.0
    style_contrastive_temperature: float = 0.08
    style_contrastive_pool_size: int = 4
    w_residual_style_direction: float = 0.0
    w_generated_delta_diversity: float = 0.0
    generated_delta_diversity_margin: float = 0.0
    w_spectral_amplitude: float = 0.0
    spectral_amplitude_channels: int = 2
    spectral_amplitude_highpass: bool = True
    sb_noise_epsilon: float = 0.0
    retinex_target_blend: float = 0.0
    retinex_kernel_size: int = 15
    w_anisotropic_kinetic: float = 0.0
    anisotropic_normal_weight: float = 25.0
    anisotropic_tangent_weight: float = 0.25
    anisotropic_edge_gate_gamma: float = 0.0
    anisotropic_edge_gate_quantile: float = 0.0
    anisotropic_edge_gate_power: float = 1.0
    w_stokes_viscous: float = 0.0
    kinetic_penalty_mode: str = "global_l2"
    kinetic_lambda_low: float = 1.0
    kinetic_lambda_high: float = 0.02
    kinetic_lowpass_kernel: int = 5
    kinetic_spectral_cutoff: float = 12.0
    kinetic_manifold_gamma: float = 10.0
    structure_penalty_mode: str = "off"
    style_energy_floor_ratio: float = 0.6
    anchor_pool_size: int = 9
    terminal_num_steps: int = 4
    terminal_swd_on_identity: bool = False
    w_kinetic: float = 1.0
    w_flow: float = 0.0
    w_curvature: float = 0.0
    curvature_dt: float = 0.15
    kinetic_mode: str = "endpoint"
    kinetic_gate_exponent: float = 1.0
    semantic_swd_num_projections: int = 64
    terminal_swd_mode: str = "standard"
    terminal_swd_axis_source: str = "semantic"
    spectral_swd_low_weight: float = 1.0
    spectral_swd_high_weight: float = 1.0
    spectral_swd_low_kernel: int = 5
    semantic_quotient_bins: int = 4
    swd_distance_mode: str = "cdf"
    swd_use_high_freq: bool = True
    swd_patch_sizes: list[int] = field(default_factory=lambda: [1, 3, 5, 9])
    swd_num_projections: int = 64
    swd_projection_chunk_size: int = 32
    swd_cdf_num_bins: int = 32
    swd_cdf_tau: float = 0.01
    swd_cdf_sample_size: int = 256
    swd_cdf_bin_chunk_size: int = 4
    swd_cdf_sample_chunk_size: int = 128
    swd_hf_weight_ratio: float = 1.0
    swd_micro_patch_max: int = 3
    swd_macro_patch_min: int = 5
    swd_micro_weight: float = 1.0
    swd_macro_weight: float = 1.0
    swd_deterministic_subsample: bool = True
    swd_scale_invariant_patches: bool = False
    swd_adaptive_highpass: bool = False
    swd_highpass_kernel_size: int = 5
    swd_use_dilated_projections: bool = False
    swd_projection_dilation: int = 2
    target_teacher_mode: str = "off"
    target_teacher_decay: float = 0.99
    target_teacher_weight: float = 0.0
    cycle_consistency_weight: float = 0.0
    cycle_consistency_num_steps: int = 4
    normalize_eps: float = 1e-8
    logit_clamp: float = 50.0
    velocity_clamp: float = 20.0
    endpoint_clamp: float = 24.0
    proximal_target_weight: float = 0.0
    similarity_clamp: float = 50.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "BridgeConfig":
        known, extra = _split_known_fields(cls, payload)
        for key in _RETIRED_BRIDGE_KEYS:
            extra.pop(key, None)
        cfg = cls(**known)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict[str, Any]:
        _materialize_missing_dataclass_fields(self)
        payload = asdict(self)
        extra = payload.pop("extra", {})
        payload.update(extra)
        return payload


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    accumulation_steps: int = 1
    num_workers: int = 0
    shuffle: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    pin_memory: bool = True
    cpu_threads: int | None = None
    cpu_interop_threads: int | None = None
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    multistep_milestones: list[int] = field(default_factory=lambda: [40, 55])
    multistep_gamma: float = 0.1
    grad_clip_norm: float = 1.0
    num_epochs: int = 60
    save_interval: int = 10
    async_checkpoint_save: bool = False
    log_interval: int = 20
    use_tqdm: bool = True
    use_amp: bool = False
    amp_dtype: str = "bf16"
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    channels_last: bool = False
    torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool | None = None
    torch_compile_cache_dir: str = ""
    use_gradient_checkpointing: bool = False
    profile_modules: bool = False
    profile_sync_cuda: bool = False
    fused_adamw: bool = True
    resume_checkpoint: str = ""
    resume_optimizer: bool = True
    resume_model_strict: bool = True
    resume_ignore_prefixes: list[str] = field(default_factory=list)
    resume_include_prefixes: list[str] = field(default_factory=list)
    resume_training_state: bool = True
    resume_prefer_local_checkpoint: bool = True
    freeze_mode: str = "none"
    freeze_reinit_trainable: bool = False
    full_eval_batch_size: int = 8
    full_eval_output_subdir: str = "full_eval"
    full_eval_generation_batch_size: int | None = None
    full_eval_metric_batch_size: int | None = None
    full_eval_num_steps: int | None = None
    full_eval_step_size: float | None = None
    full_eval_style_strength: float | None = None
    full_eval_max_src_samples: int | None = None
    full_eval_max_ref_compare: int | None = None
    full_eval_max_ref_cache: int | None = None
    full_eval_ref_feature_batch_size: int | None = None
    full_eval_target_chunk_size: int | None = 2
    full_eval_vae_decode_batch_size: int | None = 16
    full_eval_vae_compile_decoder: bool = False
    full_eval_vae_compile_method: str = "pt2"
    full_eval_vae_compile_mode: str = "reduce-overhead"
    full_eval_vae_compile_fullgraph: bool = False
    full_eval_vae_compile_cache_dir: str = ""
    full_eval_vae_onnx_decoder: str = ""
    full_eval_vae_onnx_tensorrt: bool = False
    full_eval_vae_onnx_trt_cache_dir: str = ""
    full_eval_skip_diffusers_vae_when_onnx: bool = True
    full_eval_only_lpips_clip_style: bool | None = None
    full_eval_transfer_only: bool = False
    full_eval_postprocess_mode: str = "none"
    full_eval_postprocess_strength: float = 0.0
    full_eval_postprocess_mean_strength: float = 1.0
    full_eval_postprocess_std_strength: float = 1.0
    full_eval_postprocess_ref_limit: int = 64
    full_eval_allow_metric_postprocess: bool = False
    full_eval_latent_postprocess_mode: str = "none"
    full_eval_latent_postprocess_strength: float = 0.0
    full_eval_latent_postprocess_mean_strength: float = 1.0
    full_eval_latent_postprocess_std_strength: float = 1.0
    full_eval_latent_postprocess_ref_limit: int = 64
    full_eval_enable_introstyle: bool = False
    full_eval_introstyle_style_bank_root: str = ""
    full_eval_introstyle_model_id: str = ""
    full_eval_introstyle_modelscope_id: str = "stabilityai/stable-diffusion-2-1-base"
    full_eval_introstyle_modelscope_cache_dir: str = ""
    full_eval_introstyle_allow_network: bool = False
    full_eval_introstyle_bank_limit_per_style: int = 64
    full_eval_introstyle_batch_size: int = 4
    full_eval_introstyle_topk: int = 8
    full_eval_introstyle_t: int = 25
    full_eval_introstyle_up_ft_index: int = 1
    full_eval_introstyle_ensemble_size: int = 1
    full_eval_save_generated_images: bool | None = False
    full_eval_save_summary_grid: bool | None = False
    full_eval_keep_generated_on_device: bool = True
    full_eval_delta_observability: bool = False
    full_eval_source_latent_cache: bool = False
    full_eval_lpips_chunk_size: int = 4
    full_eval_in_process: bool = False
    full_eval_runtime_model_cache: bool = False
    full_eval_each_epoch: bool = False
    full_eval_defer_until_training_end: bool = False
    full_eval_force_regen: bool = False
    full_eval_profile_timing: bool = False
    test_image_dir: str = "../style_data/overfit50"
    full_eval_cache_dir: str = "../eval_cache"
    full_eval_clip_hf_cache_dir: str = "../eval_cache/hf"
    full_eval_clip_backend: str = "hf"
    full_eval_hf_clip_skip_processor: bool = False
    full_eval_disable_lpips: bool = False
    full_eval_enable_art_fid: bool = False
    full_eval_enable_kid: bool = False
    numeric_debug: bool = False
    numeric_debug_interval: int = 10
    numeric_debug_halt_on_nonfinite: bool = True
    numeric_debug_dump_limit: int = 200
    distill: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "TrainingConfig":
        known, extra = _split_known_fields(cls, payload)
        cfg = cls(**known)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict[str, Any]:
        _materialize_missing_dataclass_fields(self)
        payload = asdict(self)
        extra = payload.pop("extra", {})
        payload.update(extra)
        return payload


@dataclass
class DataConfig:
    data_root: str = "../latent-256"
    style_subdirs: list[str] = field(default_factory=lambda: ["photo", "Hayao", "monet", "vangogh", "cezanne"])
    allow_hflip: bool = True
    identity_ratio: float | None = None
    balance_target_styles_per_batch: bool = True
    preload_to_gpu: bool = False
    preload_max_vram_gb: float = 0.0
    preload_reserve_ratio: float = 0.35
    virtual_length_multiplier: float = 1.0
    content_style_sampling_weights: list[float] | None = None
    target_style_sampling_weights: list[float] | None = None
    pairing_cache_path: str = ""
    pairing_cache_topk: int = 4
    pairing_cache_active_topk: int = 0
    pairing_cache_sample_mode: str = "uniform_topk"
    pairing_cache_rank_schedule: str = "fixed"
    pairing_cache_min_topk: int = 1
    pairing_cache_curriculum_epochs: int = 0
    pairing_cache_rank_power: float = 1.0
    pairing_cache_explore_prob: float = 0.0
    pairing_cache_explore_topk: int = 0
    pairing_cache_dual_target_mix: float = 0.0
    pairing_cache_dual_target_topk: int = 0
    pairing_cache_aux_target_topk: int = 0
    pairing_cache_cross_only: bool = True
    latent_cache_mode: str = "off"
    latent_cache_dir: str = ""
    dino_cache_path: str = ""
    dino_cache_required: bool = False
    dino_bank_limit_per_style: int = 8
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DataConfig":
        known, extra = _split_known_fields(cls, payload)
        cfg = cls(**known)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict[str, Any]:
        _materialize_missing_dataclass_fields(self)
        payload = asdict(self)
        extra = payload.pop("extra", {})
        payload.update(extra)
        return payload


@dataclass
class CheckpointConfig:
    save_dir: str = "./artifacts"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CheckpointConfig":
        known, extra = _split_known_fields(cls, payload)
        cfg = cls(**known)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict[str, Any]:
        _materialize_missing_dataclass_fields(self)
        payload = asdict(self)
        extra = payload.pop("extra", {})
        payload.update(extra)
        return payload


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    inference: dict[str, Any] = field(default_factory=dict)
    full_eval: dict[str, Any] = field(default_factory=dict)
    ablation: dict[str, Any] = field(default_factory=dict)
    extra_sections: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ExperimentConfig":
        data = _section_dict(payload)
        known_sections = {"model", "bridge", "training", "data", "checkpoint", "inference", "full_eval", "ablation"}
        cfg = cls(
            model=ModelConfig.from_mapping(data.get("model")),
            bridge=BridgeConfig.from_mapping(data.get("bridge")),
            training=TrainingConfig.from_mapping(data.get("training")),
            data=DataConfig.from_mapping(data.get("data")),
            checkpoint=CheckpointConfig.from_mapping(data.get("checkpoint")),
            inference=_section_dict(data.get("inference")),
            full_eval=_section_dict(data.get("full_eval")),
            ablation=_section_dict(data.get("ablation")),
            extra_sections={key: value for key, value in data.items() if key not in known_sections},
        )
        validate_i2sb_contract(
            solver_family=str(getattr(cfg.model, "solver_family", "euler_legacy")),
            transport_prediction_mode=str(getattr(cfg.model, "transport_prediction_mode", "velocity")),
            objective_mode=str(getattr(cfg.bridge, "objective_mode", "")),
            loss_type=str(getattr(cfg.bridge, "loss_type", "")),
            bridge_noise_schedule=str(getattr(cfg.bridge, "bridge_noise_schedule", "auto")),
        )
        validate_pure_latent_contract(
            tokenizer_family=str(getattr(cfg.model, "tokenizer_family", "legacy_factorized")),
            style_tokenizer=str(getattr(cfg.model, "style_tokenizer", "")),
            semantic_supervision_family=str(getattr(cfg.bridge, "semantic_supervision_family", "legacy_terminal_swd")),
            dino_masked_swd_weight=float(getattr(cfg.bridge, "dino_masked_swd_weight", 0.0)),
            style_spatial_mode=str(getattr(cfg.model, "style_spatial_mode", "")),
            tokenizer_content_adaptive=bool(getattr(cfg.model, "tokenizer_content_adaptive", False)),
        )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model": self.model.to_dict(),
            "bridge": self.bridge.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
        }
        if self.inference:
            payload["inference"] = dict(self.inference)
        if self.full_eval:
            payload["full_eval"] = dict(self.full_eval)
        if self.ablation:
            payload["ablation"] = dict(self.ablation)
        payload.update(self.extra_sections)
        return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_config_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(base, override)


def load_config(config_path: str | Path, *, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        chain = " -> ".join(str(p) for p in [*seen, path])
        raise ValueError(f"Config inheritance cycle detected: {chain}")
    seen.add(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    base_ref = raw.get("_base")
    if not base_ref:
        return raw

    base_paths = base_ref if isinstance(base_ref, list) else [base_ref]
    merged: dict[str, Any] = {}
    for item in base_paths:
        base_path = Path(item)
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        merged = _deep_merge(merged, load_config(base_path, _seen=seen.copy()))

    return _deep_merge(merged, raw)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(load_config(config_path))


def _config_dict(config: dict[str, Any] | ExperimentConfig | None) -> dict[str, Any]:
    if isinstance(config, ExperimentConfig):
        return config.to_dict()
    if isinstance(config, dict):
        return config
    return {}


@lru_cache(maxsize=1)
def load_inference_defaults() -> dict[str, Any]:
    return copy.deepcopy(INFERENCE_DEFAULTS)


def resolve_inference_section(config: dict[str, Any] | ExperimentConfig | None) -> dict[str, Any]:
    defaults = dict(load_inference_defaults().get("inference", {}) or {})
    config_dict = _config_dict(config)
    local = config_dict.get("inference", {}) or {}
    if isinstance(local, dict):
        defaults.update(local)
    return defaults


def resolve_full_eval_section(config: dict[str, Any] | ExperimentConfig | None) -> dict[str, Any]:
    defaults = dict(load_inference_defaults().get("full_eval", {}) or {})
    config_dict = _config_dict(config)
    training = config_dict.get("training", {}) or {}
    if isinstance(training, dict):
        mapping = {
            "num_steps": "full_eval_num_steps",
            "step_size": "full_eval_step_size",
            "style_strength": "full_eval_style_strength",
            "batch_size": "full_eval_batch_size",
            "output_subdir": "full_eval_output_subdir",
            "generation_batch_size": "full_eval_generation_batch_size",
            "metric_batch_size": "full_eval_metric_batch_size",
            "max_src_samples": "full_eval_max_src_samples",
            "max_ref_compare": "full_eval_max_ref_compare",
            "max_ref_cache": "full_eval_max_ref_cache",
            "ref_feature_batch_size": "full_eval_ref_feature_batch_size",
            "target_chunk_size": "full_eval_target_chunk_size",
            "vae_decode_batch_size": "full_eval_vae_decode_batch_size",
            "vae_compile_decoder": "full_eval_vae_compile_decoder",
            "vae_compile_method": "full_eval_vae_compile_method",
            "vae_compile_mode": "full_eval_vae_compile_mode",
            "vae_compile_fullgraph": "full_eval_vae_compile_fullgraph",
            "vae_compile_cache_dir": "full_eval_vae_compile_cache_dir",
            "vae_onnx_decoder": "full_eval_vae_onnx_decoder",
            "vae_onnx_tensorrt": "full_eval_vae_onnx_tensorrt",
            "vae_onnx_trt_cache_dir": "full_eval_vae_onnx_trt_cache_dir",
            "skip_diffusers_vae_when_onnx": "full_eval_skip_diffusers_vae_when_onnx",
            "only_lpips_clip_style": "full_eval_only_lpips_clip_style",
            "transfer_only": "full_eval_transfer_only",
            "hf_clip_skip_processor": "full_eval_hf_clip_skip_processor",
            "postprocess_mode": "full_eval_postprocess_mode",
            "postprocess_strength": "full_eval_postprocess_strength",
            "postprocess_mean_strength": "full_eval_postprocess_mean_strength",
            "postprocess_std_strength": "full_eval_postprocess_std_strength",
            "postprocess_ref_limit": "full_eval_postprocess_ref_limit",
            "allow_metric_postprocess": "full_eval_allow_metric_postprocess",
            "latent_postprocess_mode": "full_eval_latent_postprocess_mode",
            "latent_postprocess_strength": "full_eval_latent_postprocess_strength",
            "latent_postprocess_mean_strength": "full_eval_latent_postprocess_mean_strength",
            "latent_postprocess_std_strength": "full_eval_latent_postprocess_std_strength",
            "latent_postprocess_ref_limit": "full_eval_latent_postprocess_ref_limit",
            "enable_introstyle": "full_eval_enable_introstyle",
            "introstyle_style_bank_root": "full_eval_introstyle_style_bank_root",
            "introstyle_model_id": "full_eval_introstyle_model_id",
            "introstyle_modelscope_id": "full_eval_introstyle_modelscope_id",
            "introstyle_modelscope_cache_dir": "full_eval_introstyle_modelscope_cache_dir",
            "introstyle_allow_network": "full_eval_introstyle_allow_network",
            "introstyle_bank_limit_per_style": "full_eval_introstyle_bank_limit_per_style",
            "introstyle_batch_size": "full_eval_introstyle_batch_size",
            "introstyle_topk": "full_eval_introstyle_topk",
            "introstyle_t": "full_eval_introstyle_t",
            "introstyle_up_ft_index": "full_eval_introstyle_up_ft_index",
            "introstyle_ensemble_size": "full_eval_introstyle_ensemble_size",
            "save_generated_images": "full_eval_save_generated_images",
            "save_summary_grid": "full_eval_save_summary_grid",
            "keep_generated_on_device": "full_eval_keep_generated_on_device",
            "delta_observability": "full_eval_delta_observability",
            "source_latent_cache": "full_eval_source_latent_cache",
            "lpips_chunk_size": "full_eval_lpips_chunk_size",
            "in_process": "full_eval_in_process",
            "runtime_model_cache": "full_eval_runtime_model_cache",
        }
        for dst_key, src_key in mapping.items():
            if src_key in training and training.get(src_key) is not None:
                defaults[dst_key] = training.get(src_key)
    local = config_dict.get("full_eval", {}) or {}
    if isinstance(local, dict):
        defaults.update(local)
    return defaults


def compact_runtime_config(config: dict[str, Any] | ExperimentConfig | None) -> dict[str, Any]:
    config_dict = _config_dict(config)
    if not config_dict:
        return {}

    compact = copy.deepcopy(config_dict)
    infer_defaults = dict(load_inference_defaults().get("inference", {}) or {})
    full_eval_defaults = dict(load_inference_defaults().get("full_eval", {}) or {})

    infer_local = compact.get("inference")
    if isinstance(infer_local, dict):
        pruned_infer = {k: v for k, v in infer_local.items() if infer_defaults.get(k) != v}
        if pruned_infer:
            compact["inference"] = pruned_infer
        else:
            compact.pop("inference", None)

    full_eval_local = compact.get("full_eval")
    if isinstance(full_eval_local, dict):
        pruned_full_eval = {k: v for k, v in full_eval_local.items() if full_eval_defaults.get(k) != v}
        if pruned_full_eval:
            compact["full_eval"] = pruned_full_eval
        else:
            compact.pop("full_eval", None)

    training = compact.get("training")
    if isinstance(training, dict):
        mapping = {
            "full_eval_num_steps": "num_steps",
            "full_eval_step_size": "step_size",
            "full_eval_style_strength": "style_strength",
            "full_eval_batch_size": "batch_size",
            "full_eval_output_subdir": "output_subdir",
            "full_eval_generation_batch_size": "generation_batch_size",
            "full_eval_metric_batch_size": "metric_batch_size",
            "full_eval_max_src_samples": "max_src_samples",
            "full_eval_max_ref_compare": "max_ref_compare",
            "full_eval_max_ref_cache": "max_ref_cache",
            "full_eval_ref_feature_batch_size": "ref_feature_batch_size",
            "full_eval_target_chunk_size": "target_chunk_size",
            "full_eval_vae_decode_batch_size": "vae_decode_batch_size",
            "full_eval_vae_compile_decoder": "vae_compile_decoder",
            "full_eval_vae_compile_method": "vae_compile_method",
            "full_eval_vae_compile_mode": "vae_compile_mode",
            "full_eval_vae_compile_fullgraph": "vae_compile_fullgraph",
            "full_eval_vae_compile_cache_dir": "vae_compile_cache_dir",
            "full_eval_vae_onnx_decoder": "vae_onnx_decoder",
            "full_eval_vae_onnx_tensorrt": "vae_onnx_tensorrt",
            "full_eval_vae_onnx_trt_cache_dir": "vae_onnx_trt_cache_dir",
            "full_eval_skip_diffusers_vae_when_onnx": "skip_diffusers_vae_when_onnx",
            "full_eval_only_lpips_clip_style": "only_lpips_clip_style",
            "full_eval_transfer_only": "transfer_only",
            "full_eval_postprocess_mode": "postprocess_mode",
            "full_eval_postprocess_strength": "postprocess_strength",
            "full_eval_postprocess_mean_strength": "postprocess_mean_strength",
            "full_eval_postprocess_std_strength": "postprocess_std_strength",
            "full_eval_postprocess_ref_limit": "postprocess_ref_limit",
            "full_eval_allow_metric_postprocess": "allow_metric_postprocess",
            "full_eval_latent_postprocess_mode": "latent_postprocess_mode",
            "full_eval_latent_postprocess_strength": "latent_postprocess_strength",
            "full_eval_latent_postprocess_mean_strength": "latent_postprocess_mean_strength",
            "full_eval_latent_postprocess_std_strength": "latent_postprocess_std_strength",
            "full_eval_latent_postprocess_ref_limit": "latent_postprocess_ref_limit",
            "full_eval_enable_introstyle": "enable_introstyle",
            "full_eval_introstyle_style_bank_root": "introstyle_style_bank_root",
            "full_eval_introstyle_model_id": "introstyle_model_id",
            "full_eval_introstyle_modelscope_id": "introstyle_modelscope_id",
            "full_eval_introstyle_modelscope_cache_dir": "introstyle_modelscope_cache_dir",
            "full_eval_introstyle_allow_network": "introstyle_allow_network",
            "full_eval_introstyle_bank_limit_per_style": "introstyle_bank_limit_per_style",
            "full_eval_introstyle_batch_size": "introstyle_batch_size",
            "full_eval_introstyle_topk": "introstyle_topk",
            "full_eval_introstyle_t": "introstyle_t",
            "full_eval_introstyle_up_ft_index": "up_ft_index",
            "full_eval_introstyle_ensemble_size": "ensemble_size",
            "full_eval_save_generated_images": "save_generated_images",
            "full_eval_save_summary_grid": "save_summary_grid",
            "full_eval_keep_generated_on_device": "keep_generated_on_device",
            "full_eval_delta_observability": "delta_observability",
            "full_eval_source_latent_cache": "source_latent_cache",
            "full_eval_lpips_chunk_size": "lpips_chunk_size",
            "full_eval_runtime_model_cache": "runtime_model_cache",
        }
        for train_key, default_key in mapping.items():
            if train_key in training and full_eval_defaults.get(default_key) == training.get(train_key):
                training.pop(train_key, None)

    return compact
