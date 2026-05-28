from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


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
        "batch_size": 2,
        "max_src_samples": 30,
        "max_ref_compare": 24,
        "max_ref_cache": 80,
        "ref_feature_batch_size": 8,
    },
}


def _section_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _split_known_fields(cls: type[Any], payload: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _section_dict(payload)
    known_names = {item.name for item in fields(cls) if item.init and item.name != "extra"}
    known = {key: data[key] for key in known_names if key in data}
    extra = {key: value for key, value in data.items() if key not in known_names}
    return known, extra


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
}


@dataclass
class ModelConfig:
    latent_channels: int = 4
    num_styles: int = 5
    style_token_identity_dim: int = 16
    style_token_grammar_dim: int = 9
    style_token_band_dim: int = 3
    style_token_band_gain_scale: float = 0.35
    style_token_learn_identity: bool = False
    style_token_flatten_strength: float = 0.0
    style_token_flatten_kernel: int = 5
    style_token_adain_gate_enable: bool = False
    style_token_reader_enable: bool = False
    style_token_reader_hidden: int = 32
    style_token_reader_scale: float = 0.20
    style_token_grammar_texture_enable: bool = False
    style_token_grammar_texture_scale: float = 0.35
    style_token_texton_carrier_enable: bool = False
    style_token_texton_carrier_strength: float = 0.12
    style_token_texton_carrier_hidden_mult: float = 0.75
    style_token_texton_carrier_tanh_scale: float = 0.45
    style_token_prototype_carrier_enable: bool = False
    style_token_prototype_carrier_strength: float = 0.16
    style_token_prototype_carrier_hidden_mult: float = 0.75
    style_token_prototype_carrier_tanh_scale: float = 0.45
    style_token_depthwise_filter_enable: bool = False
    style_token_depthwise_filter_strength: float = 0.0
    style_token_depthwise_filter_tanh_scale: float = 0.35
    style_token_depthwise_filter_basis_offset: int = 8
    style_token_depthwise_filter_learnable_gate: bool = False
    style_token_depthwise_filter_learnable_gate_scale: float = 0.5
    style_token_depthwise_filter_style_basis_gate: bool = False
    style_token_depthwise_filter_style_basis_gate_scale: float = 0.75
    style_token_depthwise_filter_style_basis_delta: bool = False
    style_token_depthwise_filter_style_basis_delta_scale: float = 0.30
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
    style_strength_default: float = 1.0
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
    velocity_head_mode: str = "identity"
    velocity_tanh_limit: float = 20.0
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
    dynamic_style_operator_band_low_kernel: int = 9
    dynamic_style_operator_band_mid_kernel: int = 3
    dynamic_style_feature_operator: bool = False
    dynamic_style_feature_operator_strength: float = 0.0
    dynamic_style_feature_operator_band_low_kernel: int = 9
    dynamic_style_feature_operator_band_mid_kernel: int = 3
    dynamic_style_feature_operator_tanh_scale: float = 4.0
    zero_init_output_head: bool = False
    diffeomorphic_head_mode: str = "standard"
    diffeomorphic_color_strength: float = 0.85
    diffeomorphic_warp_strength: float = 0.08
    diffeomorphic_texture_gate_strength: float = 8.0
    diffeomorphic_normal_leak: float = 0.0
    diffeomorphic_color_lowpass_kernel: int = 1
    diffeomorphic_lowpass_mode: str = "avg"
    diffeomorphic_gaussian_sigma: float = 1.5
    diffeomorphic_active_grad_threshold: float = 0.0
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
    diffeomorphic_guide_mode: str = "mean"
    diffeomorphic_guide_channel: int = 2
    diffeomorphic_guide_weights: list[float] = field(default_factory=list)
    latent_canvas_strength: float = 0.0
    latent_canvas_edge_gamma: float = 4.0
    latent_canvas_highpass_kernel: int = 5
    pre_integrate_moment_match: bool = False
    pre_integrate_moment_blend: float = 1.0
    output_moment_match: bool = False
    output_moment_match_eps: float = 1e-6
    output_moment_match_train_only: bool = False
    output_residual_router: bool = False
    output_router_kernel: int = 5
    output_router_edge_gamma: float = 8.0
    output_router_highpass_floor: float = 0.10
    output_router_lowpass_strength: float = 1.0
    output_router_edge_lowpass_suppression: float = 0.0
    structure_barrier_gamma: float = 0.0
    structure_barrier_smooth_kernel: int = 3
    structure_barrier_use_anchor: bool = True
    use_style_blender: bool = False
    style_blender_init_logit: float = 0.5
    style_blender_residual: bool = False
    style_blender_residual_strength: float = 1.0
    style_blender_mode: str = "replace"
    style_blender_mod_strength: float = 1.0
    style_blender_mod_tanh_scale: float = 0.5
    style_blender_band_strength: float = 1.0
    style_blender_band_tanh_scale: float = 0.75
    style_blender_band_outer_kernel: int = 9
    style_blender_band_gate_kernel: int = 5
    style_blender_band_gate_gamma: float = 3.0
    style_blender_band_gate_floor: float = 0.15
    style_blender_dual_low_strength: float = 0.20
    style_blender_dual_mid_strength: float = 0.70
    style_blender_dual_high_strength: float = 0.00
    style_blender_dual_low_kernel: int = 11
    style_blender_dual_mid_inner_kernel: int = 3
    style_blender_dual_mid_outer_kernel: int = 11
    style_blender_dual_phase_gamma: float = 3.0
    style_blender_dual_phase_floor: float = 0.35
    style_blender_region_bins: int = 5
    style_blender_region_gamma: float = 4.0
    style_blender_region_floor: float = 0.18
    style_blender_region_smooth_kernel: int = 7
    style_blender_region_hidden_mult: float = 0.5
    style_blender_region_low_strength: float = 0.30
    style_blender_region_mid_strength: float = 0.80
    style_blender_region_high_strength: float = 0.02
    style_blender_transport_gamma: float = 4.0
    style_blender_transport_floor: float = 0.12
    style_blender_transport_power: float = 1.0
    style_blender_transport_use_entropy: bool = True
    style_blender_transport_use_uniqueness: bool = True
    style_blender_transport_low_use_support: bool = True
    style_blender_transport_low_strength: float = 0.24
    style_blender_transport_mid_strength: float = 0.88
    style_blender_transport_high_strength: float = 0.04
    style_blender_adain_moment_kernel: int = 7
    style_blender_adain_eps: float = 1e-4
    style_blender_amp_gamma: float = 3.0
    style_blender_amp_floor: float = 0.30
    style_blender_amp_low_strength: float = 0.30
    style_blender_amp_mid_strength: float = 0.90
    style_blender_amp_high_strength: float = 0.04
    style_blender_texton_hidden_mult: float = 0.75
    style_blender_texton_tanh_scale: float = 0.45
    style_blender_texton_low_strength: float = 0.18
    style_blender_texton_mid_strength: float = 0.72
    style_blender_texton_high_strength: float = 0.05
    use_checkpointing: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ModelConfig":
        known, extra = _split_known_fields(cls, payload)
        cfg = cls(**known)
        cfg.extra = extra
        return cfg

    def validated(self, *, use_checkpointing: bool | None = None) -> "ModelConfig":
        cfg = ModelConfig.from_mapping(self.to_dict())
        if use_checkpointing is not None:
            cfg.use_checkpointing = bool(use_checkpointing)
        if cfg.lift_channels is None:
            cfg.lift_channels = int(cfg.base_dim)
        return cfg

    def to_dict(self) -> dict[str, Any]:
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
    coupling_feature_mode: str = "latent"
    coupling_lowfreq_kernel: int = 9
    coupling_edge_weight: float = 0.0
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iters: int = 60
    sinkhorn_stabilize: bool = True
    bridge_sigma: float = 0.05
    bridge_noise_mode: str = "gaussian"
    bridge_style_noise_kernel: int = 5
    bridge_style_noise_flat_gamma: float = 0.0
    terminal_swd_weight: float = 0.1
    w_variance_penalty: float = 0.0
    w_content_anchor: float = 0.0
    w_edge_anchor: float = 0.0
    w_style_energy_floor: float = 0.0
    w_lowfreq_velocity: float = 0.0
    w_style_contrastive: float = 0.0
    style_contrastive_temperature: float = 0.08
    style_contrastive_pool_size: int = 4
    w_residual_style_direction: float = 0.0
    w_semantic_entropy: float = 0.0
    semantic_entropy_target: float = 2.2
    w_spectral_amplitude: float = 0.0
    spectral_amplitude_channels: int = 2
    spectral_amplitude_highpass: bool = True
    target_style_loss_weights: list[float] | None = None
    w_divergence: float = 0.0
    divergence_samples: int = 1
    w_feature_riemannian: float = 0.0
    w_kantorovich: float = 0.0
    kantorovich_steps: int = 1
    kantorovich_lr: float = 1e-4
    kantorovich_channels: int = 64
    w_nonlocal_structure: float = 0.0
    nonlocal_structure_pool: int = 8
    sb_noise_epsilon: float = 0.0
    retinex_target_blend: float = 0.0
    retinex_kernel_size: int = 15
    w_anisotropic_kinetic: float = 0.0
    anisotropic_normal_weight: float = 25.0
    anisotropic_tangent_weight: float = 0.25
    w_stokes_viscous: float = 0.0
    w_phase_separation: float = 0.0
    phase_gradient_weight: float = 0.05
    w_fourier_phase_lock: float = 0.0
    fourier_phase_lock_highpass: bool = True
    w_flat_highpass_suppression: float = 0.0
    flat_highpass_gamma: float = 8.0
    flat_highpass_kernel: int = 5
    w_edge_phase_alignment: float = 0.0
    edge_phase_gamma: float = 8.0
    edge_phase_kernel: int = 5
    w_head_color_tv: float = 0.0
    w_head_color_energy: float = 0.0
    w_head_amp_energy: float = 0.0
    w_head_warp_energy: float = 0.0
    w_head_warp_tv: float = 0.0
    w_warp_curl_reward: float = 0.0
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
    swd_signed_highpass_weight: float = 1.0
    swd_abs_highpass_weight: float = 0.0
    swd_use_dilated_projections: bool = False
    swd_projection_dilation: int = 2
    swd_channel_whiten: bool = True
    swd_channel_whiten_eps: float = 1e-3
    normalize_eps: float = 1e-8
    logit_clamp: float = 50.0
    velocity_clamp: float = 20.0
    endpoint_clamp: float = 24.0
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
    log_interval: int = 20
    use_tqdm: bool = True
    use_amp: bool = False
    amp_dtype: str = "bf16"
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    channels_last: bool = False
    use_gradient_checkpointing: bool = False
    fused_adamw: bool = True
    resume_checkpoint: str = ""
    resume_allow_missing_name_patterns: list[str] = field(default_factory=list)
    resume_skip_optimizer: bool = False
    trainable_lr_multipliers: list[list[Any]] = field(default_factory=list)
    full_eval_batch_size: int = 6
    full_eval_num_steps: int | None = None
    full_eval_step_size: float | None = None
    full_eval_style_strength: float | None = None
    full_eval_max_src_samples: int | None = None
    full_eval_max_ref_compare: int | None = None
    full_eval_max_ref_cache: int | None = None
    full_eval_ref_feature_batch_size: int | None = None
    test_image_dir: str = "../style_data/overfit50"
    full_eval_cache_dir: str = "../eval_cache"
    full_eval_image_classifier_path: str = "../eval_cache/eval_style_image_classifier.pt"
    full_eval_clip_hf_cache_dir: str = "../eval_cache/hf"
    full_eval_clip_backend: str = "hf"
    full_eval_classifier_only: bool = False
    full_eval_disable_lpips: bool = False
    full_eval_enable_art_fid: bool = False
    full_eval_enable_kid: bool = False
    numeric_debug: bool = False
    numeric_debug_interval: int = 10
    numeric_debug_halt_on_nonfinite: bool = True
    numeric_debug_dump_limit: int = 200
    max_train_batches_per_epoch: int = 0
    distill: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "TrainingConfig":
        known, extra = _split_known_fields(cls, payload)
        cfg = cls(**known)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict[str, Any]:
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
    pairing_cache_sample_mode: str = "uniform_topk"
    pairing_cache_cross_only: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DataConfig":
        known, extra = _split_known_fields(cls, payload)
        cfg = cls(**known)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict[str, Any]:
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
        return cls(
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
            "max_src_samples": "full_eval_max_src_samples",
            "max_ref_compare": "full_eval_max_ref_compare",
            "max_ref_cache": "full_eval_max_ref_cache",
            "ref_feature_batch_size": "full_eval_ref_feature_batch_size",
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
            "full_eval_max_src_samples": "max_src_samples",
            "full_eval_max_ref_compare": "max_ref_compare",
            "full_eval_max_ref_cache": "max_ref_cache",
            "full_eval_ref_feature_batch_size": "ref_feature_batch_size",
        }
        for train_key, default_key in mapping.items():
            if train_key in training and full_eval_defaults.get(default_key) == training.get(train_key):
                training.pop(train_key, None)

    return compact
