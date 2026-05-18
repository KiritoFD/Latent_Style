from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple


_INFERENCE_DEFAULTS: dict[str, Any] = {
    "inference": {"num_steps": 12, "step_size": 1.0, "style_strength": 1.0},
    "full_eval": {
        "num_steps": 12, "step_size": 1.0, "style_strength": 1.0,
        "batch_size": 2, "max_src_samples": 30, "max_ref_compare": 24,
        "max_ref_cache": 80, "ref_feature_batch_size": 8,
    },
}


@lru_cache(maxsize=1)
def load_inference_defaults() -> dict[str, Any]:
    return dict(_INFERENCE_DEFAULTS)


def resolve_inference_section(config: dict[str, Any] | None) -> dict[str, Any]:
    defaults = dict(load_inference_defaults().get("inference", {}) or {})
    if not isinstance(config, dict):
        return defaults
    local = config.get("inference", {}) or {}
    if isinstance(local, dict):
        defaults.update(local)
    return defaults


def resolve_full_eval_section(config: dict[str, Any] | None) -> dict[str, Any]:
    defaults = dict(load_inference_defaults().get("full_eval", {}) or {})
    if not isinstance(config, dict):
        return defaults
    training = config.get("training", {}) or {}
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
            if src_key in training:
                defaults[dst_key] = training.get(src_key)
    local = config.get("full_eval", {}) or {}
    if isinstance(local, dict):
        defaults.update(local)
    return defaults


def compact_runtime_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    compact = copy.deepcopy(config)
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


@dataclass
class ModelConfig:
    latent_channels: int = 4
    num_styles: int = 3
    style_dim: int = 256
    base_dim: int = 64
    lift_channels: int | None = None
    num_hires_blocks: int = 2
    num_res_blocks: int = 4
    num_decoder_blocks: int = 1
    num_groups: int = 8
    use_checkpointing: bool = False
    latent_scale_factor: float = 0.18215
    residual_gain: float = 0.1
    style_spatial_pre_gain_16: float = 0.35
    style_strength_default: float = 1.0
    style_strength_step_curve: str = "linear"
    upsample_mode: str = "nearest"
    style_id_spatial_jitter_px: int = 0
    upsample_blur: bool = True
    upsample_blur_kernel: str = "box3"
    style_attn_num_tokens: int = 64
    style_attn_num_heads: int = 4
    style_attn_sharpen_scale: float = 2.0
    style_attn_temperature: float = 0.5
    hires_block_type: str = "conv"
    body_block_type: str = "conv"
    decoder_block_type: str = "conv"
    semantic_attn_temperature: float = 0.08
    semantic_attn_temp_schedule: str = "constant"
    semantic_attn_temp_start: float = 0.08
    semantic_attn_temp_end: float = 0.08
    semantic_attn_routing_mode: str = "softmax"
    semantic_sinkhorn_iters: int = 3
    semantic_gumbel_tau: float = 1.0
    feature_attn_num_heads: int = 4
    window_attn_window_size: int = 8
    skip_fusion_mode: str = "concat_conv"
    skip_routing_mode: str = "normalized"
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
    output_moment_match: bool = False
    output_moment_match_eps: float = 1e-6
    output_moment_match_train_only: bool = True
    use_style_blender: bool = False
    time_dim: int = 0
    velocity_head_mode: str = "identity"
    velocity_tanh_limit: float = 20.0

    def __post_init__(self) -> None:
        self.latent_channels = max(1, int(self.latent_channels))
        self.num_styles = max(1, int(self.num_styles))
        self.style_dim = max(1, int(self.style_dim))
        self.base_dim = max(1, int(self.base_dim))
        self.num_hires_blocks = max(0, int(self.num_hires_blocks))
        self.num_res_blocks = max(0, int(self.num_res_blocks))
        self.num_decoder_blocks = max(0, int(self.num_decoder_blocks))
        self.num_groups = max(1, int(self.num_groups))
        self.latent_scale_factor = float(self.latent_scale_factor)
        self.residual_gain = float(self.residual_gain)
        self.style_spatial_pre_gain_16 = float(self.style_spatial_pre_gain_16)
        self.style_strength_default = max(0.0, min(1.0, float(self.style_strength_default)))
        ssc = str(self.style_strength_step_curve).lower()
        self.style_strength_step_curve = ssc if ssc in {"linear", "smoothstep", "sqrt"} else "linear"
        self.upsample_mode = str(self.upsample_mode)
        self.style_id_spatial_jitter_px = max(0, int(self.style_id_spatial_jitter_px))
        self.upsample_blur = bool(self.upsample_blur)
        ubk = str(self.upsample_blur_kernel).lower()
        self.upsample_blur_kernel = ubk if ubk in {"box3", "gaussian3"} else "box3"
        self.style_attn_num_tokens = max(1, int(self.style_attn_num_tokens))
        self.style_attn_num_heads = max(1, int(self.style_attn_num_heads))
        self.style_attn_sharpen_scale = max(0.1, float(self.style_attn_sharpen_scale))
        self.style_attn_temperature = max(1e-3, float(self.style_attn_temperature))
        self.semantic_attn_temperature = max(1e-4, float(self.semantic_attn_temperature))
        sats = str(self.semantic_attn_temp_schedule).strip().lower()
        self.semantic_attn_temp_schedule = sats if sats in {"constant", "linear", "cosine", "exp"} else "constant"
        self.semantic_attn_temp_start = max(1e-4, float(self.semantic_attn_temp_start))
        self.semantic_attn_temp_end = max(1e-4, float(self.semantic_attn_temp_end))
        srm = str(self.semantic_attn_routing_mode).strip().lower()
        self.semantic_attn_routing_mode = srm if srm in {"softmax", "sinkhorn", "gumbel_hard"} else "softmax"
        self.semantic_sinkhorn_iters = max(1, int(self.semantic_sinkhorn_iters))
        self.semantic_gumbel_tau = max(1e-3, float(self.semantic_gumbel_tau))
        self.feature_attn_num_heads = max(1, int(self.feature_attn_num_heads))
        self.window_attn_window_size = max(1, int(self.window_attn_window_size))
        sfm = str(self.skip_fusion_mode).strip().lower()
        self.skip_fusion_mode = sfm if sfm in {"concat_conv", "add_proj"} else "concat_conv"
        srm_skip = str(self.skip_routing_mode).strip().lower()
        self.skip_routing_mode = srm_skip if srm_skip in {"none", "naive", "adaptive", "normalized"} else "normalized"
        self.skip_naive_gain = max(0.0, float(self.skip_naive_gain))
        self.skip_residual_weight = max(0.0, float(self.skip_residual_weight))
        self.style_skip_content_retention_boost = max(0.0, min(1.0, float(self.style_skip_content_retention_boost)))
        self.input_anchor_noise_std = max(0.0, float(self.input_anchor_noise_std))
        self.input_anchor_noise_eval = bool(self.input_anchor_noise_eval)
        self.ablation_no_residual = bool(self.ablation_no_residual)
        self.ablation_no_residual_gain = max(0.0, float(self.ablation_no_residual_gain))
        self.ablation_disable_spatial_prior = bool(self.ablation_disable_spatial_prior)
        self.ablation_direct_delta_blend = bool(self.ablation_direct_delta_blend)
        self.raw_latent_splat_highway = bool(self.raw_latent_splat_highway)
        self.ablation_skip_clean = bool(self.ablation_skip_clean)
        self.ablation_skip_blur = bool(self.ablation_skip_blur)
        self.skip_bottleneck_channels = max(1, int(self.skip_bottleneck_channels))
        self.skip_spatial_dropout_p = max(0.0, min(1.0, float(self.skip_spatial_dropout_p)))
        self.ablation_decoder_highpass = bool(self.ablation_decoder_highpass)
        self.color_highway_gain = float(self.color_highway_gain)
        self.output_moment_match = bool(self.output_moment_match)
        self.output_moment_match_eps = max(1e-8, float(self.output_moment_match_eps))
        self.output_moment_match_train_only = bool(self.output_moment_match_train_only)
        self.use_style_blender = bool(self.use_style_blender)
        self.time_dim = max(0, int(self.time_dim))
        self.velocity_head_mode = str(self.velocity_head_mode).strip().lower()
        self.velocity_tanh_limit = max(1e-3, float(self.velocity_tanh_limit))

    @classmethod
    def from_flat_dict(cls, d: dict, *, use_checkpointing: bool = False) -> ModelConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known, use_checkpointing=use_checkpointing)


@dataclass
class BridgeConfig:
    ot_cost_mode: str = "swd"
    terminal_num_steps: int = 4
    terminal_swd_on_identity: bool = False
    w_kinetic: float = 1.5
    w_curvature: float = 0.0
    terminal_swd_weight: float = 0.15
    semantic_swd_num_projections: int = 64
    swd_distance_mode: str = "cdf"


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 2
    persistent_workers: bool = True
    prefetch_factor: int = 4
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    grad_clip_norm: float = 1.0
    num_epochs: int = 80
    save_interval: int = 10
    log_interval: int = 20
    use_tqdm: bool = True
    use_amp: bool = True
    amp_dtype: str = "bf16"
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    channels_last: bool = True
    use_gradient_checkpointing: bool = True
    fused_adamw: bool = True
    full_eval_batch_size: int = 6
    test_image_dir: str = "../style_data/overfit50"
    full_eval_cache_dir: str = "../Cycle-NCE/eval_cache"
    full_eval_image_classifier_path: str = "../Cycle-NCE/eval_cache/eval_style_image_classifier.pt"
    full_eval_clip_hf_cache_dir: str = "../Cycle-NCE/eval_cache/hf"
    full_eval_clip_backend: str = "hf"


@dataclass
class DataConfig:
    data_root: str = "../latent-256"
    style_subdirs: Tuple[str, ...] = ("photo", "Hayao", "monet", "vangogh", "cezanne")
    allow_hflip: bool = True
    balance_target_styles_per_batch: bool = True
    virtual_length_multiplier: int = 1


@dataclass
class CheckpointConfig:
    save_dir: str = "./ot-exp-80"


@dataclass
class LBMConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def load(cls, path: str | Path) -> LBMConfig:
        raw = load_config(path)
        return cls(
            model=ModelConfig.from_flat_dict(raw.get("model", {})),
            bridge=BridgeConfig(**raw.get("bridge", {})),
            training=TrainConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            checkpoint=CheckpointConfig(**raw.get("checkpoint", {})),
        )
