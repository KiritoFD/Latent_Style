from __future__ import annotations

import warnings
from dataclasses import MISSING, fields

import torch
import torch.nn as nn

from config_schema import ModelConfig
from lancet_blocks import (
    DecoderTextureBlock,
    GatedSpadeAttention,
    GWOTAttention,
    NormFreeModulation,
    PnPSelfAttentionInject,
    SemanticCrossAttn,
    SimpleResBlock,
    SpatialModulatedSelfAttn,
    StyleBlender,
    StyleRoutingSkip,
    _build_feature_block,
    _normalize_feature_block_type,
    _resolve_group_count,
)
from lancet_runtime import LatentAdaCUTRuntimeMixin
from semantic_tokenizer import (
    AffineConnectionTokenizer,
    CrossImageRoutingTokenizer,
    DinoDictionaryTokenizer,
    PureLatentSpatialTokenizer,
    ResidualSemanticAdapterTokenizer,
    SMoETranslatorTokenizer,
    VLMPromptStyleTokenizer,
)
from style_families import BACKBONE_ATTENTION_FAMILIES, TOKENIZER_FAMILIES, normalize_family
from style_tokenizer import FactorizedStyleTokenizer


_SKIP_FUSION_MODES = {"concat_conv", "add_proj"}


def _materialize_local_missing_dataclass_fields(obj: object) -> None:
    for item in fields(obj):
        name = str(item.name)
        if hasattr(obj, name):
            continue
        if item.default is not MISSING:
            value = item.default
        elif item.default_factory is not MISSING:  # type: ignore[attr-defined]
            value = item.default_factory()  # type: ignore[misc]
        else:
            value = None
        setattr(obj, name, value)


class LatentAdaCUT(LatentAdaCUTRuntimeMixin, nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        cfg = config.validated()
        _materialize_local_missing_dataclass_fields(cfg)
        self.config = cfg
        latent_channels = int(cfg.latent_channels)
        style_dim = int(cfg.style_dim)
        num_groups = int(cfg.num_groups)

        self.latent_channels = latent_channels
        self.num_styles = int(cfg.num_styles)
        self.use_checkpointing = bool(cfg.use_checkpointing)
        self.latent_scale_factor = float(cfg.latent_scale_factor)
        self.residual_gain = float(cfg.residual_gain)
        self.style_code_dim = style_dim
        self.register_buffer("_style_code_anchor", torch.zeros(1, style_dim), persistent=False)
        self.lift_channels = int(cfg.lift_channels)
        self.body_channels = int(cfg.base_dim * 2)
        tokenizer_num_atoms = max(2, int(getattr(cfg, "tokenizer_num_atoms", 32)))
        self.num_hires_blocks = max(0, int(getattr(cfg, "num_hires_blocks", 2)))
        self.num_res_blocks = max(0, int(getattr(cfg, "num_res_blocks", 4)))
        self.style_strength_max = max(1e-6, float(getattr(cfg, "style_strength_max", 1.0)))
        self.style_strength_default = max(0.0, min(self.style_strength_max, float(getattr(cfg, "style_strength_default", 1.0))))
        self.style_strength_step_curve = str(getattr(cfg, "style_strength_step_curve", "linear")).lower()
        if self.style_strength_step_curve not in {"linear", "smoothstep", "sqrt"}:
            self.style_strength_step_curve = "linear"
        self.last_style_strength_debug: dict[str, float] = {}
        self.upsample_mode = str(getattr(cfg, "upsample_mode", "nearest"))
        self.upsample_blur = bool(getattr(cfg, "upsample_blur", True))
        self.upsample_blur_kernel = str(getattr(cfg, "upsample_blur_kernel", "box3")).lower()
        self.style_attn_num_tokens = max(1, int(getattr(cfg, "style_attn_num_tokens", 128)))
        self.style_attn_num_heads = max(1, int(getattr(cfg, "style_attn_num_heads", 4)))
        self.num_atoms = tokenizer_num_atoms
        self.tokenizer_content_adaptive = bool(getattr(cfg, "tokenizer_content_adaptive", False))
        self.tokenizer_content_gain = float(getattr(cfg, "tokenizer_content_gain", 0.5))
        self.tokenizer_content_stopgrad = bool(getattr(cfg, "tokenizer_content_stopgrad", True))
        self.tokenizer_content_style_gate = bool(getattr(cfg, "tokenizer_content_style_gate", False))
        self.tokenizer_content_style_gate_max = max(1e-3, float(getattr(cfg, "tokenizer_content_style_gate_max", 2.0)))
        self.style_attn_sharpen_scale = 0  # 629 D15: confirmed ineffective (baseline 2.5 → prune_to 0)
        self.style_attn_temperature = max(1e-3, float(getattr(cfg, "style_attn_temperature", 0.08)))
        self.hires_block_type = _normalize_feature_block_type(getattr(cfg, "hires_block_type", "conv"))
        self.body_block_type = _normalize_feature_block_type(getattr(cfg, "body_block_type", "global_attn"))
        self.decoder_block_type = _normalize_feature_block_type(getattr(cfg, "decoder_block_type", "conv"))
        self.tokenizer_family = normalize_family(
            str(getattr(cfg, "tokenizer_family", "legacy_factorized")),
            allowed=TOKENIZER_FAMILIES,
            default="legacy_factorized",
        )
        self.backbone_attention_family = normalize_family(
            str(getattr(cfg, "backbone_attention_family", "legacy_semantic_crossattn")),
            allowed=BACKBONE_ATTENTION_FAMILIES,
            default="legacy_semantic_crossattn",
        )
        self.semantic_attn_temperature = max(1e-4, float(getattr(cfg, "semantic_attn_temperature", 0.08)))
        self.semantic_attn_routing_mode = str(getattr(cfg, "semantic_attn_routing_mode", "softmax")).strip().lower()
        if self.semantic_attn_routing_mode not in {"softmax", "sinkhorn", "gumbel_hard"}:
            self.semantic_attn_routing_mode = "softmax"
        self.semantic_self_topology_gate = bool(getattr(cfg, "semantic_self_topology_gate", False))
        self.semantic_self_topology_blend = max(0.0, min(1.0, float(getattr(cfg, "semantic_self_topology_blend", 1.0))))
        if (not self.semantic_self_topology_gate) and self.semantic_self_topology_blend > 0.0:
            warnings.warn(
                "semantic_self_topology_blend is non-zero but semantic_self_topology_gate is disabled; "
                "the topology-blend sweep would be a no-op until the gate is enabled.",
                category=UserWarning,
                stacklevel=2,
            )
        self.matched_target_conditioning_mode = str(
            getattr(cfg, "matched_target_conditioning_mode", "auto")
        ).strip().lower()
        if self.matched_target_conditioning_mode not in {"auto", "none", "spatial", "code", "both"}:
            self.matched_target_conditioning_mode = "auto"
        self.matched_target_style_encoder_mode = str(
            getattr(cfg, "matched_target_style_encoder_mode", "none")
        ).strip().lower()
        if self.matched_target_style_encoder_mode not in {"none", "residual", "replace"}:
            self.matched_target_style_encoder_mode = "none"
        self.matched_target_style_encoder_hidden_dim = max(
            8,
            int(getattr(cfg, "matched_target_style_encoder_hidden_dim", 192)),
        )
        self.matched_target_style_encoder_highpass_kernel = max(
            1,
            int(getattr(cfg, "matched_target_style_encoder_highpass_kernel", 5)),
        )
        if self.matched_target_style_encoder_highpass_kernel % 2 == 0:
            self.matched_target_style_encoder_highpass_kernel += 1
        self.matched_target_style_encoder_residual_scale = max(
            0.0,
            float(getattr(cfg, "matched_target_style_encoder_residual_scale", 1.0)),
        )
        self.semantic_sinkhorn_iters = max(1, int(getattr(cfg, "semantic_sinkhorn_iters", 3)))
        self.semantic_gumbel_tau = max(1e-3, float(getattr(cfg, "semantic_gumbel_tau", 1.0)))
        self.num_decoder_blocks = max(0, int(getattr(cfg, "num_decoder_blocks", 2)))
        self.feature_attn_num_heads = max(1, int(getattr(cfg, "feature_attn_num_heads", 4)))
        self.window_attn_window_size = max(1, int(getattr(cfg, "window_attn_window_size", 8)))
        self.skip_fusion_mode = str(getattr(cfg, "skip_fusion_mode", "add_proj")).strip().lower()
        if self.skip_fusion_mode not in _SKIP_FUSION_MODES:
            self.skip_fusion_mode = "concat_conv"
        self.skip_routing_mode = str(getattr(cfg, "skip_routing_mode", "none")).strip().lower()
        if self.skip_routing_mode not in {"none", "naive", "adaptive", "normalized"}:
            self.skip_routing_mode = "normalized"
        self.skip_disabled = self.skip_routing_mode == "none"
        self.skip_naive_gain = max(0.0, float(getattr(cfg, "skip_naive_gain", 1.0)))
        self.skip_residual_weight = 0  # 629 D17: confirmed ineffective (baseline 0.1 → prune_to 0)
        self.style_skip_content_retention_boost = max(0.0, min(1.0, float(getattr(cfg, "style_skip_content_retention_boost", 0.0))))
        self.input_anchor_noise_std = max(0.0, float(getattr(cfg, "input_anchor_noise_std", 0.0)))
        self.input_anchor_noise_eval = bool(getattr(cfg, "input_anchor_noise_eval", False))
        if self.decoder_block_type == "window_attn" and (self.num_decoder_blocks % 2) != 0:
            warnings.warn(
                "decoder_block_type=window_attn works best with even num_decoder_blocks for shifted-window pairing.",
                category=UserWarning,
                stacklevel=2,
            )
        self.ablation_no_residual = bool(getattr(cfg, "ablation_no_residual", False))
        self.ablation_no_residual_gain = max(0.0, float(getattr(cfg, "ablation_no_residual_gain", 1.0)))
        self.ablation_direct_delta_blend = bool(getattr(cfg, "ablation_direct_delta_blend", False))
        self.raw_latent_splat_highway = bool(getattr(cfg, "raw_latent_splat_highway", False))
        self.ablation_skip_clean = bool(getattr(cfg, "ablation_skip_clean", True))
        self.ablation_skip_blur = bool(getattr(cfg, "ablation_skip_blur", True))
        self.skip_bottleneck_channels = max(1, int(getattr(cfg, "skip_bottleneck_channels", 16)))
        self.skip_spatial_dropout_p = max(0.0, min(1.0, float(getattr(cfg, "skip_spatial_dropout_p", 0.15))))
        self.ablation_decoder_highpass = bool(getattr(cfg, "ablation_decoder_highpass", True))
        self.color_highway_gain = float(getattr(cfg, "color_highway_gain", 1.0))
        self.pre_integrate_moment_match = bool(getattr(cfg, "pre_integrate_moment_match", False))
        self.pre_integrate_moment_blend = max(0.0, min(1.0, float(getattr(cfg, "pre_integrate_moment_blend", 1.0))))
        self.output_moment_match = bool(getattr(cfg, "output_moment_match", False))
        self.output_moment_match_eps = max(1e-8, float(getattr(cfg, "output_moment_match_eps", 1e-6)))
        self.output_moment_match_train_only = bool(getattr(cfg, "output_moment_match_train_only", False))
        self.output_appearance_alignment_mode = str(getattr(cfg, "output_appearance_alignment_mode", "none")).strip().lower()
        if self.output_appearance_alignment_mode not in {"none", "tokenizer_latent_affine"}:
            self.output_appearance_alignment_mode = "none"
        self.output_appearance_hidden_dim = max(4, int(getattr(cfg, "output_appearance_hidden_dim", 96)))
        self.output_appearance_log_scale_span = max(0.0, float(getattr(cfg, "output_appearance_log_scale_span", 0.22314355131420976)))
        self.output_appearance_shift_span = max(0.0, float(getattr(cfg, "output_appearance_shift_span", 0.35)))
        self.output_appearance_blend = max(0.0, min(1.0, float(getattr(cfg, "output_appearance_blend", 1.0))))
        self.output_appearance_use_spatial_stats = bool(getattr(cfg, "output_appearance_use_spatial_stats", True))
        self.output_appearance_use_gate_mask_stats = bool(getattr(cfg, "output_appearance_use_gate_mask_stats", True))
        self.output_appearance_head: nn.Module | None = None
        self.last_output_appearance_debug: dict[str, float] = {}
        self.last_output_style_context: dict[str, object] | None = None
        if self.output_appearance_alignment_mode != "none":
            appearance_feature_dim = style_dim
            if self.output_appearance_use_spatial_stats:
                appearance_feature_dim += int(self.body_channels) * 2
            if self.output_appearance_use_gate_mask_stats:
                appearance_feature_dim += 4
            self.output_appearance_head = nn.Sequential(
                nn.LayerNorm(appearance_feature_dim),
                nn.Linear(appearance_feature_dim, self.output_appearance_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.output_appearance_hidden_dim, int(self.latent_channels) * 2),
            )
            last = self.output_appearance_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, mean=0.0, std=0.01)
                nn.init.zeros_(last.bias)
        matched_target_feature_dim = int(self.body_channels) * 4
        self.matched_target_style_encoder_head: nn.Module | None = None
        if self.matched_target_style_encoder_mode != "none":
            self.matched_target_style_encoder_head = nn.Sequential(
                nn.LayerNorm(matched_target_feature_dim),
                nn.Linear(matched_target_feature_dim, self.matched_target_style_encoder_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.matched_target_style_encoder_hidden_dim, style_dim),
            )
            final = self.matched_target_style_encoder_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.normal_(final.weight, mean=0.0, std=0.02)
                nn.init.zeros_(final.bias)
        self.style_code_spatial_mode = str(getattr(cfg, "style_code_spatial_mode", "none")).strip().lower()
        if self.style_code_spatial_mode not in {"none", "lowrank"}:
            self.style_code_spatial_mode = "none"
        self.style_code_spatial_hidden_dim = max(4, int(getattr(cfg, "style_code_spatial_hidden_dim", 64)))
        self.style_code_spatial_rank = max(1, int(getattr(cfg, "style_code_spatial_rank", 8)))
        self.style_code_spatial_base_hw = max(4, int(getattr(cfg, "style_code_spatial_base_hw", 16)))
        self.style_code_spatial_scale = max(0.0, float(getattr(cfg, "style_code_spatial_scale", 0.35)))
        self.style_code_spatial_head: nn.Module | None = None
        self.style_code_spatial_channel_bias: nn.Module | None = None
        self.style_code_spatial_basis: nn.Parameter | None = None
        if self.style_code_spatial_mode == "lowrank" and self.style_code_spatial_scale > 0.0:
            self.style_code_spatial_head = nn.Sequential(
                nn.LayerNorm(style_dim),
                nn.Linear(style_dim, self.style_code_spatial_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.style_code_spatial_hidden_dim, self.style_code_spatial_rank),
            )
            self.style_code_spatial_channel_bias = nn.Sequential(
                nn.LayerNorm(style_dim),
                nn.Linear(style_dim, self.body_channels),
            )
            self.style_code_spatial_basis = nn.Parameter(
                torch.randn(
                    self.style_code_spatial_rank,
                    self.body_channels,
                    self.style_code_spatial_base_hw,
                    self.style_code_spatial_base_hw,
                ) * 0.02
            )
            for module in (self.style_code_spatial_head, self.style_code_spatial_channel_bias):
                last = module[-1]
                if isinstance(last, nn.Linear):
                    nn.init.normal_(last.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(last.bias)
        self.use_style_blender = bool(getattr(cfg, "use_style_blender", False))
        if self.upsample_blur_kernel not in {"box3", "gaussian3"}:
            self.upsample_blur_kernel = "box3"

        tokenizer_kind = str(getattr(cfg, "style_tokenizer", "factorized")).strip().lower()
        latent_spatial_tokenizer_family = self.tokenizer_family in {
            "pure_latent_spatial",
            "smoe_translator",
            "affine_connection_tokenizer",
        }
        if latent_spatial_tokenizer_family:
            if tokenizer_kind not in {"", "null", "none", "pure_placeholder"}:
                raise ValueError(
                    f"tokenizer_family={self.tokenizer_family!r} only supports a null compatibility tokenizer, "
                    f"got style_tokenizer={getattr(cfg, 'style_tokenizer', tokenizer_kind)!r}"
                )
            self.style_tokenizer = None
        else:
            if tokenizer_kind != "factorized":
                raise ValueError(f"Unsupported style_tokenizer: {getattr(cfg, 'style_tokenizer', tokenizer_kind)}")
            self.style_tokenizer = FactorizedStyleTokenizer(
                num_styles=self.num_styles,
                style_dim=style_dim,
                identity_dim=int(getattr(cfg, "tokenizer_identity_dim", 24)),
                texture_dim=int(getattr(cfg, "tokenizer_texture_dim", 32)),
                geometry_dim=int(getattr(cfg, "tokenizer_geometry_dim", 24)),
                init_std=float(getattr(cfg, "tokenizer_init_std", 0.02)),
                projection_mode=str(getattr(cfg, "tokenizer_projection_mode", "concat")),
                residual_gain=0,  # 629 D14: confirmed ineffective (baseline 0.5 → prune_to 0)
                num_atoms=tokenizer_num_atoms,
                num_prototypes=int(getattr(cfg, "tokenizer_num_prototypes", 4)),
                atom_temperature=float(getattr(cfg, "tokenizer_atom_temperature", 0.25)),
                field_dropout_p=float(getattr(cfg, "tokenizer_field_dropout_p", 0.0)),
                code_l2_norm=bool(getattr(cfg, "tokenizer_code_l2_norm", False)),
                code_scale=float(getattr(cfg, "tokenizer_code_scale", 1.0)),
                atom_topk=int(getattr(cfg, "tokenizer_atom_topk", 0)),
                atom_hard_eval=bool(getattr(cfg, "tokenizer_atom_hard_eval", False)),
            )
        self.structured_style_tokenizer: nn.Module | None = None
        structured_dino_dim = max(1, int(getattr(cfg, "tokenizer_dino_dim", 384)))
        structured_num_clusters = max(1, int(getattr(cfg, "tokenizer_num_clusters", 16)))
        structured_query_dim = max(8, int(getattr(cfg, "tokenizer_query_dim", 64)))
        structured_query_num_blocks = max(1, int(getattr(cfg, "tokenizer_query_num_blocks", 4)))
        structured_spatial_dim = max(1, int(getattr(cfg, "tokenizer_spatial_dim", 0) or self.body_channels))
        structured_pe_temperature = max(0.0, float(getattr(cfg, "tokenizer_pe_temperature", 1.0)))
        structured_global_gate_hidden_dim = max(1, int(getattr(cfg, "tokenizer_global_gate_hidden_dim", style_dim)))
        structured_global_gate_scale = max(0.0, float(getattr(cfg, "tokenizer_global_gate_scale", 1.0)))
        structured_temperature = max(1e-3, float(getattr(cfg, "tokenizer_structured_temperature", 0.1)))
        structured_prompt_dim = max(1, int(getattr(cfg, "tokenizer_prompt_dim", 256)))
        structured_prompt_length = max(1, int(getattr(cfg, "tokenizer_prompt_length", 8)))
        smoe_translation_rank = max(0, int(getattr(cfg, "smoe_translation_rank", 0)))
        affine_gamma_scale = max(0.0, float(getattr(cfg, "affine_connection_gamma_scale", 0.5)))
        affine_beta_scale = max(0.0, float(getattr(cfg, "affine_connection_beta_scale", 1.0)))
        affine_fiber_mode = str(getattr(cfg, "affine_connection_fiber_mode", "wavelet") or "wavelet")
        affine_lowpass_kernel = max(1, int(getattr(cfg, "affine_connection_lowpass_kernel", 5)))
        self.tokenizer_spatial_dim = structured_spatial_dim
        self.structured_style_map_proj: nn.Module | None = None
        if structured_spatial_dim != self.body_channels:
            self.structured_style_map_proj = nn.Conv2d(structured_spatial_dim, self.body_channels, kernel_size=1, bias=False)
            with torch.no_grad():
                self.structured_style_map_proj.weight.zero_()
                diag = min(int(self.body_channels), int(structured_spatial_dim))
                for idx in range(diag):
                    self.structured_style_map_proj.weight[idx, idx, 0, 0] = 1.0
        if self.tokenizer_family == "pure_latent_spatial":
            self.structured_style_tokenizer = PureLatentSpatialTokenizer(
                num_styles=self.num_styles,
                global_dim=style_dim,
                spatial_dim=structured_spatial_dim,
                latent_channels=self.latent_channels,
                num_clusters=structured_num_clusters,
                temperature=structured_temperature,
                query_dim=structured_query_dim,
                query_num_blocks=structured_query_num_blocks,
                pe_temperature=structured_pe_temperature,
                global_gate_hidden_dim=structured_global_gate_hidden_dim,
                global_gate_scale=structured_global_gate_scale,
            )
        elif self.tokenizer_family == "smoe_translator":
            self.structured_style_tokenizer = SMoETranslatorTokenizer(
                num_styles=self.num_styles,
                global_dim=style_dim,
                spatial_dim=structured_spatial_dim,
                latent_channels=self.latent_channels,
                num_clusters=structured_num_clusters,
                temperature=structured_temperature,
                query_dim=structured_query_dim,
                query_num_blocks=structured_query_num_blocks,
                pe_temperature=structured_pe_temperature,
                global_gate_hidden_dim=structured_global_gate_hidden_dim,
                global_gate_scale=structured_global_gate_scale,
                translation_rank=smoe_translation_rank,
            )
        elif self.tokenizer_family == "affine_connection_tokenizer":
            self.structured_style_tokenizer = AffineConnectionTokenizer(
                num_styles=self.num_styles,
                global_dim=style_dim,
                spatial_dim=structured_spatial_dim,
                latent_channels=self.latent_channels,
                num_clusters=structured_num_clusters,
                temperature=structured_temperature,
                query_dim=structured_query_dim,
                query_num_blocks=structured_query_num_blocks,
                pe_temperature=structured_pe_temperature,
                global_gate_hidden_dim=structured_global_gate_hidden_dim,
                global_gate_scale=structured_global_gate_scale,
                gamma_scale=affine_gamma_scale,
                beta_scale=affine_beta_scale,
                fiber_mode=affine_fiber_mode,
                lowpass_kernel=affine_lowpass_kernel,
            )
        elif self.tokenizer_family == "tok_a_dino_dict":
            self.structured_style_tokenizer = DinoDictionaryTokenizer(
                num_styles=self.num_styles,
                global_dim=style_dim,
                spatial_dim=structured_spatial_dim,
                dino_dim=structured_dino_dim,
                num_clusters=structured_num_clusters,
                temperature=structured_temperature,
            )
        elif self.tokenizer_family == "tok_b_cross_image":
            self.structured_style_tokenizer = CrossImageRoutingTokenizer(
                num_styles=self.num_styles,
                global_dim=style_dim,
                spatial_dim=structured_spatial_dim,
                dino_dim=structured_dino_dim,
                temperature=structured_temperature,
            )
        elif self.tokenizer_family == "tok_c_residual_adapter":
            self.structured_style_tokenizer = ResidualSemanticAdapterTokenizer(
                num_styles=self.num_styles,
                global_dim=style_dim,
                spatial_dim=structured_spatial_dim,
                dino_dim=structured_dino_dim,
                num_clusters=structured_num_clusters,
                temperature=structured_temperature,
            )
        elif self.tokenizer_family == "tok_d_vlm_prompt":
            self.structured_style_tokenizer = VLMPromptStyleTokenizer(
                num_styles=self.num_styles,
                global_dim=style_dim,
                spatial_dim=structured_spatial_dim,
                dino_dim=structured_dino_dim,
                prompt_dim=structured_prompt_dim,
                prompt_length=structured_prompt_length,
            )
        self.style_code_content_router: nn.Module | None = None
        self.style_code_content_style_gate: nn.Embedding | None = None
        if self.tokenizer_content_adaptive and not latent_spatial_tokenizer_family:
            router_in = self.body_channels * 4 + 1
            hidden = max(4, int(getattr(cfg, "tokenizer_content_hidden_dim", 64)))
            self.style_code_content_router = nn.Sequential(
                nn.LayerNorm(router_in),
                nn.Linear(router_in, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.num_atoms),
            )
            last = self.style_code_content_router[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            if self.tokenizer_content_style_gate:
                self.style_code_content_style_gate = nn.Embedding(self.num_styles, 1)
                init_ratio = float(getattr(cfg, "tokenizer_content_style_gate_init", 1.0)) / self.tokenizer_content_style_gate_max
                init_ratio = max(1e-4, min(1.0 - 1e-4, init_ratio))
                init_logit = torch.logit(torch.tensor(init_ratio, dtype=torch.float32)).item()
                nn.init.constant_(self.style_code_content_style_gate.weight, init_logit)

        # 32x32 lift stage before downsampling.
        self.enc_in = nn.Conv2d(latent_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        self.enc_in_act = nn.SiLU()
        self.hires_body = nn.ModuleList(
            [
                _build_feature_block(
                    self.hires_block_type,
                    dim=self.lift_channels,
                    style_dim=style_dim,
                    num_groups=num_groups,
                    style_attn_num_tokens=self.style_attn_num_tokens,
                    style_attn_num_heads=self.style_attn_num_heads,
                    style_attn_sharpen_scale=0,  # 629 D15: confirmed ineffective
                    feature_attn_num_heads=self.feature_attn_num_heads,
                    style_attn_temperature=self.style_attn_temperature,
                    window_attn_window_size=self.window_attn_window_size,
                )
                for _ in range(self.num_hires_blocks)
            ]
        )
        self.down = nn.Conv2d(self.lift_channels, self.body_channels, kernel_size=4, stride=2, padding=1)

        def _make_body_block() -> nn.Module:
            if self.backbone_attention_family == "attn_sa_mod":
                return SpatialModulatedSelfAttn(
                    dim=self.body_channels,
                    num_groups=num_groups,
                    temperature=self.semantic_attn_temperature,
                )
            if self.backbone_attention_family == "attn_gw_ot":
                return GWOTAttention(
                    dim=self.body_channels,
                    num_groups=num_groups,
                    temperature=self.semantic_attn_temperature,
                    spatial_lambda=float(getattr(cfg, "semantic_gw_spatial_lambda", 0.25)),
                    sinkhorn_iters=self.semantic_sinkhorn_iters,
                )
            if self.backbone_attention_family == "attn_gated_spade":
                return GatedSpadeAttention(
                    dim=self.body_channels,
                    num_groups=num_groups,
                    temperature=self.semantic_attn_temperature,
                )
            if self.backbone_attention_family == "attn_pnp_selfinject":
                return PnPSelfAttentionInject(
                    dim=self.body_channels,
                    num_groups=num_groups,
                    temperature=self.semantic_attn_temperature,
                )
            return SemanticCrossAttn(
                dim=self.body_channels,
                num_groups=num_groups,
                temperature=self.semantic_attn_temperature,
                paint_only=self.use_style_blender,
                routing_mode=self.semantic_attn_routing_mode,
                sinkhorn_iters=self.semantic_sinkhorn_iters,
                gumbel_tau=self.semantic_gumbel_tau,
                self_topology_gate=self.semantic_self_topology_gate,
                self_topology_blend=self.semantic_self_topology_blend,
            )
        self.body_blocks = nn.ModuleList([_make_body_block() for _ in range(self.num_res_blocks)])
        self.blender = StyleBlender(dim=self.body_channels, num_groups=num_groups) if self.use_style_blender else None

        # Decoder: 16 -> 32
        upsample_kwargs = {"scale_factor": 2, "mode": self.upsample_mode}
        if self.upsample_mode in {"bilinear", "bicubic"}:
            upsample_kwargs["align_corners"] = False
        self.dec_up = nn.Upsample(**upsample_kwargs)
        skip_gn_groups = _resolve_group_count(self.lift_channels, num_groups)
        if self.skip_disabled:
            # In no-skip mode, keep only the upsample projection path and do not build
            # any skip-source routing/projection modules.
            self.skip_up_proj = nn.Conv2d(self.body_channels, self.lift_channels, kernel_size=1, stride=1, padding=0)
            self.skip_src_proj = nn.Identity()
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(skip_gn_groups, self.lift_channels),
                nn.SiLU(inplace=True),
            )
        elif self.skip_fusion_mode == "add_proj":
            self.skip_up_proj = nn.Conv2d(self.body_channels, self.lift_channels, kernel_size=1, stride=1, padding=0)
            self.skip_src_proj = nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=1, stride=1, padding=0)
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(skip_gn_groups, self.lift_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.skip_up_proj = nn.Identity()
            self.skip_src_proj = nn.Identity()
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(self.body_channels + self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(skip_gn_groups, self.lift_channels),
                nn.SiLU(inplace=True),
            )
        self.skip_router = None
        if not self.skip_disabled:
            self.skip_router = StyleRoutingSkip(
                channels=self.lift_channels,
                style_dim=style_dim,
                mode=self.skip_routing_mode,
                content_retention_boost=self.style_skip_content_retention_boost,
            )
        squeeze_channels = max(1, min(self.lift_channels, self.skip_bottleneck_channels))
        self.skip_squeeze = nn.Sequential(
            nn.Conv2d(self.lift_channels, squeeze_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(squeeze_channels, affine=False),
            nn.SiLU(),
            nn.Conv2d(squeeze_channels, self.lift_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.skip_spatial_dropout = nn.Dropout2d(p=self.skip_spatial_dropout_p)
        self.decoder_blocks = nn.ModuleList(
            [
                SimpleResBlock(
                    dim=self.lift_channels,
                    num_groups=num_groups,
                )
                for _ in range(self.num_decoder_blocks)
            ]
        )
        self.dec_post = nn.Sequential(
            nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.dec_mod = NormFreeModulation(self.lift_channels, style_dim)
        self.dec_act = nn.SiLU()
        self.dec_out = nn.Conv2d(self.lift_channels, latent_channels, kernel_size=3, stride=1, padding=1)
        self.highway_proj = nn.Conv2d(
            self.body_channels,
            self.latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        nn.init.normal_(self.highway_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.highway_proj.bias)

        if self.upsample_blur:
            if self.upsample_blur_kernel == "gaussian3":
                k = torch.tensor(
                    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                    dtype=torch.float32,
                ) / 16.0
            else:
                k = torch.ones((3, 3), dtype=torch.float32) / 9.0
            self.register_buffer("_upsample_blur_kernel", k.view(1, 1, 3, 3), persistent=False)
            self.register_buffer(
                "_upsample_blur_kernel_body",
                k.view(1, 1, 3, 3).repeat(self.body_channels, 1, 1, 1).contiguous(),
                persistent=False,
            )
        else:
            self.register_buffer("_upsample_blur_kernel", torch.empty(0), persistent=False)
            self.register_buffer("_upsample_blur_kernel_body", torch.empty(0), persistent=False)
        self._upsample_blur_kernel_cache: dict[tuple[int, str], torch.Tensor] = {}
        total_blocks = self.num_hires_blocks + self.num_res_blocks + self.num_decoder_blocks
        init_gains = torch.linspace(-2.0, 1.0, max(1, total_blocks))
        self.block_gains = nn.Parameter(init_gains)
        self.alpha_predictor = nn.Sequential(
            nn.Conv2d(self.latent_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
