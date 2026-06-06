from __future__ import annotations

import warnings
from dataclasses import MISSING, fields

import torch
import torch.nn as nn

from config_schema import ModelConfig
from lancet_blocks import (
    DecoderTextureBlock,
    NormFreeModulation,
    SemanticCrossAttn,
    SimpleResBlock,
    StyleBlender,
    StyleRoutingSkip,
    _build_feature_block,
    _normalize_feature_block_type,
    _resolve_group_count,
)
from lancet_runtime import LatentAdaCUTRuntimeMixin
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
        self.lift_channels = int(cfg.lift_channels)
        self.body_channels = int(cfg.base_dim * 2)
        tokenizer_num_atoms = max(2, int(getattr(cfg, "tokenizer_num_atoms", 32)))
        self.num_hires_blocks = max(0, int(cfg.num_hires_blocks))
        self.num_res_blocks = max(0, int(cfg.num_res_blocks))
        self.style_spatial_pre_gain_16 = float(cfg.style_spatial_pre_gain_16)
        self.style_strength_default = max(0.0, min(1.0, float(cfg.style_strength_default)))
        self.style_strength_step_curve = str(cfg.style_strength_step_curve).lower()
        if self.style_strength_step_curve not in {"linear", "smoothstep", "sqrt"}:
            self.style_strength_step_curve = "linear"
        self.upsample_mode = str(cfg.upsample_mode)
        self.style_id_spatial_jitter_px = max(0, int(cfg.style_id_spatial_jitter_px))
        self.upsample_blur = bool(cfg.upsample_blur)
        self.upsample_blur_kernel = str(cfg.upsample_blur_kernel).lower()
        self.style_attn_num_tokens = max(1, int(cfg.style_attn_num_tokens))
        self.style_attn_num_heads = max(1, int(cfg.style_attn_num_heads))
        self.num_atoms = tokenizer_num_atoms
        self.tokenizer_content_adaptive = bool(cfg.tokenizer_content_adaptive)
        self.tokenizer_content_gain = float(cfg.tokenizer_content_gain)
        self.tokenizer_content_stopgrad = bool(cfg.tokenizer_content_stopgrad)
        self.tokenizer_content_style_gate = bool(cfg.tokenizer_content_style_gate)
        self.tokenizer_content_style_gate_max = max(1e-3, float(cfg.tokenizer_content_style_gate_max))
        self.style_attn_sharpen_scale = max(0.1, float(cfg.style_attn_sharpen_scale))
        self.style_attn_temperature = max(1e-3, float(cfg.style_attn_temperature))
        self.hires_block_type = _normalize_feature_block_type(cfg.hires_block_type)
        self.body_block_type = _normalize_feature_block_type(cfg.body_block_type)
        self.decoder_block_type = _normalize_feature_block_type(cfg.decoder_block_type)
        self.semantic_attn_temperature = max(1e-4, float(cfg.semantic_attn_temperature))
        self.semantic_attn_routing_mode = str(cfg.semantic_attn_routing_mode).strip().lower()
        if self.semantic_attn_routing_mode not in {"softmax", "sinkhorn", "gumbel_hard"}:
            self.semantic_attn_routing_mode = "softmax"
        self.semantic_sinkhorn_iters = max(1, int(cfg.semantic_sinkhorn_iters))
        self.semantic_gumbel_tau = max(1e-3, float(cfg.semantic_gumbel_tau))
        self.num_decoder_blocks = max(0, int(cfg.num_decoder_blocks))
        self.feature_attn_num_heads = max(1, int(cfg.feature_attn_num_heads))
        self.window_attn_window_size = max(1, int(cfg.window_attn_window_size))
        self.skip_fusion_mode = str(cfg.skip_fusion_mode).strip().lower()
        if self.skip_fusion_mode not in _SKIP_FUSION_MODES:
            self.skip_fusion_mode = "concat_conv"
        self.skip_routing_mode = str(cfg.skip_routing_mode).strip().lower()
        if self.skip_routing_mode not in {"none", "naive", "adaptive", "normalized"}:
            self.skip_routing_mode = "normalized"
        self.skip_disabled = self.skip_routing_mode == "none"
        self.skip_naive_gain = max(0.0, float(cfg.skip_naive_gain))
        self.skip_residual_weight = max(0.0, float(cfg.skip_residual_weight))
        self.style_skip_content_retention_boost = max(0.0, min(1.0, float(cfg.style_skip_content_retention_boost)))
        self.input_anchor_noise_std = max(0.0, float(cfg.input_anchor_noise_std))
        self.input_anchor_noise_eval = bool(cfg.input_anchor_noise_eval)
        if self.decoder_block_type == "window_attn" and (self.num_decoder_blocks % 2) != 0:
            warnings.warn(
                "decoder_block_type=window_attn works best with even num_decoder_blocks for shifted-window pairing.",
                category=UserWarning,
                stacklevel=2,
            )
        self.ablation_no_residual = bool(cfg.ablation_no_residual)
        self.ablation_no_residual_gain = max(0.0, float(cfg.ablation_no_residual_gain))
        self.ablation_disable_spatial_prior = bool(cfg.ablation_disable_spatial_prior)
        self.ablation_direct_delta_blend = bool(cfg.ablation_direct_delta_blend)
        self.raw_latent_splat_highway = bool(cfg.raw_latent_splat_highway)
        self.ablation_skip_clean = bool(cfg.ablation_skip_clean)
        self.ablation_skip_blur = bool(cfg.ablation_skip_blur)
        self.skip_bottleneck_channels = max(1, int(cfg.skip_bottleneck_channels))
        self.skip_spatial_dropout_p = max(0.0, min(1.0, float(cfg.skip_spatial_dropout_p)))
        self.ablation_decoder_highpass = bool(cfg.ablation_decoder_highpass)
        self.color_highway_gain = float(cfg.color_highway_gain)
        self.pre_integrate_moment_match = bool(cfg.pre_integrate_moment_match)
        self.pre_integrate_moment_blend = max(0.0, min(1.0, float(cfg.pre_integrate_moment_blend)))
        self.output_moment_match = bool(cfg.output_moment_match)
        self.output_moment_match_eps = max(1e-8, float(cfg.output_moment_match_eps))
        self.output_moment_match_train_only = bool(cfg.output_moment_match_train_only)
        self.use_style_blender = bool(cfg.use_style_blender)
        if self.upsample_blur_kernel not in {"box3", "gaussian3"}:
            self.upsample_blur_kernel = "box3"

        tokenizer_kind = str(cfg.style_tokenizer).strip().lower()
        if tokenizer_kind != "factorized":
            raise ValueError(f"Unsupported style_tokenizer: {cfg.style_tokenizer}")
        self.style_tokenizer = FactorizedStyleTokenizer(
            num_styles=self.num_styles,
            style_dim=style_dim,
            identity_dim=int(cfg.tokenizer_identity_dim),
            texture_dim=int(cfg.tokenizer_texture_dim),
            geometry_dim=int(cfg.tokenizer_geometry_dim),
            init_std=float(cfg.tokenizer_init_std),
            projection_mode=str(cfg.tokenizer_projection_mode),
            residual_gain=float(cfg.tokenizer_residual_gain),
            num_atoms=tokenizer_num_atoms,
            num_prototypes=int(cfg.tokenizer_num_prototypes),
            atom_temperature=float(cfg.tokenizer_atom_temperature),
            field_dropout_p=float(cfg.tokenizer_field_dropout_p),
            code_l2_norm=bool(cfg.tokenizer_code_l2_norm),
            code_scale=float(cfg.tokenizer_code_scale),
            atom_topk=int(cfg.tokenizer_atom_topk),
            atom_hard_eval=bool(cfg.tokenizer_atom_hard_eval),
        )
        self.style_code_content_router: nn.Module | None = None
        self.style_code_content_style_gate: nn.Embedding | None = None
        if self.tokenizer_content_adaptive:
            router_in = self.body_channels * 4 + 1
            hidden = max(4, int(cfg.tokenizer_content_hidden_dim))
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
                init_ratio = float(cfg.tokenizer_content_style_gate_init) / self.tokenizer_content_style_gate_max
                init_ratio = max(1e-4, min(1.0 - 1e-4, init_ratio))
                init_logit = torch.logit(torch.tensor(init_ratio, dtype=torch.float32)).item()
                nn.init.constant_(self.style_code_content_style_gate.weight, init_logit)

        # Learnable style priors for inference without reference image. The
        # default class prior keeps historical behavior; prototype/VQ modes are
        # opt-in representation probes.
        self.style_spatial_mode = str(cfg.style_spatial_mode).strip().lower()
        if self.style_spatial_mode not in {"class", "prototype", "content_guided", "vq", "vq_content_guided"}:
            self.style_spatial_mode = "class"
        self.style_spatial_num_prototypes = max(1, int(cfg.style_spatial_num_prototypes))
        self.style_spatial_routing_temperature = max(1e-3, float(cfg.style_spatial_routing_temperature))
        self.style_spatial_id_16 = nn.Parameter(torch.zeros(self.num_styles, self.body_channels, 16, 16))
        nn.init.normal_(self.style_spatial_id_16, mean=0.0, std=0.02)
        self.style_spatial_proto_16: nn.Parameter | None = None
        self.style_spatial_atoms_16: nn.Parameter | None = None
        self.style_spatial_logits: nn.Embedding | None = None
        self.style_spatial_content_router: nn.Module | None = None
        if self.style_spatial_mode in {"prototype", "content_guided"}:
            self.style_spatial_proto_16 = nn.Parameter(
                torch.zeros(self.num_styles, self.style_spatial_num_prototypes, self.body_channels, 16, 16)
            )
            nn.init.normal_(self.style_spatial_proto_16, mean=0.0, std=0.02)
            self.style_spatial_logits = nn.Embedding(self.num_styles, self.style_spatial_num_prototypes)
            nn.init.zeros_(self.style_spatial_logits.weight)
        if self.style_spatial_mode in {"vq", "vq_content_guided"}:
            self.style_spatial_atoms_16 = nn.Parameter(torch.zeros(self.num_atoms, self.body_channels, 16, 16))
            nn.init.normal_(self.style_spatial_atoms_16, mean=0.0, std=0.02)
            self.style_spatial_logits = nn.Embedding(self.num_styles, self.num_atoms)
            nn.init.zeros_(self.style_spatial_logits.weight)
        if self.style_spatial_mode in {"content_guided", "vq_content_guided"}:
            router_out = self.style_spatial_num_prototypes if self.style_spatial_mode == "content_guided" else self.num_atoms
            router_in = self.body_channels * 4 + 1
            hidden = max(4, int(cfg.style_spatial_content_hidden_dim))
            self.style_spatial_content_router = nn.Sequential(
                nn.LayerNorm(router_in),
                nn.Linear(router_in, hidden),
                nn.SiLU(),
                nn.Linear(hidden, router_out),
            )
            last = self.style_spatial_content_router[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

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
                    style_attn_sharpen_scale=self.style_attn_sharpen_scale,
                    feature_attn_num_heads=self.feature_attn_num_heads,
                    style_attn_temperature=self.style_attn_temperature,
                    window_attn_window_size=self.window_attn_window_size,
                )
                for _ in range(self.num_hires_blocks)
            ]
        )
        self.down = nn.Conv2d(self.lift_channels, self.body_channels, kernel_size=4, stride=2, padding=1)

        self.body_blocks = nn.ModuleList(
            [
                SemanticCrossAttn(
                    dim=self.body_channels,
                    num_groups=num_groups,
                    temperature=self.semantic_attn_temperature,
                    paint_only=self.use_style_blender,
                    routing_mode=self.semantic_attn_routing_mode,
                    sinkhorn_iters=self.semantic_sinkhorn_iters,
                    gumbel_tau=self.semantic_gumbel_tau,
                )
                for _ in range(self.num_res_blocks)
            ]
        )
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
