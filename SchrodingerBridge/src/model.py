from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_schema import BridgeConfig, ModelConfig
from lancet_blocks import StyleMaps, _gumbel_hard_attention, _sinkhorn_attention
from lancet_backbone import LatentAdaCUT, count_parameters
from style_families import SOLVER_FAMILIES, normalize_family, validate_i2sb_contract
from utils.diffeomorphic import apply_texture_aligned_diffeomorphic_stroke


logger = logging.getLogger(__name__)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half <= 0:
        return t.unsqueeze(-1)
    scale = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -scale)
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TimeConditionedLANCETBridge(LatentAdaCUT):
    def __init__(self, config: ModelConfig) -> None:
        bridge_config = config.validated()
        super().__init__(bridge_config)
        self.time_dim = int(bridge_config.time_dim)
        self.solver_family = normalize_family(
            str(getattr(bridge_config, "solver_family", "euler_legacy")),
            allowed=SOLVER_FAMILIES,
            default="euler_legacy",
        )
        self.velocity_head_mode = str(bridge_config.velocity_head_mode).strip().lower()
        self.velocity_tanh_limit = max(1e-3, float(bridge_config.velocity_tanh_limit))
        self.transport_prediction_mode = str(getattr(bridge_config, "transport_prediction_mode", "velocity")).strip().lower()
        if self.transport_prediction_mode not in {"velocity", "endpoint"}:
            self.transport_prediction_mode = "velocity"
        self.endpoint_parameterization = str(getattr(bridge_config, "endpoint_parameterization", "absolute")).strip().lower()
        if self.endpoint_parameterization not in {"absolute", "residual", "blend", "orthogonal_lowhigh"}:
            self.endpoint_parameterization = "absolute"
        self.endpoint_residual_blend = float(getattr(bridge_config, "endpoint_residual_blend", 0.0))
        self.endpoint_residual_blend = min(1.0, max(0.0, self.endpoint_residual_blend))
        self.endpoint_orthogonal_kernel = max(1, int(getattr(bridge_config, "endpoint_orthogonal_kernel", 5)))
        if self.endpoint_orthogonal_kernel % 2 == 0:
            self.endpoint_orthogonal_kernel += 1
        self.endpoint_orthogonal_high_scale = max(
            0.0,
            float(getattr(bridge_config, "endpoint_orthogonal_high_scale", 1.0)),
        )
        self.endpoint_orthogonal_low_anchor = min(
            1.0,
            max(0.0, float(getattr(bridge_config, "endpoint_orthogonal_low_anchor", 1.0))),
        )
        self.endpoint_orthogonal_low_mode = str(
            getattr(bridge_config, "endpoint_orthogonal_low_mode", "all")
        ).strip().lower()
        if self.endpoint_orthogonal_low_mode not in {"all", "channel_mean"}:
            self.endpoint_orthogonal_low_mode = "all"
        self.bridge_noise_schedule = str(getattr(bridge_config, "bridge_noise_schedule", "auto")).strip().lower()
        if self.bridge_noise_schedule not in {"auto", "exact_brownian", "delayed_window"}:
            self.bridge_noise_schedule = "auto"
        validate_i2sb_contract(
            solver_family=self.solver_family,
            transport_prediction_mode=self.transport_prediction_mode,
            objective_mode=str(getattr(bridge_config, "objective_mode", "")),
            loss_type=str(getattr(bridge_config, "loss_type", "")),
            bridge_noise_schedule=self.bridge_noise_schedule,
        )
        self.transport_endpoint_scale = max(1e-3, float(getattr(bridge_config, "transport_endpoint_scale", 4.0)))
        self.objective_mode = str(getattr(bridge_config, "objective_mode", "")).strip().lower()
        self.loss_type = str(getattr(bridge_config, "loss_type", "")).strip().lower()
        self.bridge_sigma = max(0.0, float(getattr(bridge_config, "bridge_sigma", 0.0)))
        self.i2sb_predictor_time_floor = max(0.0, float(getattr(bridge_config, "i2sb_predictor_time_floor", 0.0)))
        self.i2sb_noise_family = str(getattr(bridge_config, "i2sb_noise_family", "gaussian")).strip().lower()
        if self.i2sb_noise_family not in {"gaussian", "style_covariant"}:
            self.i2sb_noise_family = "gaussian"
        self.i2sb_style_noise_amplitude_power = max(
            0.0,
            float(getattr(bridge_config, "i2sb_style_noise_amplitude_power", 1.0)),
        )
        self.endpoint_velocity_time_floor = float(getattr(bridge_config, "endpoint_velocity_time_floor", 0.05))
        if self.endpoint_velocity_time_floor < 0.01:
            raise ValueError(
                "model.endpoint_velocity_time_floor must be >= 0.01 for endpoint velocity mode. "
                "Lower floors amplify endpoint deltas near t=1 and can silently destabilize training."
            )
        self.solver_corrector_mode = self._normalize_solver_corrector_mode(
            str(getattr(bridge_config, "solver_corrector_mode", "none"))
        )
        self.transport_stats_mode = self._normalize_transport_stats_mode(
            str(getattr(bridge_config, "transport_stats_mode", "none"))
        )
        self.transport_stats_bank_path = str(getattr(bridge_config, "transport_stats_bank_path", "") or "").strip()
        self.transport_stats_bank_required = bool(getattr(bridge_config, "transport_stats_bank_required", False))
        self.transport_stats_eps = max(1e-8, float(getattr(bridge_config, "transport_stats_eps", 1e-6)))
        self.allow_style_overdrive = bool(getattr(bridge_config, "allow_style_overdrive", False))
        self.last_i2sb_transport_debug: dict[str, float] = {}
        self.solver_stochastic_noise_scale = max(0.0, float(getattr(bridge_config, "solver_stochastic_noise_scale", 0.0)))
        self.solver_fiber_aligned = bool(getattr(bridge_config, "solver_fiber_aligned", False))
        self.i2sb_fiber_aligned_noise = bool(getattr(bridge_config, "i2sb_fiber_aligned_noise", False))
        self.i2sb_fiber_noise_rms_normalize = bool(getattr(bridge_config, "i2sb_fiber_noise_rms_normalize", True))
        self.i2sb_fiber_project_endpoint = bool(getattr(bridge_config, "i2sb_fiber_project_endpoint", False))
        self.i2sb_fiber_project_noise = bool(getattr(bridge_config, "i2sb_fiber_project_noise", False))
        self.i2sb_fiber_project_kernel = max(1, int(getattr(bridge_config, "i2sb_fiber_project_kernel", 5)))
        if self.i2sb_fiber_project_kernel % 2 == 0:
            self.i2sb_fiber_project_kernel += 1
        self.i2sb_fiber_project_use_gate = bool(getattr(bridge_config, "i2sb_fiber_project_use_gate", False))
        self.i2sb_fiber_project_noise_mode = str(
            getattr(bridge_config, "i2sb_fiber_project_noise_mode", "highpass")
        ).strip().lower()
        if self.i2sb_fiber_project_noise_mode not in {"highpass", "residual_envelope", "residual_direction"}:
            self.i2sb_fiber_project_noise_mode = "highpass"
        self.i2sb_fiber_project_residual_power = max(
            0.0,
            float(getattr(bridge_config, "i2sb_fiber_project_residual_power", 1.0)),
        )
        self.last_i2sb_fiber_noise_debug: dict[str, float] = {}
        self.last_i2sb_fiber_project_debug: dict[str, float] = {}
        self.last_solver_noise_debug: dict[str, float] = {}
        stats_shape = (max(1, int(self.num_styles)), int(self.latent_channels), 1, 1)
        self.register_buffer("_transport_style_stats_mean", torch.zeros(stats_shape, dtype=torch.float32), persistent=False)
        self.register_buffer("_transport_style_stats_std", torch.ones(stats_shape, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "_transport_style_stats_valid",
            torch.zeros((max(1, int(self.num_styles)),), dtype=torch.bool),
            persistent=False,
        )
        self.last_transport_stats_debug: dict[str, float] = {}
        self.bridge_style_dim = int(getattr(self, "style_code_dim", getattr(self.style_tokenizer, "embedding_dim", 0)))
        self.execution_budget_mode = str(getattr(bridge_config, "execution_budget_mode", "none")).strip().lower()
        if self.execution_budget_mode not in {"none", "scalar", "low_high"}:
            self.execution_budget_mode = "none"
        self.execution_budget_log_span = max(0.0, float(getattr(bridge_config, "execution_budget_log_span", 0.22314355131420976)))
        self.execution_budget_feature_dim = int(bridge_config.latent_channels) * 4 + 1
        self.execution_budget_head: nn.Module | None = None
        if self.execution_budget_mode != "none":
            hidden = max(4, int(getattr(bridge_config, "execution_budget_hidden_dim", 64)))
            out_dim = 1 if self.execution_budget_mode == "scalar" else 2
            self.execution_budget_head = nn.Sequential(
                nn.LayerNorm(self.bridge_style_dim + self.execution_budget_feature_dim),
                nn.Linear(self.bridge_style_dim + self.execution_budget_feature_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )
            last = self.execution_budget_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        self.style_injection_mode = str(getattr(bridge_config, "style_injection_mode", "none")).strip().lower()
        if self.style_injection_mode not in {"none", "body", "decoder", "body_decoder"}:
            self.style_injection_mode = "none"
        self.style_injection_form = str(getattr(bridge_config, "style_injection_form", "mixed")).strip().lower()
        if self.style_injection_form not in {"mixed", "carrier_gate", "spatial_carrier_gate"}:
            self.style_injection_form = "mixed"
        self.style_injection_scale = max(0.0, float(getattr(bridge_config, "style_injection_scale", 1.0)))
        self.style_injection_gate_log_span = max(0.0, float(getattr(bridge_config, "style_injection_gate_log_span", 0.4054651081081644)))
        self.style_injection_spatial_kernel = max(1, int(getattr(bridge_config, "style_injection_spatial_kernel", 5)))
        if self.style_injection_spatial_kernel % 2 == 0:
            self.style_injection_spatial_kernel += 1
        self.style_injection_force_highpass = bool(getattr(bridge_config, "style_injection_force_highpass", True))
        self.style_injection_live_init = bool(getattr(bridge_config, "style_injection_live_init", False))
        self.style_injection_live_init_std = max(
            0.0,
            float(getattr(bridge_config, "style_injection_live_init_std", 0.02)),
        )
        injection_in_dim = self.bridge_style_dim + self.execution_budget_feature_dim
        self.body_style_injector: nn.Module | None = None
        self.decoder_style_injector: nn.Module | None = None
        self.body_style_carrier: nn.Module | None = None
        self.body_content_gate: nn.Module | None = None
        self.decoder_style_carrier: nn.Module | None = None
        self.decoder_content_gate: nn.Module | None = None
        self.body_style_spatial_proj: nn.Module | None = None
        self.body_structure_gate: nn.Module | None = None
        self.decoder_style_spatial_proj: nn.Module | None = None
        self.decoder_structure_gate: nn.Module | None = None
        if self.style_injection_mode in {"body", "body_decoder"}:
            if self.style_injection_form == "carrier_gate":
                self.body_style_carrier, self.body_content_gate = self._make_carrier_gate_injector(
                    self.bridge_style_dim,
                    self.execution_budget_feature_dim,
                    int(self.body_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                    live_init=self.style_injection_live_init,
                    live_init_std=self.style_injection_live_init_std,
                )
            elif self.style_injection_form == "spatial_carrier_gate":
                self.body_style_spatial_proj, self.body_content_gate, self.body_structure_gate = self._make_spatial_carrier_gate_injector(
                    int(self.body_channels),
                    int(self.body_channels),
                    self.execution_budget_feature_dim,
                    int(self.latent_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                    live_init=self.style_injection_live_init,
                    live_init_std=self.style_injection_live_init_std,
                )
            else:
                self.body_style_injector = self._make_style_injector(
                    injection_in_dim,
                    int(self.body_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                    live_init=self.style_injection_live_init,
                    live_init_std=self.style_injection_live_init_std,
                )
        if self.style_injection_mode in {"decoder", "body_decoder"}:
            if self.style_injection_form == "carrier_gate":
                self.decoder_style_carrier, self.decoder_content_gate = self._make_carrier_gate_injector(
                    self.bridge_style_dim,
                    self.execution_budget_feature_dim,
                    int(self.lift_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                    live_init=self.style_injection_live_init,
                    live_init_std=self.style_injection_live_init_std,
                )
            elif self.style_injection_form == "spatial_carrier_gate":
                self.decoder_style_spatial_proj, self.decoder_content_gate, self.decoder_structure_gate = self._make_spatial_carrier_gate_injector(
                    int(self.body_channels),
                    int(self.lift_channels),
                    self.execution_budget_feature_dim,
                    int(self.latent_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                    live_init=self.style_injection_live_init,
                    live_init_std=self.style_injection_live_init_std,
                )
            else:
                self.decoder_style_injector = self._make_style_injector(
                    injection_in_dim,
                    int(self.lift_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                    live_init=self.style_injection_live_init,
                    live_init_std=self.style_injection_live_init_std,
                )
        self.style_delta_mode = str(getattr(bridge_config, "style_delta_mode", "none")).strip().lower()
        if self.style_delta_mode not in {"none", "basis", "predec_section", "head_adapter"}:
            self.style_delta_mode = "none"
        self.style_delta_rank = max(1, int(getattr(bridge_config, "style_delta_rank", 4)))
        self.style_delta_scale = max(0.0, float(getattr(bridge_config, "style_delta_scale", 0.15)))
        self.style_delta_highpass_kernel = max(1, int(getattr(bridge_config, "style_delta_highpass_kernel", 5)))
        if self.style_delta_highpass_kernel % 2 == 0:
            self.style_delta_highpass_kernel += 1
        self.style_delta_force_highpass = bool(getattr(bridge_config, "style_delta_force_highpass", True))
        self.style_delta_basis_proj: nn.Conv2d | None = None
        self.style_delta_weight_head: nn.Module | None = None
        self.style_section_basis_proj: nn.Conv2d | None = None
        self.style_section_weight_head: nn.Module | None = None
        self.style_section_out: nn.Conv2d | None = None
        self.style_section_scale = max(0.0, float(getattr(bridge_config, "style_section_scale", 0.10)))
        self.style_section_force_highpass = bool(getattr(bridge_config, "style_section_force_highpass", True))
        self.style_head_adapter_in: nn.Conv2d | None = None
        self.style_head_adapter_film: nn.Module | None = None
        self.style_head_adapter_out: nn.Conv2d | None = None
        self.style_head_adapter_scale = max(0.0, float(getattr(bridge_config, "style_head_adapter_scale", 0.10)))
        self.style_head_adapter_force_highpass = bool(getattr(bridge_config, "style_head_adapter_force_highpass", False))
        self.style_head_adapter_use_gate = bool(getattr(bridge_config, "style_head_adapter_use_gate", False))
        self.style_head_adapter_gate_power = max(0.0, float(getattr(bridge_config, "style_head_adapter_gate_power", 1.0)))
        self.last_style_delta_debug: dict[str, float] = {}
        if self.style_delta_mode == "basis" and self.style_delta_scale > 0.0:
            self.style_delta_basis_proj = nn.Conv2d(
                int(self.lift_channels),
                int(self.latent_channels) * int(self.style_delta_rank),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            hidden = max(4, int(getattr(bridge_config, "style_delta_hidden_dim", 64)))
            self.style_delta_weight_head = nn.Sequential(
                nn.LayerNorm(int(self.bridge_style_dim)),
                nn.Linear(int(self.bridge_style_dim), hidden),
                nn.SiLU(),
                nn.Linear(hidden, int(self.style_delta_rank)),
            )
            nn.init.normal_(self.style_delta_basis_proj.weight, mean=0.0, std=0.02)
            if self.style_delta_basis_proj.bias is not None:
                nn.init.zeros_(self.style_delta_basis_proj.bias)
            last = self.style_delta_weight_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        if self.style_delta_mode == "predec_section" and self.style_section_scale > 0.0:
            rank = int(self.style_delta_rank)
            hidden = max(4, int(getattr(bridge_config, "style_section_hidden_dim", 64)))
            self.style_section_basis_proj = nn.Conv2d(
                int(self.lift_channels),
                int(self.lift_channels) * rank,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.style_section_weight_head = nn.Sequential(
                nn.LayerNorm(int(self.bridge_style_dim)),
                nn.Linear(int(self.bridge_style_dim), hidden),
                nn.SiLU(),
                nn.Linear(hidden, rank),
            )
            self.style_section_out = nn.Conv2d(
                int(self.lift_channels),
                int(self.lift_channels),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.normal_(self.style_section_basis_proj.weight, mean=0.0, std=0.02)
            if self.style_section_basis_proj.bias is not None:
                nn.init.zeros_(self.style_section_basis_proj.bias)
            last = self.style_section_weight_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            nn.init.zeros_(self.style_section_out.weight)
            if self.style_section_out.bias is not None:
                nn.init.zeros_(self.style_section_out.bias)
        if self.style_delta_mode == "head_adapter" and self.style_head_adapter_scale > 0.0:
            hidden = max(4, int(getattr(bridge_config, "style_head_adapter_hidden_dim", 32)))
            self.style_head_adapter_in = nn.Conv2d(
                int(self.lift_channels),
                hidden,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.style_head_adapter_film = nn.Sequential(
                nn.LayerNorm(int(self.bridge_style_dim)),
                nn.Linear(int(self.bridge_style_dim), hidden * 2),
            )
            self.style_head_adapter_out = nn.Conv2d(
                hidden,
                int(self.latent_channels),
                kernel_size=3,
                stride=1,
                padding=1,
            )
            nn.init.kaiming_normal_(self.style_head_adapter_in.weight, nonlinearity="linear")
            if self.style_head_adapter_in.bias is not None:
                nn.init.zeros_(self.style_head_adapter_in.bias)
            film_last = self.style_head_adapter_film[-1]
            if isinstance(film_last, nn.Linear):
                nn.init.zeros_(film_last.weight)
                nn.init.zeros_(film_last.bias)
            nn.init.zeros_(self.style_head_adapter_out.weight)
            if self.style_head_adapter_out.bias is not None:
                nn.init.zeros_(self.style_head_adapter_out.bias)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.bridge_style_dim),
            nn.SiLU(),
            nn.Linear(self.bridge_style_dim, self.bridge_style_dim),
        )
        self.proximal_mode = str(getattr(bridge_config, "proximal_mode", "off")).strip().lower()
        if self.proximal_mode not in {"off", "crossattn_texture"}:
            self.proximal_mode = "off"
        self.proximal_hidden_channels = max(4, int(getattr(bridge_config, "proximal_hidden_channels", self.latent_channels)))
        self.proximal_highpass_kernel = max(1, int(getattr(bridge_config, "proximal_highpass_kernel", 5)))
        if self.proximal_highpass_kernel % 2 == 0:
            self.proximal_highpass_kernel += 1
        self.proximal_residual_energy_weight = max(0.0, float(getattr(bridge_config, "proximal_residual_energy_weight", 0.0)))
        self.proximal_trust_ratio = max(0.0, float(getattr(bridge_config, "proximal_trust_ratio", 0.0)))
        self.proximal_trust_weight = max(0.0, float(getattr(bridge_config, "proximal_trust_weight", 0.0)))
        self.proximal_clamp_ratio = max(0.0, float(getattr(bridge_config, "proximal_clamp_ratio", 0.0)))
        self.proximal_clamp_ratio_mid = max(0.0, float(getattr(bridge_config, "proximal_clamp_ratio_mid", 0.0)))
        self.proximal_clamp_ratio_end = max(0.0, float(getattr(bridge_config, "proximal_clamp_ratio_end", 0.0)))
        self.proximal_clamp_schedule = str(getattr(bridge_config, "proximal_clamp_schedule", "linear")).strip().lower()
        if self.proximal_clamp_schedule not in {"linear", "hold_linear", "hold_two_stage"}:
            self.proximal_clamp_schedule = "linear"
        self.proximal_clamp_hold_epochs = max(0, int(getattr(bridge_config, "proximal_clamp_hold_epochs", 0)))
        self.proximal_clamp_release_epochs = max(0, int(getattr(bridge_config, "proximal_clamp_release_epochs", 0)))
        self.proximal_clamp_mid_hold_epochs = max(0, int(getattr(bridge_config, "proximal_clamp_mid_hold_epochs", 0)))
        self.proximal_clamp_second_release_epochs = max(0, int(getattr(bridge_config, "proximal_clamp_second_release_epochs", 0)))
        self.proximal_force_highpass = bool(getattr(bridge_config, "proximal_force_highpass", True))
        self.proximal_bind_terminal_losses = bool(getattr(bridge_config, "proximal_bind_terminal_losses", True))
        self.record_base_endpoint_metrics = bool(getattr(bridge_config, "record_base_endpoint_metrics", False))
        self.proximal_attn_routing_mode = str(getattr(bridge_config, "proximal_attn_routing_mode", "softmax")).strip().lower()
        if self.proximal_attn_routing_mode not in {"softmax", "sinkhorn", "gumbel_hard"}:
            self.proximal_attn_routing_mode = "softmax"
        self.proximal_attn_sinkhorn_iters = max(1, int(getattr(bridge_config, "proximal_attn_sinkhorn_iters", 3)))
        self.proximal_attn_gumbel_tau = max(1e-3, float(getattr(bridge_config, "proximal_attn_gumbel_tau", 1.0)))
        self.proximal_attn_q: nn.Conv2d | None = None
        self.proximal_attn_k: nn.Conv2d | None = None
        self.proximal_attn_v: nn.Conv2d | None = None
        self.proximal_attn_out: nn.Conv2d | None = None
        self.proximal_style_tokens: nn.Linear | None = None
        if self.proximal_mode == "crossattn_texture":
            hidden = int(self.proximal_hidden_channels)
            self.proximal_attn_q = nn.Conv2d(int(self.latent_channels), hidden, kernel_size=1, stride=1, padding=0)
            self.proximal_attn_k = nn.Conv2d(int(self.body_channels), hidden, kernel_size=1, stride=1, padding=0)
            self.proximal_attn_v = nn.Conv2d(int(self.body_channels), hidden, kernel_size=1, stride=1, padding=0)
            self.proximal_attn_out = nn.Conv2d(hidden, int(self.latent_channels), kernel_size=1, stride=1, padding=0)
            self.proximal_style_tokens = nn.Linear(int(self.bridge_style_dim), int(self.body_channels))
            for mod in (self.proximal_attn_q, self.proximal_attn_k, self.proximal_attn_v):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            # Keep the proximal branch small, but do not make it a hard zero path.
            nn.init.normal_(self.proximal_attn_out.weight, mean=0.0, std=0.01)
            if self.proximal_attn_out.bias is not None:
                nn.init.zeros_(self.proximal_attn_out.bias)
            nn.init.normal_(self.proximal_style_tokens.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.proximal_style_tokens.bias)
        self.profile_modules = False
        self.profile_sync_cuda = False
        self.last_profile: dict[str, float] = {}
        self.last_proximal_residual: torch.Tensor | None = None
        self.last_base_endpoint: torch.Tensor | None = None
        self.last_final_endpoint: torch.Tensor | None = None
        self.last_proximal_clamp_scale: torch.Tensor | None = None
        self.last_solver_noise_debug = {}
        self.current_epoch: int = 1
        self.total_epochs: int = 1
        self.runtime_conditioning: dict[str, Any] = {}

    def _profile_start(self, ref: torch.Tensor) -> float:
        if not bool(getattr(self, "profile_modules", False)):
            return 0.0
        if bool(getattr(self, "profile_sync_cuda", False)) and ref.device.type == "cuda":
            torch.cuda.synchronize(ref.device)
        return time.perf_counter()

    def _profile_end(self, name: str, start: float, ref: torch.Tensor) -> None:
        if not bool(getattr(self, "profile_modules", False)):
            return
        if bool(getattr(self, "profile_sync_cuda", False)) and ref.device.type == "cuda":
            torch.cuda.synchronize(ref.device)
        self.last_profile[name] = self.last_profile.get(name, 0.0) + max(0.0, time.perf_counter() - start)

    def set_runtime_conditioning(self, payload: Mapping[str, Any] | None) -> None:
        if payload is None:
            self.runtime_conditioning = {}
            return
        self.runtime_conditioning = dict(payload)

    def clear_runtime_conditioning(self) -> None:
        self.runtime_conditioning = {}

    def clear_runtime_caches(self, *, clear_conditioning: bool = True) -> None:
        if clear_conditioning:
            self.runtime_conditioning = {}
        self.last_profile = {}
        self.last_proximal_residual = None
        self.last_base_endpoint = None
        self.last_final_endpoint = None
        self.last_proximal_clamp_scale = None
        self.last_output_appearance_debug = {}
        self.last_output_style_context = None
        self.last_solver_noise_debug = {}
        self.last_transport_stats_debug = {}
        self.last_style_delta_debug = {}
        for module in self.modules():
            if hasattr(module, "last_attn"):
                setattr(module, "last_attn", None)
            if hasattr(module, "last_k"):
                setattr(module, "last_k", None)

    def _normalize_transport_stats_mode(self, mode: str) -> str:
        normalized = str(mode or "none").strip().lower()
        aliases = {
            "": "none",
            "off": "none",
            "disabled": "none",
            "style_bank_terminal": "terminal_affine",
            "bank_terminal": "terminal_affine",
            "style_bank_terminal_affine": "terminal_affine",
            "terminal": "terminal_affine",
            "style_bank_normalized": "normalized_solver",
            "bank_normalized": "normalized_solver",
            "normalized": "normalized_solver",
            "normalized_track": "normalized_solver",
            "style_bank_normalized_solver": "normalized_solver",
        }
        normalized = aliases.get(normalized, normalized)
        valid = {"none", "terminal_affine", "normalized_solver"}
        if normalized not in valid:
            raise ValueError(
                f"Unsupported model.transport_stats_mode={mode!r}; "
                "expected one of 'none', 'terminal_affine', or 'normalized_solver'."
            )
        return normalized

    def load_transport_style_stats_bank(self, source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        path: Path | None = None
        if isinstance(source, (str, Path)):
            path = Path(source)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        else:
            payload = dict(source)
        if not isinstance(payload, Mapping):
            raise ValueError("transport stats bank must be a mapping or a .pt payload mapping")
        means = payload.get("means", payload.get("style_means", payload.get("latent_means")))
        stds = payload.get("stds", payload.get("style_stds", payload.get("latent_stds")))
        if means is None or stds is None:
            raise ValueError("transport stats bank is missing means/stds tensors")
        means_t = torch.as_tensor(means, dtype=torch.float32)
        stds_t = torch.as_tensor(stds, dtype=torch.float32)
        if means_t.ndim == 2:
            means_t = means_t.unsqueeze(-1).unsqueeze(-1)
        if stds_t.ndim == 2:
            stds_t = stds_t.unsqueeze(-1).unsqueeze(-1)
        if means_t.ndim != 4 or stds_t.ndim != 4:
            raise ValueError(
                "transport stats bank expects tensors shaped [num_styles, channels, 1, 1] "
                f"or [num_styles, channels]; got means={tuple(means_t.shape)} stds={tuple(stds_t.shape)}"
            )
        expected_shape = tuple(self._transport_style_stats_mean.shape)
        if tuple(means_t.shape) != expected_shape or tuple(stds_t.shape) != expected_shape:
            raise ValueError(
                "transport stats bank shape mismatch: "
                f"expected {expected_shape}, got means={tuple(means_t.shape)} stds={tuple(stds_t.shape)}"
            )
        valid_mask = payload.get("valid_mask", payload.get("valid", None))
        if valid_mask is None:
            valid_t = torch.ones((means_t.shape[0],), dtype=torch.bool)
        else:
            valid_t = torch.as_tensor(valid_mask, dtype=torch.bool).view(-1)
            if int(valid_t.numel()) != int(means_t.shape[0]):
                raise ValueError(
                    "transport stats valid_mask length mismatch: "
                    f"expected {int(means_t.shape[0])}, got {int(valid_t.numel())}"
                )
        self._transport_style_stats_mean.copy_(means_t)
        self._transport_style_stats_std.copy_(stds_t.clamp_min(self.transport_stats_eps))
        self._transport_style_stats_valid.copy_(valid_t)
        if path is not None:
            self.transport_stats_bank_path = str(path)
        loaded = int(valid_t.sum().item())
        return {
            "loaded_styles": loaded,
            "num_styles": int(means_t.shape[0]),
            "channels": int(means_t.shape[1]),
            "path": str(path) if path is not None else "",
        }

    def _transport_stats_tensor_metrics(
        self,
        ref: torch.Tensor,
        debug: Mapping[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        raw = debug if isinstance(debug, Mapping) else self.last_transport_stats_debug
        metrics: dict[str, torch.Tensor] = {}
        if not isinstance(raw, Mapping):
            return metrics
        for key, value in raw.items():
            if isinstance(value, (int, float, bool)):
                metrics[str(key)] = ref.new_tensor(float(value), dtype=torch.float32)
        return metrics

    def _transport_style_stats_bank_loaded(self) -> bool:
        valid = getattr(self, "_transport_style_stats_valid", None)
        return bool(torch.is_tensor(valid) and bool(valid.any().item()))

    def _resolve_transport_style_targets(
        self,
        *,
        style_id: torch.Tensor | int,
        batch: int,
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self._transport_style_stats_bank_loaded():
            return None
        style_id_t = self._normalize_style_id_input(style_id, device=ref.device)
        if style_id_t.numel() == 1 and batch > 1:
            style_id_t = style_id_t.expand(batch)
        elif style_id_t.numel() != batch:
            raise ValueError(f"style_id batch mismatch for transport stats: expected {batch} or 1, got {style_id_t.numel()}")
        valid_mask = self._transport_style_stats_valid.to(device=ref.device)
        selected_valid = valid_mask.index_select(0, style_id_t)
        if not bool(selected_valid.all().item()):
            missing = torch.nonzero(~selected_valid, as_tuple=False).flatten().tolist()
            if self.transport_stats_bank_required:
                raise RuntimeError(f"transport stats bank missing valid entries for batch positions {missing}")
            return None
        mean_bank = self._transport_style_stats_mean.to(device=ref.device, dtype=ref.dtype)
        std_bank = self._transport_style_stats_std.to(device=ref.device, dtype=ref.dtype)
        return (
            mean_bank.index_select(0, style_id_t),
            std_bank.index_select(0, style_id_t).clamp_min(self.transport_stats_eps),
        )

    def _build_transport_stats_context(
        self,
        source: torch.Tensor,
        *,
        style_id: torch.Tensor | int,
    ) -> dict[str, torch.Tensor] | None:
        mode = self.transport_stats_mode
        base_debug = {
            "transport_stats_active": 0.0,
            "transport_stats_bank_loaded": 1.0 if self._transport_style_stats_bank_loaded() else 0.0,
            "transport_stats_mode_terminal_affine": 1.0 if mode == "terminal_affine" else 0.0,
            "transport_stats_mode_normalized_solver": 1.0 if mode == "normalized_solver" else 0.0,
            "transport_stats_source_mean_abs": 0.0,
            "transport_stats_source_std_mean": 0.0,
            "transport_stats_target_mean_abs": 0.0,
            "transport_stats_target_std_mean": 0.0,
            "transport_stats_mean_delta": 0.0,
            "transport_stats_std_delta": 0.0,
            "transport_stats_valid_styles": float(self._transport_style_stats_valid.sum().item()),
            "transport_stats_missing_bank": 0.0,
        }
        if mode == "none":
            self.last_transport_stats_debug = base_debug
            return None
        targets = self._resolve_transport_style_targets(style_id=style_id, batch=int(source.shape[0]), ref=source)
        if targets is None:
            base_debug["transport_stats_missing_bank"] = 1.0
            self.last_transport_stats_debug = base_debug
            return None
        target_mean, target_std = targets
        source_mean = source.mean(dim=(2, 3), keepdim=True)
        source_std = source.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.transport_stats_eps)
        base_debug.update(
            {
                "transport_stats_active": 1.0,
                "transport_stats_source_mean_abs": float(source_mean.detach().abs().mean().cpu().item()),
                "transport_stats_source_std_mean": float(source_std.detach().mean().cpu().item()),
                "transport_stats_target_mean_abs": float(target_mean.detach().abs().mean().cpu().item()),
                "transport_stats_target_std_mean": float(target_std.detach().mean().cpu().item()),
                "transport_stats_mean_delta": float((target_mean - source_mean).detach().abs().mean().cpu().item()),
                "transport_stats_std_delta": float((target_std - source_std).detach().abs().mean().cpu().item()),
            }
        )
        self.last_transport_stats_debug = base_debug
        return {
            "source_mean": source_mean.to(dtype=source.dtype),
            "source_std": source_std.to(dtype=source.dtype),
            "target_mean": target_mean.to(dtype=source.dtype),
            "target_std": target_std.to(dtype=source.dtype),
        }

    def _prepare_transport_stats_input(
        self,
        source: torch.Tensor,
        *,
        style_id: torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        ctx = self._build_transport_stats_context(source, style_id=style_id)
        if ctx is None:
            return source, source, None
        if self.transport_stats_mode != "normalized_solver":
            return source, source, ctx
        normalized = ((source - ctx["source_mean"]) / ctx["source_std"]).to(dtype=source.dtype)
        return normalized, normalized, ctx

    def restore_transport_output(
        self,
        latent: torch.Tensor,
        *,
        style_id: torch.Tensor | int,
    ) -> torch.Tensor:
        if self.transport_stats_mode == "none":
            return latent
        targets = self._resolve_transport_style_targets(style_id=style_id, batch=int(latent.shape[0]), ref=latent)
        if targets is None:
            return latent
        target_mean, target_std = targets
        if self.transport_stats_mode == "normalized_solver":
            return (latent * target_std + target_mean).to(dtype=latent.dtype)
        pred_mean = latent.mean(dim=(2, 3), keepdim=True)
        pred_std = latent.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.transport_stats_eps)
        return (((latent - pred_mean) / pred_std) * target_std + target_mean).to(dtype=latent.dtype)

    def prepare_transport_training_pair(
        self,
        *,
        content: torch.Tensor,
        target: torch.Tensor,
        style_id: torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        ctx = self._build_transport_stats_context(content, style_id=style_id)
        metrics = self._transport_stats_tensor_metrics(content)
        if ctx is None or self.transport_stats_mode != "normalized_solver":
            return content, target, metrics
        content_norm = ((content - ctx["source_mean"]) / ctx["source_std"]).to(dtype=content.dtype)
        target_norm = ((target - ctx["target_mean"]) / ctx["target_std"]).to(dtype=target.dtype)
        return content_norm, target_norm, metrics

    def _fiber_aligned_solver_noise(
        self,
        reference: torch.Tensor,
        noise: torch.Tensor,
        *,
        noise_scale: float,
    ) -> torch.Tensor:
        debug: dict[str, float] = {
            "fiber_gate_active": 0.0,
            "fiber_gate_mean": 0.0,
            "fiber_gate_rms": 0.0,
            "noise_scale": float(noise_scale),
            "isotropic_or_fiber": 1.0 if bool(getattr(self, "solver_fiber_aligned", False)) else 0.0,
        }
        if not bool(getattr(self, "solver_fiber_aligned", False)):
            self.last_solver_noise_debug = debug
            return noise
        cached = getattr(self, "last_output_style_context", None)
        style_maps = cached.get("style_maps") if isinstance(cached, dict) else None
        gate = getattr(style_maps, "gate_16", None)
        if not torch.is_tensor(gate):
            self.last_solver_noise_debug = debug
            return noise
        gate = gate.to(device=reference.device).float()
        gate = torch.sigmoid(gate).clamp(0.0, 1.0)
        if gate.shape[-2:] != reference.shape[-2:]:
            gate = F.interpolate(gate, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        if gate.shape[1] != reference.shape[1]:
            if gate.shape[1] == 1:
                gate = gate.expand(reference.shape[0], reference.shape[1], reference.shape[2], reference.shape[3])
            else:
                gate = gate.mean(dim=1, keepdim=True).expand(reference.shape[0], reference.shape[1], reference.shape[2], reference.shape[3])
        gate_mean = gate.detach().float().mean()
        debug.update(
            {
                "fiber_gate_active": 1.0,
                "fiber_gate_mean": float(gate_mean.cpu().item()),
                "fiber_gate_rms": float(gate.detach().float().square().mean().sqrt().cpu().item()),
            }
        )
        self.last_solver_noise_debug = debug
        return noise * gate.to(dtype=noise.dtype)

    def _i2sb_fiber_aligned_noise(
        self,
        reference: torch.Tensor,
        noise: torch.Tensor,
        *,
        noise_scale: float,
    ) -> torch.Tensor:
        debug: dict[str, float] = {
            "fiber_noise_active": 0.0,
            "fiber_gate_mean": 0.0,
            "fiber_gate_rms": 0.0,
            "fiber_noise_scale": float(noise_scale),
            "fiber_noise_rms_normalize": float(bool(getattr(self, "i2sb_fiber_noise_rms_normalize", True))),
        }
        if not bool(getattr(self, "i2sb_fiber_aligned_noise", False)):
            self.last_i2sb_transport_debug.update(debug)
            self.last_i2sb_fiber_noise_debug = debug
            return noise
        cached = getattr(self, "last_output_style_context", None)
        style_maps = cached.get("style_maps") if isinstance(cached, dict) else None
        gate = getattr(style_maps, "gate_16", None)
        if not torch.is_tensor(gate):
            self.last_i2sb_transport_debug.update(debug)
            self.last_i2sb_fiber_noise_debug = debug
            return noise
        gate = gate.to(device=reference.device).float()
        gate = torch.sigmoid(gate).clamp(0.0, 1.0)
        if gate.shape[-2:] != reference.shape[-2:]:
            gate = F.interpolate(gate, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        if gate.shape[0] == 1 and reference.shape[0] > 1:
            gate = gate.expand(reference.shape[0], -1, -1, -1)
        if gate.shape[1] != reference.shape[1]:
            gate = gate.mean(dim=1, keepdim=True).expand(reference.shape[0], reference.shape[1], reference.shape[2], reference.shape[3])
        gate_rms = gate.detach().float().square().mean().sqrt().clamp_min(1e-6)
        gate_weight = gate / gate_rms if bool(getattr(self, "i2sb_fiber_noise_rms_normalize", True)) else gate
        debug.update(
            {
                "fiber_noise_active": 1.0,
                "fiber_gate_mean": float(gate.detach().float().mean().cpu().item()),
                "fiber_gate_rms": float(gate_rms.cpu().item()),
            }
        )
        self.last_i2sb_transport_debug.update(debug)
        self.last_i2sb_fiber_noise_debug = debug
        return noise * gate_weight.to(dtype=noise.dtype)

    def _i2sb_lowpass(self, tensor: torch.Tensor) -> torch.Tensor:
        kernel = max(1, int(getattr(self, "i2sb_fiber_project_kernel", 5)))
        if kernel <= 1:
            return tensor.float()
        if kernel % 2 == 0:
            kernel += 1
        return F.avg_pool2d(tensor.float(), kernel_size=kernel, stride=1, padding=kernel // 2)

    def _i2sb_highpass(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() - self._i2sb_lowpass(tensor)

    def _i2sb_sample_transport_noise(
        self,
        reference: torch.Tensor,
        *,
        target_style_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        debug = {
            "style_noise_family_style_covariant": float(self.i2sb_noise_family == "style_covariant"),
            "style_noise_family_gaussian": float(self.i2sb_noise_family == "gaussian"),
            "style_noise_bank_active": 0.0,
            "style_noise_amp_mean": 0.0,
            "style_noise_amp_std": 0.0,
            "style_noise_post_std": 0.0,
            "style_noise_amplitude_power": float(self.i2sb_style_noise_amplitude_power),
            "style_noise_fallback_gaussian": 0.0,
        }
        if self.i2sb_noise_family != "style_covariant" or target_style_latent is None:
            if self.i2sb_noise_family == "style_covariant":
                debug["style_noise_fallback_gaussian"] = 1.0
                self.last_i2sb_transport_debug.update(debug)
            return torch.randn_like(reference)
        style_ref = target_style_latent
        if not torch.is_tensor(style_ref) or style_ref.ndim != 4:
            debug["style_noise_fallback_gaussian"] = 1.0
            self.last_i2sb_transport_debug.update(debug)
            return torch.randn_like(reference)
        style_ref = style_ref.to(device=reference.device, dtype=torch.float32)
        if style_ref.shape[-2:] != reference.shape[-2:]:
            style_ref = F.interpolate(style_ref, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        if style_ref.shape[0] == 1 and reference.shape[0] > 1:
            style_ref = style_ref.expand(reference.shape[0], -1, -1, -1)
        if style_ref.shape[0] != reference.shape[0] or style_ref.shape[1] != reference.shape[1]:
            debug["style_noise_fallback_gaussian"] = 1.0
            self.last_i2sb_transport_debug.update(debug)
            return torch.randn_like(reference)
        fft_style = torch.fft.rfft2(style_ref, norm="ortho")
        amplitude = torch.abs(fft_style).clamp_min(1e-8)
        if abs(self.i2sb_style_noise_amplitude_power - 1.0) > 1e-6:
            amplitude = amplitude.pow(self.i2sb_style_noise_amplitude_power)
        random_phase = (torch.rand_like(amplitude) * 2.0 - 1.0) * math.pi
        phase_complex = torch.complex(torch.cos(random_phase), torch.sin(random_phase))
        fft_noise = amplitude * phase_complex
        style_noise = torch.fft.irfft2(fft_noise, s=reference.shape[-2:], norm="ortho")
        noise_mean = style_noise.mean(dim=(-2, -1), keepdim=True)
        noise_std = style_noise.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = (style_noise - noise_mean) / noise_std
        debug.update(
            {
                "style_noise_bank_active": 1.0,
                "style_noise_amp_mean": float(amplitude.detach().mean().cpu().item()),
                "style_noise_amp_std": float(amplitude.detach().std(unbiased=False).cpu().item()),
                "style_noise_post_std": float(normalized.detach().std(unbiased=False).cpu().item()),
            }
        )
        self.last_i2sb_transport_debug.update(debug)
        return normalized.to(dtype=reference.dtype)

    def _i2sb_fiber_project_gate(self, reference: torch.Tensor) -> torch.Tensor | None:
        if not bool(getattr(self, "i2sb_fiber_project_use_gate", False)):
            return None
        cached = getattr(self, "last_output_style_context", None)
        style_maps = cached.get("style_maps") if isinstance(cached, dict) else None
        gate = getattr(style_maps, "gate_16", None)
        if not torch.is_tensor(gate):
            self.last_i2sb_transport_debug.update(
                {
                    "fiber_project_gate_active": 0.0,
                    "fiber_project_gate_mean": 0.0,
                    "fiber_project_gate_rms": 0.0,
                }
            )
            return None
        gate = gate.to(device=reference.device).float()
        gate = torch.sigmoid(gate).clamp(0.0, 1.0)
        if gate.shape[-2:] != reference.shape[-2:]:
            gate = F.interpolate(gate, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        if gate.shape[0] == 1 and reference.shape[0] > 1:
            gate = gate.expand(reference.shape[0], -1, -1, -1)
        if gate.shape[1] != reference.shape[1]:
            gate = gate.mean(dim=1, keepdim=True).expand(
                reference.shape[0],
                reference.shape[1],
                reference.shape[2],
                reference.shape[3],
            )
        gate_stats = gate.detach().float()
        self.last_i2sb_transport_debug.update(
            {
                "fiber_project_gate_active": 1.0,
                "fiber_project_gate_mean": float(gate_stats.mean().cpu().item()),
                "fiber_project_gate_rms": float(gate_stats.square().mean().sqrt().cpu().item()),
            }
        )
        return gate.to(dtype=reference.dtype)

    def _i2sb_project_endpoint_to_fiber(
        self,
        endpoint: torch.Tensor,
        source_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        if not bool(getattr(self, "i2sb_fiber_project_endpoint", False)):
            return endpoint
        if source_latent is None:
            raise ValueError("model.i2sb_fiber_project_endpoint requires source_latent during solver_i2sb inference.")
        source = source_latent.to(device=endpoint.device)
        if source.shape[-2:] != endpoint.shape[-2:]:
            source = F.interpolate(source.float(), size=endpoint.shape[-2:], mode="bilinear", align_corners=False)
        source_base = self._i2sb_lowpass(source)
        endpoint_fiber = self._i2sb_highpass(endpoint)
        projected = (source_base + endpoint_fiber).to(dtype=endpoint.dtype)
        gate = self._i2sb_fiber_project_gate(endpoint)
        if gate is None:
            return projected
        # Preserve the original endpoint in texture/fiber-active regions and
        # apply the hard low-frequency source anchor mainly on structure regions.
        return torch.lerp(projected, endpoint, gate)

    def _i2sb_residual_fiber_direction(
        self,
        endpoint: torch.Tensor | None,
        source_latent: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if endpoint is None or source_latent is None:
            return None
        source = source_latent.to(device=reference.device)
        if source.shape[-2:] != reference.shape[-2:]:
            source = F.interpolate(source.float(), size=reference.shape[-2:], mode="bilinear", align_corners=False)
        residual = endpoint.to(device=reference.device).float() - source.float()
        residual = self._i2sb_highpass(residual)
        rms = residual.detach().float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
        residual = residual / rms
        return residual.to(dtype=reference.dtype)

    def _i2sb_project_noise_to_fiber(
        self,
        noise: torch.Tensor,
        *,
        endpoint: torch.Tensor | None = None,
        source_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not bool(getattr(self, "i2sb_fiber_project_noise", False)):
            return noise
        mode = str(getattr(self, "i2sb_fiber_project_noise_mode", "highpass")).strip().lower()
        if mode not in {"highpass", "residual_envelope", "residual_direction"}:
            mode = "highpass"
        projected = self._i2sb_highpass(noise).to(dtype=noise.dtype)
        residual_direction = self._i2sb_residual_fiber_direction(endpoint, source_latent, noise)
        if mode == "residual_envelope" and residual_direction is not None:
            envelope = residual_direction.detach().float().abs().mean(dim=1, keepdim=True)
            power = float(getattr(self, "i2sb_fiber_project_residual_power", 1.0))
            if abs(power - 1.0) > 1e-6:
                envelope = envelope.clamp_min(1e-6).pow(power)
            envelope_rms = envelope.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
            envelope = envelope / envelope_rms
            projected = projected * envelope.to(device=projected.device, dtype=projected.dtype)
            project_debug = {
                "fiber_project_residual_active": 1.0,
                "fiber_project_residual_mode_envelope": 1.0,
                "fiber_project_residual_mode_direction": 0.0,
                "fiber_project_residual_power": power,
                "fiber_project_residual_rms": 1.0,
            }
            self.last_i2sb_transport_debug.update(project_debug)
            self.last_i2sb_fiber_project_debug = project_debug
        elif mode == "residual_direction" and residual_direction is not None:
            scalar_noise = projected.float().mean(dim=1, keepdim=True)
            scalar_rms = scalar_noise.detach().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
            scalar_noise = scalar_noise / scalar_rms
            projected = residual_direction.to(dtype=projected.dtype) * scalar_noise.to(dtype=projected.dtype)
            project_debug = {
                "fiber_project_residual_active": 1.0,
                "fiber_project_residual_mode_envelope": 0.0,
                "fiber_project_residual_mode_direction": 1.0,
                "fiber_project_residual_power": float(getattr(self, "i2sb_fiber_project_residual_power", 1.0)),
                "fiber_project_residual_rms": 1.0,
            }
            self.last_i2sb_transport_debug.update(project_debug)
            self.last_i2sb_fiber_project_debug = project_debug
        else:
            project_debug = {
                "fiber_project_residual_active": 0.0,
                "fiber_project_residual_mode_envelope": 0.0,
                "fiber_project_residual_mode_direction": 0.0,
                "fiber_project_residual_power": float(getattr(self, "i2sb_fiber_project_residual_power", 1.0)),
                "fiber_project_residual_rms": 0.0,
            }
            self.last_i2sb_transport_debug.update(project_debug)
            self.last_i2sb_fiber_project_debug = project_debug
        gate = self._i2sb_fiber_project_gate(noise)
        if gate is None:
            return projected
        return projected * gate

    @staticmethod
    def _make_style_injector(
        input_dim: int,
        channels: int,
        hidden_dim: int,
        *,
        live_init: bool = False,
        live_init_std: float = 0.02,
    ) -> nn.Module:
        hidden = max(4, int(hidden_dim))
        module = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(channels)),
        )
        last = module[-1]
        if isinstance(last, nn.Linear):
            if live_init and live_init_std > 0.0:
                nn.init.normal_(last.weight, mean=0.0, std=float(live_init_std))
            else:
                nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        return module

    @staticmethod
    def _make_carrier_gate_injector(
        style_dim: int,
        content_dim: int,
        channels: int,
        hidden_dim: int,
        *,
        live_init: bool = False,
        live_init_std: float = 0.02,
    ) -> tuple[nn.Module, nn.Module]:
        hidden = max(4, int(hidden_dim))
        carrier = nn.Sequential(
            nn.LayerNorm(int(style_dim)),
            nn.Linear(int(style_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(channels)),
        )
        gate = nn.Sequential(
            nn.LayerNorm(int(content_dim)),
            nn.Linear(int(content_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(channels)),
        )
        carrier_last = carrier[-1]
        if isinstance(carrier_last, nn.Linear):
            if live_init and live_init_std > 0.0:
                nn.init.normal_(carrier_last.weight, mean=0.0, std=float(live_init_std))
            else:
                nn.init.zeros_(carrier_last.weight)
            nn.init.zeros_(carrier_last.bias)
        gate_last = gate[-1]
        if isinstance(gate_last, nn.Linear):
            # Keep gate identity-like at init so live-init only wakes the style carrier.
            nn.init.zeros_(gate_last.weight)
            nn.init.zeros_(gate_last.bias)
        return carrier, gate

    @staticmethod
    def _make_spatial_carrier_gate_injector(
        style_map_channels: int,
        feat_channels: int,
        content_dim: int,
        source_channels: int,
        hidden_dim: int,
        *,
        live_init: bool = False,
        live_init_std: float = 0.02,
    ) -> tuple[nn.Module, nn.Module, nn.Module]:
        hidden = max(4, int(hidden_dim))
        structure_hidden = max(4, hidden // 4)
        style_proj = nn.Sequential(
            nn.Conv2d(int(style_map_channels), int(feat_channels), kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(int(feat_channels), int(feat_channels), kernel_size=1, stride=1, padding=0),
        )
        content_gate = nn.Sequential(
            nn.LayerNorm(int(content_dim)),
            nn.Linear(int(content_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(feat_channels)),
        )
        structure_gate = nn.Sequential(
            nn.Conv2d(int(source_channels), structure_hidden, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(structure_hidden, int(feat_channels), kernel_size=3, stride=1, padding=1),
        )
        style_last = style_proj[-1]
        if isinstance(style_last, nn.Conv2d):
            if live_init and live_init_std > 0.0:
                nn.init.normal_(style_last.weight, mean=0.0, std=float(live_init_std))
            else:
                nn.init.zeros_(style_last.weight)
            if style_last.bias is not None:
                nn.init.zeros_(style_last.bias)
        for module in (content_gate, structure_gate):
            last = module[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            elif isinstance(last, nn.Conv2d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)
        return style_proj, content_gate, structure_gate

    def _content_budget_features(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        mean = xf.mean(dim=(2, 3))
        std = xf.std(dim=(2, 3), unbiased=False)
        abs_mean = xf.abs().mean(dim=(2, 3))
        low = F.avg_pool2d(xf, kernel_size=3, stride=1, padding=1)
        high_abs = (xf - low).abs().mean(dim=(2, 3))
        energy = xf.flatten(1).square().mean(dim=1, keepdim=True).sqrt()
        feat = torch.cat([mean, std, abs_mean, high_abs, energy], dim=1)
        return feat.to(device=x.device, dtype=x.dtype)

    def _apply_execution_budget(self, delta: torch.Tensor, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        if self.execution_budget_mode == "none" or self.execution_budget_head is None or self.execution_budget_log_span <= 0.0:
            return delta
        content_feat = self._content_budget_features(x)
        budget_in = torch.cat([style_code, content_feat], dim=1)
        logits = self.execution_budget_head(budget_in)
        gains = torch.exp(torch.tanh(logits.float()) * self.execution_budget_log_span).to(dtype=delta.dtype)
        if self.execution_budget_mode == "scalar":
            return delta * gains.view(-1, 1, 1, 1)
        low = F.avg_pool2d(delta.float(), kernel_size=3, stride=1, padding=1).to(dtype=delta.dtype)
        high = delta - low
        low_gain = gains[:, 0].view(-1, 1, 1, 1)
        high_gain = gains[:, 1].view(-1, 1, 1, 1)
        return low * low_gain + high * high_gain

    def _style_injection_highpass(self, x: torch.Tensor) -> torch.Tensor:
        if not self.style_injection_force_highpass:
            return x
        pad = self.style_injection_spatial_kernel // 2
        low = F.avg_pool2d(x.float(), kernel_size=self.style_injection_spatial_kernel, stride=1, padding=pad)
        return x - low.to(dtype=x.dtype)

    def _style_delta_highpass(self, x: torch.Tensor) -> torch.Tensor:
        if not self.style_delta_force_highpass:
            return x
        pad = self.style_delta_highpass_kernel // 2
        low = F.avg_pool2d(x.float(), kernel_size=self.style_delta_highpass_kernel, stride=1, padding=pad)
        return x - low.to(dtype=x.dtype)

    def _style_section_highpass(self, x: torch.Tensor) -> torch.Tensor:
        if not self.style_section_force_highpass:
            return x
        pad = self.style_delta_highpass_kernel // 2
        low = F.avg_pool2d(x.float(), kernel_size=self.style_delta_highpass_kernel, stride=1, padding=pad)
        return x - low.to(dtype=x.dtype)

    def _style_head_adapter_highpass(self, x: torch.Tensor) -> torch.Tensor:
        if not self.style_head_adapter_force_highpass:
            return x
        pad = self.style_delta_highpass_kernel // 2
        low = F.avg_pool2d(x.float(), kernel_size=self.style_delta_highpass_kernel, stride=1, padding=pad)
        return x - low.to(dtype=x.dtype)

    def _apply_predec_style_section(
        self,
        h: torch.Tensor,
        style_code: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self.style_delta_mode != "predec_section"
            or self.style_section_scale <= 0.0
            or self.style_section_basis_proj is None
            or self.style_section_weight_head is None
            or self.style_section_out is None
            or style_code is None
        ):
            if self.style_delta_mode == "predec_section":
                self.last_style_delta_debug = {"style_predec_section_active": 0.0}
            return h
        bsz, ch, h_dim, w_dim = h.shape
        rank = int(self.style_delta_rank)
        basis = self.style_section_basis_proj(h.float()).view(bsz, rank, ch, h_dim, w_dim)
        weights = torch.tanh(self.style_section_weight_head(style_code).float())
        section = torch.einsum("br,brchw->bchw", weights, basis)
        section = self._style_section_highpass(section.to(dtype=h.dtype))
        section = self.style_section_out(section.float()).to(dtype=h.dtype)
        section = torch.tanh(section.float()).to(dtype=h.dtype) * float(self.style_section_scale)
        with torch.no_grad():
            base_rms = h.detach().float().square().mean().sqrt().clamp_min(1e-8)
            section_rms = section.detach().float().square().mean().sqrt()
            self.last_style_delta_debug = {
                "style_predec_section_active": 1.0,
                "style_predec_section_rank": float(rank),
                "style_predec_section_basis_abs": float(basis.detach().float().abs().mean().cpu().item()),
                "style_predec_section_weight_abs": float(weights.detach().float().abs().mean().cpu().item()),
                "style_predec_section_abs": float(section.detach().float().abs().mean().cpu().item()),
                "style_predec_section_rms": float(section_rms.cpu().item()),
                "style_predec_section_rel_rms": float((section_rms / base_rms).cpu().item()),
                "style_predec_section_scale": float(self.style_section_scale),
            }
        return h + section

    def _apply_style_head_adapter(
        self,
        delta: torch.Tensor,
        h: torch.Tensor,
        style_code: torch.Tensor | None,
        style_maps: StyleMaps | None = None,
    ) -> torch.Tensor:
        if (
            self.style_delta_mode != "head_adapter"
            or self.style_head_adapter_scale <= 0.0
            or self.style_head_adapter_in is None
            or self.style_head_adapter_film is None
            or self.style_head_adapter_out is None
            or style_code is None
        ):
            if self.style_delta_mode == "head_adapter":
                self.last_style_delta_debug = {"style_head_adapter_active": 0.0}
            return delta
        hidden = self.style_head_adapter_in(h.float())
        gamma_beta = self.style_head_adapter_film(style_code).float()
        gamma, beta = gamma_beta.chunk(2, dim=1)
        hidden = hidden * (1.0 + torch.tanh(gamma).view(gamma.shape[0], -1, 1, 1))
        hidden = hidden + torch.tanh(beta).view(beta.shape[0], -1, 1, 1)
        hidden = F.silu(hidden)
        side = self.style_head_adapter_out(hidden).to(dtype=delta.dtype)
        side = self._style_head_adapter_highpass(side)
        gate_active = 0.0
        gate_mean = 0.0
        if bool(getattr(self, "style_head_adapter_use_gate", False)) and isinstance(style_maps, StyleMaps):
            gate = getattr(style_maps, "gate_16", None)
            if torch.is_tensor(gate):
                gate = gate.to(device=side.device).float()
                gate = torch.sigmoid(gate).clamp(0.0, 1.0)
                if gate.shape[-2:] != side.shape[-2:]:
                    gate = F.interpolate(gate, size=side.shape[-2:], mode="bilinear", align_corners=False)
                if gate.shape[0] == 1 and side.shape[0] > 1:
                    gate = gate.expand(side.shape[0], -1, -1, -1)
                if gate.shape[1] != side.shape[1]:
                    gate = gate.mean(dim=1, keepdim=True).expand(side.shape[0], side.shape[1], side.shape[2], side.shape[3])
                power = float(getattr(self, "style_head_adapter_gate_power", 1.0))
                if abs(power - 1.0) > 1e-6:
                    gate = gate.clamp_min(1e-6).pow(power)
                gate_mean = float(gate.detach().float().mean().cpu().item())
                gate_active = 1.0
                side = side * gate.to(dtype=side.dtype)
        side = torch.tanh(side.float()).to(dtype=delta.dtype) * float(self.style_head_adapter_scale)
        with torch.no_grad():
            delta_rms = delta.detach().float().square().mean().sqrt().clamp_min(1e-8)
            side_rms = side.detach().float().square().mean().sqrt()
            self.last_style_delta_debug = {
                "style_head_adapter_active": 1.0,
                "style_head_adapter_gate_active": gate_active,
                "style_head_adapter_gate_mean": gate_mean,
                "style_head_adapter_gate_power": float(getattr(self, "style_head_adapter_gate_power", 1.0)),
                "style_head_adapter_abs": float(side.detach().float().abs().mean().cpu().item()),
                "style_head_adapter_rms": float(side_rms.cpu().item()),
                "style_head_adapter_rel_rms": float((side_rms / delta_rms).cpu().item()),
                "style_head_adapter_gamma_abs": float(gamma.detach().float().abs().mean().cpu().item()),
                "style_head_adapter_beta_abs": float(beta.detach().float().abs().mean().cpu().item()),
                "style_head_adapter_scale": float(self.style_head_adapter_scale),
            }
        return delta + side

    def _apply_style_delta_basis(
        self,
        delta: torch.Tensor,
        h: torch.Tensor,
        style_code: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self.style_delta_mode != "basis"
            or self.style_delta_scale <= 0.0
            or self.style_delta_basis_proj is None
            or self.style_delta_weight_head is None
            or style_code is None
        ):
            if self.style_delta_mode != "predec_section":
                self.last_style_delta_debug = {"style_delta_basis_active": 0.0}
            return delta
        bsz, _, h_dim, w_dim = delta.shape
        rank = int(self.style_delta_rank)
        basis = self.style_delta_basis_proj(h.float()).view(
            bsz,
            rank,
            int(self.latent_channels),
            h_dim,
            w_dim,
        )
        weights = torch.tanh(self.style_delta_weight_head(style_code).float())
        side = torch.einsum("br,brchw->bchw", weights, basis)
        side = self._style_delta_highpass(side.to(dtype=delta.dtype))
        side = torch.tanh(side.float()).to(dtype=delta.dtype) * float(self.style_delta_scale)
        with torch.no_grad():
            self.last_style_delta_debug = {
                "style_delta_basis_active": 1.0,
                "style_delta_basis_rank": float(rank),
                "style_delta_basis_abs": float(basis.detach().float().abs().mean().cpu().item()),
                "style_delta_weight_abs": float(weights.detach().float().abs().mean().cpu().item()),
                "style_delta_side_abs": float(side.detach().float().abs().mean().cpu().item()),
                "style_delta_side_rms": float(side.detach().float().square().mean().sqrt().cpu().item()),
                "style_delta_scale": float(self.style_delta_scale),
            }
        return delta + side

    def _apply_style_feature_injection(
        self,
        feat: torch.Tensor,
        x: torch.Tensor,
        style_code: torch.Tensor,
        *,
        site: str,
        style_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.style_injection_mode == "none" or self.style_injection_scale <= 0.0:
            return feat
        content_feat = self._content_budget_features(x)
        if self.style_injection_form == "carrier_gate":
            carrier = self.body_style_carrier if site == "body" else self.decoder_style_carrier
            gate_head = self.body_content_gate if site == "body" else self.decoder_content_gate
            if carrier is None or gate_head is None:
                return feat
            carrier_bias = torch.tanh(carrier(style_code).float())
            gate_logits = gate_head(content_feat).float()
            gate = torch.exp(torch.tanh(gate_logits) * self.style_injection_gate_log_span)
            bias = (carrier_bias * gate).to(dtype=feat.dtype)
            return feat + bias.view(feat.shape[0], feat.shape[1], 1, 1) * self.style_injection_scale
        if self.style_injection_form == "mixed":
            injector = self.body_style_injector if site == "body" else self.decoder_style_injector
            if injector is None:
                return feat
            inject_in = torch.cat([style_code, content_feat], dim=1)
            bias = torch.tanh(injector(inject_in).float()).to(dtype=feat.dtype)
            return feat + bias.view(feat.shape[0], feat.shape[1], 1, 1) * self.style_injection_scale
        if self.style_injection_form == "spatial_carrier_gate":
            spatial_proj = self.body_style_spatial_proj if site == "body" else self.decoder_style_spatial_proj
            content_gate = self.body_content_gate if site == "body" else self.decoder_content_gate
            structure_gate = self.body_structure_gate if site == "body" else self.decoder_structure_gate
            if spatial_proj is None or content_gate is None or structure_gate is None or style_map is None:
                return feat
            if style_map.shape[-2:] != feat.shape[-2:]:
                style_map = F.interpolate(style_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            if style_map.device != feat.device:
                style_map = style_map.to(device=feat.device)
            if style_map.dtype != feat.dtype:
                style_map = style_map.to(dtype=feat.dtype)
            style_field = torch.tanh(spatial_proj(style_map.float())).to(dtype=feat.dtype)
            style_field = self._style_injection_highpass(style_field)
            channel_gate = torch.exp(
                torch.tanh(content_gate(content_feat).float()) * self.style_injection_gate_log_span
            ).to(dtype=feat.dtype).view(feat.shape[0], feat.shape[1], 1, 1)
            src_local = x
            if src_local.shape[-2:] != feat.shape[-2:]:
                src_local = F.interpolate(src_local.float(), size=feat.shape[-2:], mode="bilinear", align_corners=False).to(dtype=feat.dtype)
            local_gate = torch.sigmoid(structure_gate(src_local.float())).to(dtype=feat.dtype)
            return feat + style_field * local_gate * channel_gate * self.style_injection_scale
        return feat

    def _compute_delta(
        self,
        h: torch.Tensor,
        x: torch.Tensor | None = None,
        style_code: torch.Tensor | None = None,
        style_maps: StyleMaps | None = None,
    ) -> torch.Tensor:
        h = self._apply_predec_style_section(h, style_code)
        raw_delta = self.dec_out(h)
        if bool(getattr(self, "use_diffeomorphic_stroke", False)):
            if x is None:
                raise ValueError("diffeomorphic stroke mode requires input x.")
            t_profile = self._profile_start(x)
            stroked = apply_texture_aligned_diffeomorphic_stroke(
                x,
                raw_delta,
                color_strength=float(getattr(self, "diffeomorphic_color_strength", 0.85)),
                warp_strength=float(getattr(self, "diffeomorphic_warp_strength", 0.08)),
                gate_strength=float(getattr(self, "diffeomorphic_texture_gate_strength", 8.0)),
                normal_leak=float(getattr(self, "diffeomorphic_normal_leak", 0.0)),
            )
            self._profile_end("diffeomorphic_stroke", t_profile, x)
            if self.transport_prediction_mode == "endpoint":
                if self.endpoint_parameterization == "residual":
                    delta = stroked - x.float()
                    delta = self._apply_style_delta_basis(delta, h, style_code)
                    return self._apply_style_head_adapter(delta, h, style_code, style_maps)
                endpoint = self._apply_style_delta_basis(stroked.float(), h, style_code)
                return self._apply_style_head_adapter(endpoint, h, style_code, style_maps)
            delta = stroked - x.float()
            delta = self._apply_style_delta_basis(delta, h, style_code)
            return self._apply_style_head_adapter(delta, h, style_code, style_maps)
        if self.transport_prediction_mode == "endpoint":
            bounded = torch.tanh(raw_delta / self.transport_endpoint_scale) * self.transport_endpoint_scale
            if self.endpoint_parameterization == "residual":
                bounded = self._apply_style_delta_basis(bounded, h, style_code)
                return self._apply_style_head_adapter(bounded, h, style_code, style_maps)
            bounded = self._apply_style_delta_basis(bounded, h, style_code)
            return self._apply_style_head_adapter(bounded, h, style_code, style_maps)
        if self.velocity_head_mode == "tanh":
            raw_delta = torch.tanh(raw_delta) * self.velocity_tanh_limit
        delta = raw_delta * self.latent_scale_factor * self.residual_gain
        delta = self._apply_style_delta_basis(delta, h, style_code)
        return self._apply_style_head_adapter(delta, h, style_code, style_maps)

    def _endpoint_delta_from_raw(self, raw_transport: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raw = raw_transport.to(dtype=x.dtype)
        if self.endpoint_parameterization == "residual":
            return raw
        absolute_delta = raw - x.to(dtype=raw.dtype)
        if self.endpoint_parameterization == "blend":
            blend = float(getattr(self, "endpoint_residual_blend", 0.0))
            if blend <= 0.0:
                return absolute_delta
            if blend >= 1.0:
                return raw
            return torch.lerp(absolute_delta, raw, blend)
        if self.endpoint_parameterization == "orthogonal_lowhigh":
            kernel = max(1, int(getattr(self, "endpoint_orthogonal_kernel", 5)))
            if kernel <= 1:
                return absolute_delta
            pad = kernel // 2
            raw_f = raw.float()
            x_f = x.float()
            raw_low = F.avg_pool2d(raw_f, kernel_size=kernel, stride=1, padding=pad)
            x_low = F.avg_pool2d(x_f, kernel_size=kernel, stride=1, padding=pad)
            low_anchor = float(getattr(self, "endpoint_orthogonal_low_anchor", 1.0))
            low_mode = str(getattr(self, "endpoint_orthogonal_low_mode", "all")).strip().lower()
            if low_mode == "channel_mean":
                raw_low_mean = raw_low.mean(dim=1, keepdim=True)
                x_low_mean = x_low.mean(dim=1, keepdim=True)
                # Anchor the shared low-frequency structure while retaining
                # channel-relative low-frequency style/color components.
                anchored_mean = torch.lerp(raw_low_mean, x_low_mean, low_anchor)
                anchored_low = anchored_mean + (raw_low - raw_low_mean)
            else:
                anchored_low = torch.lerp(raw_low, x_low, low_anchor)
            raw_high = (raw_f - raw_low) * float(getattr(self, "endpoint_orthogonal_high_scale", 1.0))
            endpoint = anchored_low + raw_high
            return (endpoint - x_f).to(dtype=raw.dtype)
        return absolute_delta

    def _resolve_t_input(self, x: torch.Tensor, t: torch.Tensor | float | None) -> torch.Tensor:
        if t is None:
            t = 1.0
        if not torch.is_tensor(t):
            return torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            return t.to(device=x.device, dtype=x.dtype).expand(x.shape[0])
        t = t.to(device=x.device, dtype=x.dtype).view(-1)
        if t.shape[0] == 1 and x.shape[0] > 1:
            return t.expand(x.shape[0])
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"time batch mismatch: expected {x.shape[0]} or 1, got {t.shape[0]}")
        return t

    def _compute_style_code(
        self,
        *,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        t: torch.Tensor,
        style_code_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style_code_override is None:
            if str(getattr(self, "tokenizer_family", "legacy_factorized")) in {"pure_latent_spatial", "smoe_translator", "affine_connection_tokenizer"}:
                style_code = x.new_zeros((x.shape[0], int(self.bridge_style_dim)))
            else:
                style_code = self.encode_style_id(style_id, t=t)
        else:
            style_code = style_code_override
            if style_code.ndim == 1:
                style_code = style_code.unsqueeze(0)
            style_code = style_code.to(device=x.device, dtype=x.dtype)
        if style_code.shape[0] == 1 and x.shape[0] > 1:
            style_code = style_code.expand(x.shape[0], -1)
        elif style_code.shape[0] != x.shape[0]:
            raise ValueError(f"style code batch mismatch: expected {x.shape[0]} or 1, got {style_code.shape[0]}")

        time_code = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim).to(dtype=style_code.dtype))
        return style_code + time_code

    def _resolve_integration_horizon(self, *, step_size: float, style_strength: float | None) -> float:
        strength = self._resolve_style_strength(style_strength)
        horizon = max(0.0, float(step_size)) * strength
        max_horizon = max(1e-6, float(getattr(self, "style_strength_max", 1.0)))
        if not self.allow_style_overdrive:
            max_horizon = min(max_horizon, 1.0)
        resolved = max(0.0, min(max_horizon, horizon))
        self.last_style_strength_debug.update(
            {
                "style_step_scale": float(strength),
                "integration_horizon": float(resolved),
                "integration_horizon_requested": float(horizon),
                "style_overdrive_allowed": float(self.allow_style_overdrive),
                "style_overdrive_clamped": float(horizon > resolved and horizon > 1.0),
            }
        )
        return resolved

    def _runtime_content_dino_gate(self, ref: torch.Tensor) -> torch.Tensor | None:
        payload = self.runtime_conditioning if isinstance(self.runtime_conditioning, dict) else {}
        patches = payload.get("content_dino_patches")
        if not torch.is_tensor(patches):
            return None
        patches = patches.to(device=ref.device, dtype=torch.float32)
        score = patches.std(dim=-1, unbiased=False, keepdim=True)
        hw = payload.get("content_dino_hw")
        if torch.is_tensor(hw) and hw.numel() >= 2:
            h_dim = max(1, int(hw.view(-1)[0].item()))
            w_dim = max(1, int(hw.view(-1)[1].item()))
        else:
            side = int(round(max(1, patches.shape[1]) ** 0.5))
            h_dim = side
            w_dim = max(1, patches.shape[1] // max(side, 1))
        if h_dim * w_dim != patches.shape[1]:
            h_dim, w_dim = 1, int(patches.shape[1])
        gate = score.transpose(1, 2).contiguous().view(patches.shape[0], 1, h_dim, w_dim)
        gate = F.interpolate(gate, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        gate = gate - gate.amin(dim=(2, 3), keepdim=True)
        gate = gate / gate.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return gate.to(dtype=ref.dtype)

    def _project_velocity_tangent(self, velocity: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        gate = self._runtime_content_dino_gate(ref)
        if gate is None:
            return velocity
        strength = max(0.0, float(getattr(self, "solver_tangent_projection_strength", 1.0)))
        return velocity * (1.0 - gate * strength)

    def _transport_velocity(
        self,
        h: torch.Tensor,
        *,
        t: float,
        style_id: torch.Tensor | int | None,
        style_code_override: torch.Tensor | None = None,
        target_style_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        velocity = self.forward(
            h,
            t=t,
            style_id=style_id,
            style_code_override=style_code_override,
            target_style_latent=target_style_latent,
        )
        if self.solver_family == "solver_tangent_rk":
            return self._project_velocity_tangent(velocity, h)
        return velocity

    def _i2sb_transport_step(
        self,
        h: torch.Tensor,
        *,
        t_curr: float,
        t_next: float,
        style_id: torch.Tensor | int | None,
        style_code_override: torch.Tensor | None = None,
        target_style_latent: torch.Tensor | None = None,
        source_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        predict_t = float(t_curr)
        if self.i2sb_predictor_time_floor > 0.0 and predict_t < self.i2sb_predictor_time_floor:
            predict_t = min(max(predict_t, self.i2sb_predictor_time_floor), float(t_next))
        predict_t = min(max(predict_t, 0.0), 1.0 - 1e-6)
        previous_fiber_noise_debug = getattr(self, "last_i2sb_fiber_noise_debug", {}) or {}
        previous_fiber_project_debug = getattr(self, "last_i2sb_fiber_project_debug", {}) or {}
        self.last_i2sb_transport_debug = {
            "t_curr": float(t_curr),
            "predict_t": float(predict_t),
            "t_next": float(t_next),
            "time_floor": float(self.i2sb_predictor_time_floor),
            "time_floor_active": float(abs(predict_t - float(t_curr)) > 1e-9),
            "endpoint_orthogonal_active": float(self.endpoint_parameterization == "orthogonal_lowhigh"),
            "endpoint_orthogonal_kernel": float(getattr(self, "endpoint_orthogonal_kernel", 1)),
            "endpoint_orthogonal_high_scale": float(getattr(self, "endpoint_orthogonal_high_scale", 1.0)),
            "endpoint_orthogonal_low_anchor": float(getattr(self, "endpoint_orthogonal_low_anchor", 1.0)),
            "endpoint_orthogonal_low_mode_channel_mean": float(
                str(getattr(self, "endpoint_orthogonal_low_mode", "all")).strip().lower() == "channel_mean"
            ),
            "fiber_noise_requested": float(bool(getattr(self, "i2sb_fiber_aligned_noise", False))),
            "fiber_project_endpoint_active": float(bool(getattr(self, "i2sb_fiber_project_endpoint", False))),
            "fiber_project_noise_active": float(bool(getattr(self, "i2sb_fiber_project_noise", False))),
            "fiber_project_kernel": float(getattr(self, "i2sb_fiber_project_kernel", 1)),
            "fiber_project_use_gate": float(bool(getattr(self, "i2sb_fiber_project_use_gate", False))),
            "fiber_project_noise_mode_residual_envelope": float(
                str(getattr(self, "i2sb_fiber_project_noise_mode", "highpass")).strip().lower() == "residual_envelope"
            ),
            "fiber_project_noise_mode_residual_direction": float(
                str(getattr(self, "i2sb_fiber_project_noise_mode", "highpass")).strip().lower() == "residual_direction"
            ),
            "fiber_project_gate_active": 0.0,
            "fiber_project_gate_mean": 0.0,
            "fiber_project_gate_rms": 0.0,
            "fiber_project_residual_active": float(previous_fiber_project_debug.get("fiber_project_residual_active", 0.0)),
            "fiber_project_residual_mode_envelope": float(previous_fiber_project_debug.get("fiber_project_residual_mode_envelope", 0.0)),
            "fiber_project_residual_mode_direction": float(previous_fiber_project_debug.get("fiber_project_residual_mode_direction", 0.0)),
            "fiber_project_residual_power": float(
                previous_fiber_project_debug.get(
                    "fiber_project_residual_power",
                    float(getattr(self, "i2sb_fiber_project_residual_power", 1.0)),
                )
            ),
            "fiber_project_residual_rms": float(previous_fiber_project_debug.get("fiber_project_residual_rms", 0.0)),
            "fiber_noise_active": float(previous_fiber_noise_debug.get("fiber_noise_active", 0.0)),
            "fiber_gate_mean": float(previous_fiber_noise_debug.get("fiber_gate_mean", 0.0)),
            "fiber_gate_rms": float(previous_fiber_noise_debug.get("fiber_gate_rms", 0.0)),
            "fiber_noise_scale": float(previous_fiber_noise_debug.get("fiber_noise_scale", 0.0)),
            "fiber_noise_rms_normalize": float(
                previous_fiber_noise_debug.get(
                    "fiber_noise_rms_normalize",
                    float(bool(getattr(self, "i2sb_fiber_noise_rms_normalize", True))),
                )
            ),
            "style_noise_family_style_covariant": float(self.i2sb_noise_family == "style_covariant"),
            "style_noise_family_gaussian": float(self.i2sb_noise_family == "gaussian"),
            "style_noise_bank_active": 0.0,
            "style_noise_amp_mean": 0.0,
            "style_noise_amp_std": 0.0,
            "style_noise_post_std": 0.0,
            "style_noise_amplitude_power": float(self.i2sb_style_noise_amplitude_power),
            "style_noise_fallback_gaussian": 0.0,
        }
        x_1_pred = self.predict_transport_base(
            h,
            t=predict_t,
            style_id=style_id,
            style_code_override=style_code_override,
            target_style_latent=target_style_latent,
        )
        x_1_pred = self._i2sb_project_endpoint_to_fiber(x_1_pred, source_latent)
        denom = max(1.0 - float(t_curr), 1e-6)
        c_curr = (1.0 - float(t_next)) / denom
        c_target = (float(t_next) - float(t_curr)) / denom
        mu = c_curr * h + c_target * x_1_pred
        if float(t_next) >= 1.0 - 1e-4 or self.bridge_sigma <= 0.0:
            return mu
        var = (self.bridge_sigma ** 2) * (float(t_next) - float(t_curr)) * (1.0 - float(t_next)) / denom
        if var <= 0.0:
            return mu
        noise = self._i2sb_project_noise_to_fiber(
            self._i2sb_sample_transport_noise(
                h,
                target_style_latent=target_style_latent,
            ),
            endpoint=x_1_pred,
            source_latent=source_latent,
        )
        noise = self._i2sb_fiber_aligned_noise(
            h,
            noise,
            noise_scale=math.sqrt(var),
        )
        return mu + math.sqrt(var) * noise

    def _rk_transport_step(
        self,
        h: torch.Tensor,
        *,
        t: float,
        dt: float,
        style_id: torch.Tensor | int | None,
        style_code_override: torch.Tensor | None = None,
        target_style_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        order = max(2, int(getattr(self, "solver_rk_order", 4)))
        if order <= 2:
            k1 = self._transport_velocity(
                h,
                t=t,
                style_id=style_id,
                style_code_override=style_code_override,
                target_style_latent=target_style_latent,
            )
            k2 = self._transport_velocity(
                h + 0.5 * dt * k1,
                t=t + 0.5 * dt,
                style_id=style_id,
                style_code_override=style_code_override,
                target_style_latent=target_style_latent,
            )
            return h + dt * k2
        k1 = self._transport_velocity(
            h,
            t=t,
            style_id=style_id,
            style_code_override=style_code_override,
            target_style_latent=target_style_latent,
        )
        k2 = self._transport_velocity(
            h + 0.5 * dt * k1,
            t=t + 0.5 * dt,
            style_id=style_id,
            style_code_override=style_code_override,
            target_style_latent=target_style_latent,
        )
        k3 = self._transport_velocity(
            h + 0.5 * dt * k2,
            t=t + 0.5 * dt,
            style_id=style_id,
            style_code_override=style_code_override,
            target_style_latent=target_style_latent,
        )
        k4 = self._transport_velocity(
            h + dt * k3,
            t=t + dt,
            style_id=style_id,
            style_code_override=style_code_override,
            target_style_latent=target_style_latent,
        )
        return h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @staticmethod
    def _normalize_solver_corrector_mode(mode: str) -> str:
        normalized = str(mode).strip().lower()
        aliases = {
            "": "none",
            "off": "none",
            "disabled": "none",
            "lowpass": "lowpass_source_anchor",
            "lowpass_anchor": "lowpass_source_anchor",
            "source_lowpass": "lowpass_source_anchor",
            "latent_lowpass": "lowpass_source_anchor",
        }
        normalized = aliases.get(normalized, normalized)
        valid = {"none", "legacy_dino_lerp", "lowpass_source_anchor"}
        if normalized not in valid:
            raise ValueError(
                f"Unsupported model.solver_corrector_mode={mode!r}; "
                "expected one of 'none', 'legacy_dino_lerp', or 'lowpass_source_anchor'."
            )
        return normalized

    def _correct_transport_state(self, h: torch.Tensor, source: torch.Tensor, *, dt: float) -> torch.Tensor:
        steps = max(1, int(getattr(self, "solver_corrector_steps", 2)))
        step_size = max(0.0, float(getattr(self, "solver_corrector_step_size", 0.08)))
        refine_mode = self._normalize_solver_corrector_mode(getattr(self, "solver_corrector_mode", "none"))
        if refine_mode == "none":
            return h
        if refine_mode == "legacy_dino_lerp":
            gate = self._runtime_content_dino_gate(h)
            if gate is None:
                gate = torch.ones((h.shape[0], 1, h.shape[2], h.shape[3]), device=h.device, dtype=h.dtype)
            out = h
            for _ in range(steps):
                out = torch.lerp(out, source, gate * step_size * dt)
            return out

        # Explicit diagnostic corrector: anchor low-frequency structure only.
        # Keep it opt-in because low frequency is not a reliable content proxy for all styles.
        kernel = max(3, int(getattr(self, "solver_corrector_lowpass_kernel", 5)))
        if kernel % 2 == 0:
            kernel += 1
        out = h
        pad = kernel // 2
        source_float = source.float()
        for _ in range(steps):
            out_low = F.avg_pool2d(out.float(), kernel_size=kernel, stride=1, padding=pad)
            src_low = F.avg_pool2d(source_float, kernel_size=kernel, stride=1, padding=pad)

            # Gradient: push low-freq toward source, leave high-freq intact
            correction = out_low - src_low
            out = out - step_size * dt * correction.to(dtype=out.dtype)

            # Optional: clamp correction magnitude to avoid over-correction
            clamp = float(getattr(self, "solver_corrector_clamp", 0.0))
            if clamp > 0:
                out = torch.clamp(out, source_float.to(dtype=out.dtype) - clamp, source_float.to(dtype=out.dtype) + clamp)

        return out

    def _proximal_lowpass(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.proximal_highpass_kernel // 2
        return F.avg_pool2d(x.float(), kernel_size=self.proximal_highpass_kernel, stride=1, padding=pad).to(dtype=x.dtype)

    def _apply_proximal_highpass(self, delta: torch.Tensor) -> torch.Tensor:
        if not self.proximal_force_highpass:
            return delta
        return delta - self._proximal_lowpass(delta)

    def _resolve_refine_style_code(
        self,
        z_base: torch.Tensor,
        *,
        style_id: torch.Tensor | int | None,
        style_code_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_fixed = torch.full((z_base.shape[0],), 1.0, device=z_base.device, dtype=z_base.dtype)
        return self._compute_style_code(
            x=z_base,
            style_id=style_id,
            t=t_fixed,
            style_code_override=style_code_override,
        )

    def _resolve_output_appearance_context(
        self,
        z_base: torch.Tensor,
        *,
        style_id: torch.Tensor | int | None,
        style_code_override: torch.Tensor | None = None,
        source_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, StyleMaps]:
        if self.output_appearance_alignment_mode == "none" or self.output_appearance_head is None:
            return None, StyleMaps(family=str(getattr(self, "tokenizer_family", "legacy_factorized")))
        if self._cached_output_style_context_matches(source_latent):
            cached = getattr(self, "last_output_style_context", None)
            if isinstance(cached, dict):
                cached_code = cached.get("style_code")
                cached_maps = cached.get("style_maps")
                if torch.is_tensor(cached_code) and isinstance(cached_maps, StyleMaps):
                    return cached_code, cached_maps

        style_code = self._resolve_refine_style_code(
            z_base,
            style_id=style_id,
            style_code_override=style_code_override,
        )
        if style_id is None:
            return style_code, StyleMaps(family=str(getattr(self, "tokenizer_family", "legacy_factorized")))

        content_latent = source_latent if source_latent is not None else z_base
        if content_latent.device != z_base.device:
            content_latent = content_latent.to(device=z_base.device)
        if content_latent.dtype != z_base.dtype:
            content_latent = content_latent.to(dtype=z_base.dtype)
        feat_c = content_latent / max(self.latent_scale_factor, 1e-8)
        h_c = self.enc_in_act(self.enc_in(feat_c))
        for block in self.hires_body:
            h_c = block(h_c, style_code, gate=0.0)
        content_feat_16 = self.down(h_c)
        style_code = self._adapt_style_code_from_content(
            style_id=style_id,
            style_code=style_code,
            content_feat_16=content_feat_16,
            style_code_override_active=style_code_override is not None,
        )
        style_code_map = self._decode_style_code_spatial_map(
            style_code,
            target_hw=tuple(int(v) for v in content_feat_16.shape[-2:]),
            device=content_feat_16.device,
            dtype=content_feat_16.dtype,
        )
        structured_ctx = self._structured_style_from_sidecar(
            style_id=style_id,
            style_code=style_code,
            content_latent=content_latent,
            content_feat_16=content_feat_16,
        )
        if structured_ctx is not None:
            structured_code, structured_maps = structured_ctx
            style_code_map = self._decode_style_code_spatial_map(
                structured_code,
                target_hw=tuple(int(v) for v in content_feat_16.shape[-2:]),
                device=content_feat_16.device,
                dtype=content_feat_16.dtype,
            )
            map_16 = self._prepare_spatial_map(structured_maps.map_16, content_feat_16)
            if map_16 is not None and style_code_map is not None:
                map_16 = map_16 + style_code_map
            elif map_16 is None and style_code_map is not None:
                map_16 = style_code_map
            return structured_code, StyleMaps(
                map_16=map_16,
                gate_16=structured_maps.gate_16,
                mask_16=structured_maps.mask_16,
                aux_16=structured_maps.aux_16,
                family=str(getattr(structured_maps, "family", getattr(self, "tokenizer_family", "legacy_factorized"))),
                debug=dict(getattr(structured_maps, "debug", {}) or {}),
            )
        style_maps = self._prepare_style_maps(style_id)
        if style_code_map is not None:
            style_maps = StyleMaps(
                map_16=style_code_map,
                gate_16=style_maps.gate_16,
                mask_16=style_maps.mask_16,
                aux_16=style_maps.aux_16,
                family=str(getattr(style_maps, "family", getattr(self, "tokenizer_family", "legacy_factorized"))),
                debug=dict(getattr(style_maps, "debug", {}) or {}),
            )
        return style_code, style_maps

    def _proximal_internal_style_tokens(
        self,
        z_base: torch.Tensor,
        *,
        style_id: torch.Tensor | int,
        style_code_override: torch.Tensor | None = None,
        source_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content_latent = source_latent if source_latent is not None else z_base
        if content_latent.device != z_base.device:
            content_latent = content_latent.to(device=z_base.device)
        if content_latent.dtype != z_base.dtype:
            content_latent = content_latent.to(dtype=z_base.dtype)
        style_code = self._resolve_refine_style_code(
            z_base,
            style_id=style_id,
            style_code_override=style_code_override,
        )
        feat_c = content_latent / max(self.latent_scale_factor, 1e-8)
        h_c = self.enc_in_act(self.enc_in(feat_c))
        h_style = self._run_style_blocks(
            h_c,
            blocks=self.hires_body,
            style_code=style_code,
            base_idx=0,
            gate_scale=1.0,
        )
        style_map = self.down(h_style)
        code_map = self._decode_style_code_spatial_map(
            style_code,
            target_hw=tuple(int(v) for v in style_map.shape[-2:]),
            device=style_map.device,
            dtype=style_map.dtype,
        )
        if code_map is not None:
            style_map = style_map + code_map
        style_map = F.interpolate(
            style_map.to(device=z_base.device, dtype=z_base.dtype),
            size=z_base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.proximal_style_tokens is not None:
            token_bias = self.proximal_style_tokens(style_code).view(style_code.shape[0], self.body_channels, 1, 1)
            style_map = style_map + token_bias
        return style_code, style_map

    def _structured_proximal_style_tokens(
        self,
        z_base: torch.Tensor,
        *,
        style_id: torch.Tensor | int,
        style_code_override: torch.Tensor | None = None,
        source_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content_latent = source_latent if source_latent is not None else z_base
        if content_latent.device != z_base.device:
            content_latent = content_latent.to(device=z_base.device)
        if content_latent.dtype != z_base.dtype:
            content_latent = content_latent.to(dtype=z_base.dtype)
        style_code = self._resolve_refine_style_code(
            z_base,
            style_id=style_id,
            style_code_override=style_code_override,
        )
        feat_c = content_latent / max(self.latent_scale_factor, 1e-8)
        h_c = self.enc_in_act(self.enc_in(feat_c))
        for block in self.hires_body:
            h_c = block(h_c, style_code, gate=0.0)
        content_feat_16 = self.down(h_c)
        style_code = self._adapt_style_code_from_content(
            style_id=style_id,
            style_code=style_code,
            content_feat_16=content_feat_16,
            style_code_override_active=style_code_override is not None,
        )
        structured_ctx = self._structured_style_from_sidecar(
            style_id=style_id,
            style_code=style_code,
            content_latent=content_latent,
            content_feat_16=content_feat_16,
        )
        if structured_ctx is None:
            raise RuntimeError(
                "latent structured tokenizer + crossattn_texture requires structured tokenizer output for proximal refinement."
            )
        style_code, style_maps = structured_ctx
        style_map = self._prepare_spatial_map(style_maps.map_16, content_feat_16)
        if style_map is None:
            raise RuntimeError("structured tokenizer did not produce a usable spatial map for proximal refinement.")
        code_map = self._decode_style_code_spatial_map(
            style_code,
            target_hw=tuple(int(v) for v in style_map.shape[-2:]),
            device=style_map.device,
            dtype=style_map.dtype,
        )
        if code_map is not None:
            style_map = style_map + code_map
        style_map = F.interpolate(
            style_map.to(device=z_base.device, dtype=z_base.dtype),
            size=z_base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.proximal_style_tokens is not None:
            token_bias = self.proximal_style_tokens(style_code).view(style_code.shape[0], self.body_channels, 1, 1)
            style_map = style_map + token_bias.to(device=style_map.device, dtype=style_map.dtype)
        return style_code, style_map

    def _route_proximal_attention(self, logits: torch.Tensor) -> torch.Tensor:
        if self.proximal_attn_routing_mode == "sinkhorn":
            return _sinkhorn_attention(logits, iters=self.proximal_attn_sinkhorn_iters).to(dtype=logits.dtype)
        if self.proximal_attn_routing_mode == "gumbel_hard":
            return _gumbel_hard_attention(logits, tau=self.proximal_attn_gumbel_tau).to(dtype=logits.dtype)
        return torch.softmax(logits, dim=-1)

    def refine_endpoint(
        self,
        z_base: torch.Tensor,
        *,
        style_id: torch.Tensor | int | None,
        source_latent: torch.Tensor | None = None,
        style_code_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.last_base_endpoint = z_base.detach()
        appearance_style_code, appearance_style_maps = self._resolve_output_appearance_context(
            z_base,
            style_id=style_id,
            style_code_override=style_code_override,
            source_latent=source_latent,
        )
        if self.proximal_mode == "off":
            self.last_proximal_residual = torch.zeros_like(z_base)
            self.last_proximal_clamp_scale = torch.ones((), device=z_base.device, dtype=z_base.dtype)
            z_final = z_base
            if appearance_style_code is not None:
                z_final = self._apply_output_appearance_alignment(
                    z_final,
                    style_code=appearance_style_code,
                    style_maps=appearance_style_maps,
                )
            self.last_final_endpoint = z_final.detach()
            return z_final
        if self.proximal_mode == "crossattn_texture":
            if (
                self.proximal_attn_q is None
                or self.proximal_attn_k is None
                or self.proximal_attn_v is None
                or self.proximal_attn_out is None
            ):
                raise RuntimeError("cross-attention proximal modules not initialized")
            if style_id is None:
                raise ValueError("style_id is required for crossattn_texture proximal mode.")
            if str(getattr(self, "tokenizer_family", "legacy_factorized")) in {"pure_latent_spatial", "smoe_translator", "affine_connection_tokenizer"}:
                style_code, kv_src = self._structured_proximal_style_tokens(
                    z_base,
                    style_id=style_id,
                    style_code_override=style_code_override,
                    source_latent=source_latent,
                )
            else:
                style_code, kv_src = self._proximal_internal_style_tokens(
                    z_base,
                    style_id=style_id,
                    style_code_override=style_code_override,
                    source_latent=source_latent,
                )
            q = self.proximal_attn_q(z_base.float())
            k = self.proximal_attn_k(kv_src)
            v = self.proximal_attn_v(kv_src)
            bsz, ch, h_dim, w_dim = q.shape
            q_flat = q.view(bsz, ch, -1).transpose(1, 2)
            k_flat = k.view(bsz, ch, -1)
            attn_logits = torch.bmm(q_flat, k_flat) / math.sqrt(float(ch))
            attn = self._route_proximal_attention(attn_logits)
            v_flat = v.view(bsz, ch, -1).transpose(1, 2)
            mixed = torch.bmm(attn, v_flat).transpose(1, 2).view(bsz, ch, h_dim, w_dim)
            delta = self.proximal_attn_out(mixed).to(dtype=z_base.dtype)
            delta = self._apply_proximal_highpass(delta)
        else:
            raise RuntimeError(f"retired proximal_mode is not supported in the cleaned runtime: {self.proximal_mode}")
        clamp_scale = torch.ones((), device=z_base.device, dtype=z_base.dtype)
        clamp_ratio = self._resolve_proximal_clamp_ratio()
        if source_latent is not None and clamp_ratio > 0.0:
            base_transport = (z_base - source_latent).float()
            base_rms = base_transport.square().mean().sqrt()
            delta_rms = delta.float().square().mean().sqrt()
            allowed = base_rms * clamp_ratio
            if bool((delta_rms > allowed).item()):
                clamp_scale = (allowed / delta_rms.clamp_min(1e-8)).to(dtype=z_base.dtype)
                delta = delta * clamp_scale
        z_final = z_base + delta
        if appearance_style_code is not None:
            z_final = self._apply_output_appearance_alignment(
                z_final,
                style_code=appearance_style_code,
                style_maps=appearance_style_maps,
            )
        self.last_proximal_residual = delta.detach()
        self.last_proximal_clamp_scale = clamp_scale.detach()
        self.last_final_endpoint = z_final.detach()
        return z_final

    def _resolve_proximal_clamp_ratio(self) -> float:
        start = float(self.proximal_clamp_ratio)
        mid = float(getattr(self, "proximal_clamp_ratio_mid", 0.0))
        end = float(self.proximal_clamp_ratio_end)
        schedule = str(getattr(self, "proximal_clamp_schedule", "linear")).strip().lower()
        hold_epochs = max(0, int(getattr(self, "proximal_clamp_hold_epochs", 0)))
        release_epochs = int(self.proximal_clamp_release_epochs)
        mid_hold_epochs = max(0, int(getattr(self, "proximal_clamp_mid_hold_epochs", 0)))
        second_release_epochs = max(0, int(getattr(self, "proximal_clamp_second_release_epochs", 0)))
        if start <= 0.0:
            return 0.0
        if schedule == "hold_two_stage":
            if mid <= 0.0:
                mid = end if end > 0.0 else start
            epoch_idx = max(0, int(getattr(self, "current_epoch", 1)) - 1)
            if epoch_idx < hold_epochs:
                return start
            epoch_idx = max(0, epoch_idx - hold_epochs)
            if release_epochs > 0 and epoch_idx < release_epochs:
                alpha = float(epoch_idx) / max(float(release_epochs), 1.0)
                return start + (mid - start) * alpha
            if release_epochs > 0:
                epoch_idx = max(0, epoch_idx - release_epochs)
            if epoch_idx < mid_hold_epochs:
                return mid
            epoch_idx = max(0, epoch_idx - mid_hold_epochs)
            if end <= 0.0 or second_release_epochs <= 0:
                return mid
            if epoch_idx >= second_release_epochs:
                return end
            alpha = float(epoch_idx) / max(float(second_release_epochs), 1.0)
            return mid + (end - mid) * alpha
        if end <= 0.0 or release_epochs <= 0:
            return start
        epoch_idx = max(0, int(getattr(self, "current_epoch", 1)) - 1)
        if schedule == "hold_linear":
            if epoch_idx < hold_epochs:
                return start
            epoch_idx = max(0, epoch_idx - hold_epochs)
        if epoch_idx >= release_epochs:
            return end
        alpha = float(epoch_idx) / max(float(release_epochs), 1.0)
        return start + (end - start) * alpha

    @property
    def last_semantic_attn(self) -> torch.Tensor | None:
        for block in reversed(self.body_blocks):
            attn = getattr(block, "last_attn", None)
            if attn is not None:
                return attn
        return None

    @property
    def last_semantic_k(self) -> torch.Tensor | None:
        for block in reversed(self.body_blocks):
            k_matrix = getattr(block, "last_k", None)
            if k_matrix is not None:
                return k_matrix
        return None

    @property
    def last_semantic_topology_attn(self) -> torch.Tensor | None:
        for block in reversed(self.body_blocks):
            topo_attn = getattr(block, "last_topology_attn", None)
            if topo_attn is not None:
                return topo_attn
        return None

    @torch.no_grad()
    def endpoint_map(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        *,
        step_size: float = 1.0,
        style_strength: float | None = None,
        target_style_latent: torch.Tensor | None = None,
        style_code_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style_id is None:
            raise ValueError("style_id is required for endpoint map.")
        horizon = self._resolve_integration_horizon(step_size=step_size, style_strength=style_strength)
        if horizon <= 0.0:
            return x
        z_base = self.predict_transport_base(
            x,
            t=1.0,
            style_id=style_id,
            style_code_override=style_code_override,
            target_style_latent=target_style_latent,
        )
        return self.refine_endpoint(z_base, style_id=style_id, source_latent=x, style_code_override=style_code_override)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
        style_id: torch.Tensor | int | None = None,
        step_size: float = 1.0,
        style_strength: float | None = None,
        target_style_latent: torch.Tensor | None = None,
        style_code_override: torch.Tensor | None = None,
        override_palette: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del source, step_size, style_strength, override_palette
        if style_id is None and style_code_override is None:
            raise ValueError("style_id or style_code_override is required.")
        self.last_profile = {}
        t_tensor = self._resolve_t_input(x, t)
        t_profile = self._profile_start(x)
        style_code = self._compute_style_code(
            x=x,
            style_id=style_id,
            t=t_tensor,
            style_code_override=style_code_override,
        )
        self._profile_end("tokenizer", t_profile, x)
        if style_id is None:
            raise ValueError("style_id is required for bridge spatial conditioning.")
        t_profile = self._profile_start(x)
        delta = self._predict_delta_from_context(
            x,
            style_id=style_id,
            style_code=style_code,
            style_maps=StyleMaps(),
            override_palette=None,
            strength=1.0,
            target_style_latent=target_style_latent,
            style_code_override_active=style_code_override is not None,
        )
        self._profile_end("backbone_forward", t_profile, x)
        t_profile = self._profile_start(x)
        if self.transport_prediction_mode == "endpoint":
            endpoint_delta = self._endpoint_delta_from_raw(delta, x)
            out = self._apply_execution_budget(endpoint_delta.to(dtype=x.dtype), x, style_code)
            denom = (1.0 - t_tensor).clamp_min(self.endpoint_velocity_time_floor).view(-1, 1, 1, 1)
            out = out / denom
        else:
            out = self._apply_execution_budget(delta, x, style_code)
        self._profile_end("execution_budget", t_profile, x)
        return out

    def predict_transport_base(
        self,
        x: torch.Tensor,
        *,
        t: torch.Tensor | float | None = None,
        style_id: torch.Tensor | int | None = None,
        style_code_override: torch.Tensor | None = None,
        target_style_latent: torch.Tensor | None = None,
        override_palette: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style_id is None and style_code_override is None:
            raise ValueError("style_id or style_code_override is required.")
        self.last_profile = {}
        t_tensor = self._resolve_t_input(x, t)
        t_profile = self._profile_start(x)
        style_code = self._compute_style_code(
            x=x,
            style_id=style_id,
            t=t_tensor,
            style_code_override=style_code_override,
        )
        self._profile_end("tokenizer", t_profile, x)
        if style_id is None:
            raise ValueError("style_id is required for bridge spatial conditioning.")
        t_profile = self._profile_start(x)
        raw_transport = self._predict_delta_from_context(
            x,
            style_id=style_id,
            style_code=style_code,
            style_maps=StyleMaps(),
            override_palette=override_palette,
            strength=1.0,
            target_style_latent=target_style_latent,
            style_code_override_active=style_code_override is not None,
        )
        self._profile_end("backbone_forward", t_profile, x)
        t_profile = self._profile_start(x)
        if self.transport_prediction_mode == "endpoint":
            endpoint_delta = self._endpoint_delta_from_raw(raw_transport, x)
            budgeted_delta = self._apply_execution_budget(endpoint_delta, x, style_code)
            z_base = x + budgeted_delta
        else:
            delta = self._apply_execution_budget(raw_transport.to(dtype=x.dtype), x, style_code)
            z_base = x + delta
        self._profile_end("execution_budget", t_profile, x)
        return z_base

    @torch.no_grad()
    def integrate_transport(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        num_steps: int = 16,
        step_size: float = 1.0,
        style_strength: float | None = None,
        target_style_latent: torch.Tensor | None = None,
        style_code_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style_id is None:
            raise ValueError("style_id is required for bridge integration.")
        if self.transport_prediction_mode == "endpoint" and self.solver_family != "solver_i2sb":
            return self.predict_transport_base(
                x,
                t=1.0,
                style_id=style_id,
                style_code_override=style_code_override,
                target_style_latent=target_style_latent,
            )
        steps = max(1, int(num_steps))
        horizon = self._resolve_integration_horizon(step_size=step_size, style_strength=style_strength)
        if horizon <= 0.0:
            return x
        x = self._apply_pre_integrate_moment_match(x, target_style_latent)
        x, source_latent, _transport_stats_ctx = self._prepare_transport_stats_input(x, style_id=style_id)
        dt = horizon / float(steps)
        h = x
        self.last_i2sb_fiber_noise_debug = {}
        self.last_i2sb_fiber_project_debug = {}
        for idx in range(steps):
            t = horizon * ((idx + 0.5) / float(steps))
            t_curr = horizon * (idx / float(steps))
            t_next = horizon * ((idx + 1) / float(steps))
            if self.solver_family == "solver_i2sb" and self.transport_prediction_mode == "endpoint":
                h = self._i2sb_transport_step(
                    h,
                    t_curr=t_curr,
                    t_next=t_next,
                    style_id=style_id,
                    style_code_override=style_code_override,
                    target_style_latent=target_style_latent,
                    source_latent=source_latent,
                )
            elif self.solver_family == "solver_tangent_rk":
                h = self._rk_transport_step(
                    h,
                    t=t,
                    dt=dt,
                    style_id=style_id,
                    style_code_override=style_code_override,
                    target_style_latent=target_style_latent,
                )
            elif self.solver_family == "solver_pc":
                velocity = self.forward(
                    h,
                    t=t,
                    style_id=style_id,
                    style_code_override=style_code_override,
                    target_style_latent=target_style_latent,
                )
                h = h + velocity * dt
                h = self._correct_transport_state(h, x, dt=dt)
            elif self.solver_family == "solver_unsb_cycle":
                velocity = self.forward(
                    h,
                    t=t,
                    style_id=style_id,
                    style_code_override=style_code_override,
                    target_style_latent=target_style_latent,
                )
                predictor = h + velocity * dt
                predictor = self._correct_transport_state(predictor, x, dt=dt * 0.5)
                noise_scale = max(0.0, float(getattr(self, "solver_stochastic_noise_scale", 0.0)))
                if noise_scale > 0.0:
                    noise = self._fiber_aligned_solver_noise(
                        predictor,
                        torch.randn_like(predictor),
                        noise_scale=noise_scale,
                    )
                    predictor = predictor + noise * noise_scale * math.sqrt(max(dt, 1e-8))
                else:
                    self.last_solver_noise_debug = {
                        "fiber_gate_active": 0.0,
                        "fiber_gate_mean": 0.0,
                        "fiber_gate_rms": 0.0,
                        "noise_scale": 0.0,
                        "isotropic_or_fiber": 1.0 if bool(getattr(self, "solver_fiber_aligned", False)) else 0.0,
                    }
                h = predictor
            else:
                velocity = self.forward(
                    h,
                    t=t,
                    style_id=style_id,
                    style_code_override=style_code_override,
                    target_style_latent=target_style_latent,
                )
                h = h + velocity * dt
        return self.restore_transport_output(h, style_id=style_id)

    @torch.no_grad()
    def integrate(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        num_steps: int = 16,
        step_size: float = 1.0,
        style_strength: float | None = None,
        target_style_latent: torch.Tensor | None = None,
        style_code_override: torch.Tensor | None = None,
        override_palette: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del override_palette
        z_base = self.integrate_transport(
            x,
            style_id,
            num_steps=num_steps,
            step_size=step_size,
            style_strength=style_strength,
            target_style_latent=target_style_latent,
            style_code_override=style_code_override,
        )
        return self.refine_endpoint(z_base, style_id=style_id, source_latent=x, style_code_override=style_code_override)

    def _apply_pre_integrate_moment_match(
        self,
        x: torch.Tensor,
        target_style_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        if (not self.pre_integrate_moment_match) or target_style_latent is None:
            return x
        ref = target_style_latent
        if ref.shape != x.shape:
            raise ValueError(
                "target_style_latent shape must match model input shape, "
                f"got x={tuple(x.shape)} ref={tuple(ref.shape)}"
            )
        ref = ref.to(device=x.device, dtype=x.dtype)
        eps = self.output_moment_match_eps
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x_std = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
        ref_mean = ref.mean(dim=(2, 3), keepdim=True)
        ref_std = ref.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
        mapped = ((x - x_mean) / x_std) * ref_std + ref_mean
        return x.lerp(mapped, self.pre_integrate_moment_blend)


def _normalize_skip_routing_mode(config: ModelConfig) -> ModelConfig:
    model_cfg = config.validated()
    skip_mode = str(model_cfg.skip_routing_mode).strip().lower()
    if skip_mode not in {"none", "naive", "adaptive", "normalized"}:
        if bool(model_cfg.extra.get("skip_frequency_gated", True)):
            skip_mode = "normalized"
        else:
            skip_mode = "naive"
    model_cfg.skip_routing_mode = skip_mode
    return model_cfg


def _attach_bridge_runtime_fields(
    model_cfg: ModelConfig,
    bridge_cfg: BridgeConfig | Mapping[str, object] | None,
) -> ModelConfig:
    if bridge_cfg is None:
        return model_cfg
    bridge = bridge_cfg if isinstance(bridge_cfg, BridgeConfig) else BridgeConfig.from_mapping(bridge_cfg)
    bridge_fields = {
        "objective_mode": str(getattr(bridge, "objective_mode", "")),
        "loss_type": str(getattr(bridge, "loss_type", "")),
        "bridge_sigma": float(getattr(bridge, "bridge_sigma", 0.0)),
        "bridge_noise_schedule": str(getattr(bridge, "bridge_noise_schedule", "auto")),
        "i2sb_predictor_time_floor": float(getattr(bridge, "i2sb_predictor_time_floor", 0.0)),
        "i2sb_noise_family": str(getattr(bridge, "i2sb_noise_family", "gaussian")),
        "i2sb_style_noise_amplitude_power": float(getattr(bridge, "i2sb_style_noise_amplitude_power", 1.0)),
    }
    model_cfg.extra = dict(getattr(model_cfg, "extra", {}) or {})
    for key, value in bridge_fields.items():
        setattr(model_cfg, key, value)
        model_cfg.extra[key] = value
    return model_cfg


def build_model_from_config(
    model_cfg: ModelConfig | Mapping[str, object],
    *,
    bridge_cfg: BridgeConfig | Mapping[str, object] | None = None,
    use_checkpointing: bool = False,
) -> TimeConditionedLANCETBridge:
    config = model_cfg if isinstance(model_cfg, ModelConfig) else ModelConfig.from_mapping(model_cfg)
    config = _attach_bridge_runtime_fields(config, bridge_cfg)
    config = _normalize_skip_routing_mode(config)
    config.use_checkpointing = bool(use_checkpointing)
    return TimeConditionedLANCETBridge(config)


__all__ = [
    "TimeConditionedLANCETBridge",
    "build_model_from_config",
    "count_parameters",
]
