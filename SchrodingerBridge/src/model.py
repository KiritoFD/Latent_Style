from __future__ import annotations

import math
import time
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_schema import BridgeConfig, ModelConfig
from lancet_blocks import StyleMaps, _gumbel_hard_attention, _sinkhorn_attention
from lancet_backbone import LatentAdaCUT, count_parameters
from style_families import SOLVER_FAMILIES, normalize_family, validate_i2sb_contract
from utils.diffeomorphic import apply_texture_aligned_diffeomorphic_stroke


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
        if self.endpoint_parameterization not in {"absolute", "residual", "blend"}:
            self.endpoint_parameterization = "absolute"
        self.endpoint_residual_blend = float(getattr(bridge_config, "endpoint_residual_blend", 0.0))
        self.endpoint_residual_blend = min(1.0, max(0.0, self.endpoint_residual_blend))
        validate_i2sb_contract(
            solver_family=self.solver_family,
            transport_prediction_mode=self.transport_prediction_mode,
            objective_mode=str(getattr(bridge_config, "objective_mode", "")),
            loss_type=str(getattr(bridge_config, "loss_type", "")),
            bridge_noise_schedule=str(getattr(bridge_config, "bridge_noise_schedule", "auto")),
        )
        self.transport_endpoint_scale = max(1e-3, float(getattr(bridge_config, "transport_endpoint_scale", 4.0)))
        self.objective_mode = str(getattr(bridge_config, "objective_mode", "")).strip().lower()
        self.loss_type = str(getattr(bridge_config, "loss_type", "")).strip().lower()
        self.bridge_sigma = max(0.0, float(getattr(bridge_config, "bridge_sigma", 0.0)))
        self.i2sb_predictor_time_floor = max(0.0, float(getattr(bridge_config, "i2sb_predictor_time_floor", 0.0)))
        self.endpoint_velocity_time_floor = float(getattr(bridge_config, "endpoint_velocity_time_floor", 0.05))
        if self.endpoint_velocity_time_floor < 0.01:
            raise ValueError(
                "model.endpoint_velocity_time_floor must be >= 0.01 for endpoint velocity mode. "
                "Lower floors amplify endpoint deltas near t=1 and can silently destabilize training."
            )
        self.solver_corrector_mode = self._normalize_solver_corrector_mode(
            str(getattr(bridge_config, "solver_corrector_mode", "none"))
        )
        self.allow_style_overdrive = bool(getattr(bridge_config, "allow_style_overdrive", False))
        self.last_i2sb_transport_debug: dict[str, float] = {}
        self.solver_stochastic_noise_scale = max(0.0, float(getattr(bridge_config, "solver_stochastic_noise_scale", 0.0)))
        self.solver_fiber_aligned = bool(getattr(bridge_config, "solver_fiber_aligned", False))
        self.last_solver_noise_debug: dict[str, float] = {}
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
                )
            elif self.style_injection_form == "spatial_carrier_gate":
                self.body_style_spatial_proj, self.body_content_gate, self.body_structure_gate = self._make_spatial_carrier_gate_injector(
                    int(self.body_channels),
                    int(self.body_channels),
                    self.execution_budget_feature_dim,
                    int(self.latent_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                )
            else:
                self.body_style_injector = self._make_style_injector(
                    injection_in_dim,
                    int(self.body_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                )
        if self.style_injection_mode in {"decoder", "body_decoder"}:
            if self.style_injection_form == "carrier_gate":
                self.decoder_style_carrier, self.decoder_content_gate = self._make_carrier_gate_injector(
                    self.bridge_style_dim,
                    self.execution_budget_feature_dim,
                    int(self.lift_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                )
            elif self.style_injection_form == "spatial_carrier_gate":
                self.decoder_style_spatial_proj, self.decoder_content_gate, self.decoder_structure_gate = self._make_spatial_carrier_gate_injector(
                    int(self.body_channels),
                    int(self.lift_channels),
                    self.execution_budget_feature_dim,
                    int(self.latent_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                )
            else:
                self.decoder_style_injector = self._make_style_injector(
                    injection_in_dim,
                    int(self.lift_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
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
            # The proximal texture branch is a controlled residual mechanism.
            # Start from exact identity so early metrics measure learned texture,
            # not random endpoint perturbation.
            nn.init.zeros_(self.proximal_attn_out.weight)
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
        self.last_style_delta_debug = {}
        for module in self.modules():
            if hasattr(module, "last_attn"):
                setattr(module, "last_attn", None)
            if hasattr(module, "last_k"):
                setattr(module, "last_k", None)

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

    @staticmethod
    def _make_style_injector(input_dim: int, channels: int, hidden_dim: int) -> nn.Module:
        hidden = max(4, int(hidden_dim))
        module = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(channels)),
        )
        last = module[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        return module

    @staticmethod
    def _make_carrier_gate_injector(
        style_dim: int,
        content_dim: int,
        channels: int,
        hidden_dim: int,
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
        for module in (carrier, gate):
            last = module[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        return carrier, gate

    @staticmethod
    def _make_spatial_carrier_gate_injector(
        style_map_channels: int,
        feat_channels: int,
        content_dim: int,
        source_channels: int,
        hidden_dim: int,
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
        for module in (style_proj, content_gate, structure_gate):
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
        side = torch.tanh(side.float()).to(dtype=delta.dtype) * float(self.style_head_adapter_scale)
        with torch.no_grad():
            delta_rms = delta.detach().float().square().mean().sqrt().clamp_min(1e-8)
            side_rms = side.detach().float().square().mean().sqrt()
            self.last_style_delta_debug = {
                "style_head_adapter_active": 1.0,
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
                    return self._apply_style_head_adapter(delta, h, style_code)
                endpoint = self._apply_style_delta_basis(stroked.float(), h, style_code)
                return self._apply_style_head_adapter(endpoint, h, style_code)
            delta = stroked - x.float()
            delta = self._apply_style_delta_basis(delta, h, style_code)
            return self._apply_style_head_adapter(delta, h, style_code)
        if self.transport_prediction_mode == "endpoint":
            bounded = torch.tanh(raw_delta / self.transport_endpoint_scale) * self.transport_endpoint_scale
            if self.endpoint_parameterization == "residual":
                bounded = self._apply_style_delta_basis(bounded, h, style_code)
                return self._apply_style_head_adapter(bounded, h, style_code)
            bounded = self._apply_style_delta_basis(bounded, h, style_code)
            return self._apply_style_head_adapter(bounded, h, style_code)
        if self.velocity_head_mode == "tanh":
            raw_delta = torch.tanh(raw_delta) * self.velocity_tanh_limit
        delta = raw_delta * self.latent_scale_factor * self.residual_gain
        delta = self._apply_style_delta_basis(delta, h, style_code)
        return self._apply_style_head_adapter(delta, h, style_code)

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
            if str(getattr(self, "tokenizer_family", "legacy_factorized")) in {"pure_latent_spatial", "smoe_translator"}:
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
    ) -> torch.Tensor:
        velocity = self.forward(h, t=t, style_id=style_id, style_code_override=style_code_override)
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
    ) -> torch.Tensor:
        predict_t = float(t_curr)
        if self.i2sb_predictor_time_floor > 0.0 and predict_t < self.i2sb_predictor_time_floor:
            predict_t = min(max(predict_t, self.i2sb_predictor_time_floor), float(t_next))
        predict_t = min(max(predict_t, 0.0), 1.0 - 1e-6)
        self.last_i2sb_transport_debug = {
            "t_curr": float(t_curr),
            "predict_t": float(predict_t),
            "t_next": float(t_next),
            "time_floor": float(self.i2sb_predictor_time_floor),
            "time_floor_active": float(abs(predict_t - float(t_curr)) > 1e-9),
        }
        x_1_pred = self.predict_transport_base(
            h,
            t=predict_t,
            style_id=style_id,
            style_code_override=style_code_override,
            target_style_latent=target_style_latent,
        )
        denom = max(1.0 - float(t_curr), 1e-6)
        c_curr = (1.0 - float(t_next)) / denom
        c_target = (float(t_next) - float(t_curr)) / denom
        mu = c_curr * h + c_target * x_1_pred
        if float(t_next) >= 1.0 - 1e-4 or self.bridge_sigma <= 0.0:
            return mu
        var = (self.bridge_sigma ** 2) * (float(t_next) - float(t_curr)) * (1.0 - float(t_next)) / denom
        if var <= 0.0:
            return mu
        return mu + math.sqrt(var) * torch.randn_like(h)

    def _rk_transport_step(
        self,
        h: torch.Tensor,
        *,
        t: float,
        dt: float,
        style_id: torch.Tensor | int | None,
        style_code_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        order = max(2, int(getattr(self, "solver_rk_order", 4)))
        if order <= 2:
            k1 = self._transport_velocity(h, t=t, style_id=style_id, style_code_override=style_code_override)
            k2 = self._transport_velocity(h + 0.5 * dt * k1, t=t + 0.5 * dt, style_id=style_id, style_code_override=style_code_override)
            return h + dt * k2
        k1 = self._transport_velocity(h, t=t, style_id=style_id, style_code_override=style_code_override)
        k2 = self._transport_velocity(h + 0.5 * dt * k1, t=t + 0.5 * dt, style_id=style_id, style_code_override=style_code_override)
        k3 = self._transport_velocity(h + 0.5 * dt * k2, t=t + 0.5 * dt, style_id=style_id, style_code_override=style_code_override)
        k4 = self._transport_velocity(h + dt * k3, t=t + dt, style_id=style_id, style_code_override=style_code_override)
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
        )
        structured_ctx = self._structured_style_from_sidecar(
            style_id=style_id,
            style_code=style_code,
            content_latent=content_latent,
            content_feat_16=content_feat_16,
        )
        if structured_ctx is not None:
            return structured_ctx
        return style_code, self._prepare_style_maps(style_id)

    def _style_spatial_tokens(self, z_base: torch.Tensor, style_id: torch.Tensor | int) -> torch.Tensor:
        style_map = self.encode_style_spatial_id(style_id).get(16)
        style_map = F.interpolate(style_map.to(device=z_base.device, dtype=z_base.dtype), size=z_base.shape[-2:], mode="bilinear", align_corners=False)
        if self.proximal_style_tokens is not None:
            style_code = self.encode_style_id(style_id).to(device=z_base.device, dtype=z_base.dtype)
            token_bias = self.proximal_style_tokens(style_code).view(style_code.shape[0], self.body_channels, 1, 1)
            style_map = style_map + token_bias
        return style_map

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
            if str(getattr(self, "tokenizer_family", "legacy_factorized")) in {"pure_latent_spatial", "smoe_translator"}:
                style_code, kv_src = self._structured_proximal_style_tokens(
                    z_base,
                    style_id=style_id,
                    style_code_override=style_code_override,
                    source_latent=source_latent,
                )
            else:
                style_code = self._resolve_refine_style_code(
                    z_base,
                    style_id=style_id,
                    style_code_override=style_code_override,
                )
                kv_src = self._style_spatial_tokens(z_base, style_id).float()
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
        del source, step_size, style_strength, target_style_latent, override_palette
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
            target_style_latent=None,
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
        dt = horizon / float(steps)
        h = x
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
                )
            elif self.solver_family == "solver_tangent_rk":
                h = self._rk_transport_step(
                    h,
                    t=t,
                    dt=dt,
                    style_id=style_id,
                    style_code_override=style_code_override,
                )
            elif self.solver_family == "solver_pc":
                velocity = self.forward(h, t=t, style_id=style_id, style_code_override=style_code_override)
                h = h + velocity * dt
                h = self._correct_transport_state(h, x, dt=dt)
            elif self.solver_family == "solver_unsb_cycle":
                velocity = self.forward(h, t=t, style_id=style_id, style_code_override=style_code_override)
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
                velocity = self.forward(h, t=t, style_id=style_id, style_code_override=style_code_override)
                h = h + velocity * dt
        return h

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
        "i2sb_predictor_time_floor": float(getattr(bridge, "i2sb_predictor_time_floor", 0.0)),
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
