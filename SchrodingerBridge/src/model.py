from __future__ import annotations

import math
import time
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_schema import ModelConfig
from lancet_blocks import DecoderTextureBlock, NormFreeModulation, StyleMaps
from lancet_backbone import LatentAdaCUT, count_parameters
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
        self.velocity_head_mode = str(bridge_config.velocity_head_mode).strip().lower()
        self.velocity_tanh_limit = max(1e-3, float(bridge_config.velocity_tanh_limit))
        self.transport_prediction_mode = str(getattr(bridge_config, "transport_prediction_mode", "velocity")).strip().lower()
        if self.transport_prediction_mode not in {"velocity", "endpoint"}:
            self.transport_prediction_mode = "velocity"
        self.transport_endpoint_scale = max(1e-3, float(getattr(bridge_config, "transport_endpoint_scale", 4.0)))
        self.bridge_style_dim = int(self.style_tokenizer.embedding_dim)
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
        if self.style_injection_form not in {"mixed", "carrier_gate"}:
            self.style_injection_form = "mixed"
        self.style_injection_scale = max(0.0, float(getattr(bridge_config, "style_injection_scale", 1.0)))
        self.style_injection_gate_log_span = max(0.0, float(getattr(bridge_config, "style_injection_gate_log_span", 0.4054651081081644)))
        injection_in_dim = self.bridge_style_dim + self.execution_budget_feature_dim
        self.body_style_injector: nn.Module | None = None
        self.decoder_style_injector: nn.Module | None = None
        self.body_style_carrier: nn.Module | None = None
        self.body_content_gate: nn.Module | None = None
        self.decoder_style_carrier: nn.Module | None = None
        self.decoder_content_gate: nn.Module | None = None
        if self.style_injection_mode in {"body", "body_decoder"}:
            if self.style_injection_form == "carrier_gate":
                self.body_style_carrier, self.body_content_gate = self._make_carrier_gate_injector(
                    self.bridge_style_dim,
                    self.execution_budget_feature_dim,
                    int(self.body_channels),
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
            else:
                self.decoder_style_injector = self._make_style_injector(
                    injection_in_dim,
                    int(self.lift_channels),
                    int(getattr(bridge_config, "style_injection_hidden_dim", 64)),
                )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.bridge_style_dim),
            nn.SiLU(),
            nn.Linear(self.bridge_style_dim, self.bridge_style_dim),
        )
        self.proximal_mode = str(getattr(bridge_config, "proximal_mode", "off")).strip().lower()
        if self.proximal_mode not in {"off", "highpass_residual", "normfree_modulation", "crossattn_texture"}:
            self.proximal_mode = "off"
        self.proximal_hidden_channels = max(4, int(getattr(bridge_config, "proximal_hidden_channels", self.latent_channels)))
        self.proximal_num_blocks = max(1, int(getattr(bridge_config, "proximal_num_blocks", 2)))
        self.proximal_highpass_kernel = max(1, int(getattr(bridge_config, "proximal_highpass_kernel", 5)))
        if self.proximal_highpass_kernel % 2 == 0:
            self.proximal_highpass_kernel += 1
        self.proximal_residual_energy_weight = max(0.0, float(getattr(bridge_config, "proximal_residual_energy_weight", 0.0)))
        self.proximal_trust_ratio = max(0.0, float(getattr(bridge_config, "proximal_trust_ratio", 0.0)))
        self.proximal_trust_weight = max(0.0, float(getattr(bridge_config, "proximal_trust_weight", 0.0)))
        self.proximal_clamp_ratio = max(0.0, float(getattr(bridge_config, "proximal_clamp_ratio", 0.0)))
        self.proximal_clamp_ratio_end = max(0.0, float(getattr(bridge_config, "proximal_clamp_ratio_end", 0.0)))
        self.proximal_clamp_schedule = str(getattr(bridge_config, "proximal_clamp_schedule", "linear")).strip().lower()
        if self.proximal_clamp_schedule not in {"linear", "hold_linear"}:
            self.proximal_clamp_schedule = "linear"
        self.proximal_clamp_hold_epochs = max(0, int(getattr(bridge_config, "proximal_clamp_hold_epochs", 0)))
        self.proximal_clamp_release_epochs = max(0, int(getattr(bridge_config, "proximal_clamp_release_epochs", 0)))
        self.proximal_force_highpass = bool(getattr(bridge_config, "proximal_force_highpass", True))
        self.proximal_bind_terminal_losses = bool(getattr(bridge_config, "proximal_bind_terminal_losses", True))
        self.record_base_endpoint_metrics = bool(getattr(bridge_config, "record_base_endpoint_metrics", False))
        self.proximal_texture_blocks: nn.ModuleList | None = None
        self.proximal_normfree_mod: NormFreeModulation | None = None
        self.proximal_normfree_head: nn.Sequential | None = None
        self.proximal_attn_q: nn.Conv2d | None = None
        self.proximal_attn_k: nn.Conv2d | None = None
        self.proximal_attn_v: nn.Conv2d | None = None
        self.proximal_attn_out: nn.Conv2d | None = None
        self.proximal_style_tokens: nn.Linear | None = None
        if self.proximal_mode == "highpass_residual":
            self.proximal_texture_blocks = nn.ModuleList(
                [
                    DecoderTextureBlock(
                        dim=int(self.latent_channels),
                        style_dim=int(self.bridge_style_dim),
                        num_groups=max(1, int(self.config.num_groups)),
                    )
                    for _ in range(self.proximal_num_blocks)
                ]
            )
        elif self.proximal_mode == "normfree_modulation":
            self.proximal_normfree_mod = NormFreeModulation(int(self.latent_channels), int(self.bridge_style_dim))
            hidden = int(self.proximal_hidden_channels)
            self.proximal_normfree_head = nn.Sequential(
                nn.Conv2d(int(self.latent_channels), hidden, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden, int(self.latent_channels), kernel_size=3, stride=1, padding=1),
            )
        elif self.proximal_mode == "crossattn_texture":
            hidden = int(self.proximal_hidden_channels)
            self.proximal_attn_q = nn.Conv2d(int(self.latent_channels), hidden, kernel_size=1, stride=1, padding=0)
            self.proximal_attn_k = nn.Conv2d(int(self.body_channels), hidden, kernel_size=1, stride=1, padding=0)
            self.proximal_attn_v = nn.Conv2d(int(self.body_channels), hidden, kernel_size=1, stride=1, padding=0)
            self.proximal_attn_out = nn.Conv2d(hidden, int(self.latent_channels), kernel_size=1, stride=1, padding=0)
            self.proximal_style_tokens = nn.Linear(int(self.bridge_style_dim), int(self.body_channels))
            for mod in (self.proximal_attn_q, self.proximal_attn_k, self.proximal_attn_v, self.proximal_attn_out):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            nn.init.normal_(self.proximal_style_tokens.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.proximal_style_tokens.bias)
        self.profile_modules = False
        self.profile_sync_cuda = False
        self.last_profile: dict[str, float] = {}
        self.last_proximal_residual: torch.Tensor | None = None
        self.last_base_endpoint: torch.Tensor | None = None
        self.last_final_endpoint: torch.Tensor | None = None
        self.last_proximal_clamp_scale: torch.Tensor | None = None
        self.current_epoch: int = 1
        self.total_epochs: int = 1

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

    def _apply_style_feature_injection(
        self,
        feat: torch.Tensor,
        x: torch.Tensor,
        style_code: torch.Tensor,
        *,
        site: str,
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
        else:
            injector = self.body_style_injector if site == "body" else self.decoder_style_injector
            if injector is None:
                return feat
            inject_in = torch.cat([style_code, content_feat], dim=1)
            bias = torch.tanh(injector(inject_in).float()).to(dtype=feat.dtype)
        return feat + bias.view(feat.shape[0], feat.shape[1], 1, 1) * self.style_injection_scale

    def _compute_delta(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
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
                return stroked.float()
            return stroked - x.float()
        if self.transport_prediction_mode == "endpoint":
            return torch.tanh(raw_delta / self.transport_endpoint_scale) * self.transport_endpoint_scale
        if self.velocity_head_mode == "tanh":
            raw_delta = torch.tanh(raw_delta) * self.velocity_tanh_limit
        return raw_delta * self.latent_scale_factor * self.residual_gain

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
        return max(0.0, min(1.0, horizon))

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

    def _style_spatial_tokens(self, z_base: torch.Tensor, style_id: torch.Tensor | int) -> torch.Tensor:
        style_map = self.encode_style_spatial_id(style_id).get(16)
        style_map = F.interpolate(style_map.to(device=z_base.device, dtype=z_base.dtype), size=z_base.shape[-2:], mode="bilinear", align_corners=False)
        if self.proximal_style_tokens is not None:
            style_code = self.encode_style_id(style_id).to(device=z_base.device, dtype=z_base.dtype)
            token_bias = self.proximal_style_tokens(style_code).view(style_code.shape[0], self.body_channels, 1, 1)
            style_map = style_map + token_bias
        return style_map

    def refine_endpoint(
        self,
        z_base: torch.Tensor,
        *,
        style_id: torch.Tensor | int | None,
        source_latent: torch.Tensor | None = None,
        style_code_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.last_base_endpoint = z_base.detach()
        if self.proximal_mode == "off":
            self.last_proximal_residual = torch.zeros_like(z_base)
            self.last_proximal_clamp_scale = torch.ones((), device=z_base.device, dtype=z_base.dtype)
            self.last_final_endpoint = z_base.detach()
            return z_base
        style_code = self._resolve_refine_style_code(
            z_base,
            style_id=style_id,
            style_code_override=style_code_override,
        )
        if self.proximal_mode == "highpass_residual":
            if self.proximal_texture_blocks is None:
                raise RuntimeError("proximal_texture_blocks not initialized")
            h = z_base
            for block in self.proximal_texture_blocks:
                h = block(h, style_code, gate=1.0)
            delta = h - z_base
        elif self.proximal_mode == "normfree_modulation":
            if self.proximal_normfree_mod is None or self.proximal_normfree_head is None:
                raise RuntimeError("normfree proximal modules not initialized")
            mod = self.proximal_normfree_mod(z_base, style_code, gate=1.0)
            delta = self.proximal_normfree_head(mod)
        else:
            if (
                self.proximal_attn_q is None
                or self.proximal_attn_k is None
                or self.proximal_attn_v is None
                or self.proximal_attn_out is None
            ):
                raise RuntimeError("cross-attention proximal modules not initialized")
            if style_id is None:
                raise ValueError("style_id is required for crossattn_texture proximal mode.")
            q = self.proximal_attn_q(z_base.float())
            kv_src = self._style_spatial_tokens(z_base, style_id).float()
            k = self.proximal_attn_k(kv_src)
            v = self.proximal_attn_v(kv_src)
            bsz, ch, h_dim, w_dim = q.shape
            q_flat = q.view(bsz, ch, -1).transpose(1, 2)
            k_flat = k.view(bsz, ch, -1)
            attn = torch.softmax(torch.bmm(q_flat, k_flat) / math.sqrt(float(ch)), dim=-1)
            v_flat = v.view(bsz, ch, -1).transpose(1, 2)
            mixed = torch.bmm(attn, v_flat).transpose(1, 2).view(bsz, ch, h_dim, w_dim)
            delta = self.proximal_attn_out(mixed).to(dtype=z_base.dtype)
        delta = self._apply_proximal_highpass(delta)
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
        self.last_proximal_residual = delta.detach()
        self.last_proximal_clamp_scale = clamp_scale.detach()
        self.last_final_endpoint = z_final.detach()
        return z_final

    def _resolve_proximal_clamp_ratio(self) -> float:
        start = float(self.proximal_clamp_ratio)
        end = float(self.proximal_clamp_ratio_end)
        schedule = str(getattr(self, "proximal_clamp_schedule", "linear")).strip().lower()
        hold_epochs = max(0, int(getattr(self, "proximal_clamp_hold_epochs", 0)))
        release_epochs = int(self.proximal_clamp_release_epochs)
        if start <= 0.0:
            return 0.0
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
            endpoint = delta
            endpoint_delta = endpoint - x.float()
            out = self._apply_execution_budget(endpoint_delta.to(dtype=x.dtype), x, style_code)
            denom = (1.0 - t_tensor).clamp_min(1e-3).view(-1, 1, 1, 1)
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
            endpoint_delta = raw_transport.to(dtype=x.dtype) - x.float()
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
        if self.transport_prediction_mode == "endpoint":
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


def build_model_from_config(
    model_cfg: ModelConfig | Mapping[str, object],
    *,
    use_checkpointing: bool = False,
) -> TimeConditionedLANCETBridge:
    config = model_cfg if isinstance(model_cfg, ModelConfig) else ModelConfig.from_mapping(model_cfg)
    config = _normalize_skip_routing_mode(config)
    config.use_checkpointing = bool(use_checkpointing)
    return TimeConditionedLANCETBridge(config)


__all__ = [
    "TimeConditionedLANCETBridge",
    "build_model_from_config",
    "count_parameters",
]
