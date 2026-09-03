from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_count(channels: int, preferred: int) -> int:
    groups = max(1, min(int(preferred), int(channels)))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _sinkhorn_attention(logits: torch.Tensor, *, iters: int = 3, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert attention logits into a near doubly-stochastic routing matrix.

    This is an experimental routing option for square semantic maps. It keeps
    the standard row-normalized attention semantics after the final step while
    discouraging a few style locations from absorbing all content locations.
    """
    log_p = F.log_softmax(logits.float(), dim=-1)
    for _ in range(max(1, int(iters))):
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=2, keepdim=True)
    return torch.exp(log_p).clamp_min(eps).to(dtype=logits.dtype)


def _gumbel_hard_attention(logits: torch.Tensor, *, tau: float = 1.0) -> torch.Tensor:
    flat = logits.reshape(-1, logits.shape[-1])
    attn = F.gumbel_softmax(flat, tau=max(1e-3, float(tau)), hard=True, dim=-1)
    return attn.view_as(logits).to(dtype=logits.dtype)



def _normalize_feature_block_type(block_type: str) -> str:
    kind = str(block_type).strip().lower()
    aliases = {
        "cnn": "conv",
        "res": "conv",
        "resblock": "conv",
        "global": "global_attn",
        "global_attention": "global_attn",
        "attn": "global_attn",
        "window": "window_attn",
        "window_attention": "window_attn",
        "windowed_attn": "window_attn",
    }
    return aliases.get(kind, kind if kind in {"conv", "global_attn", "window_attn"} else "conv")




class SemanticCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_groups: int = 4,
        temperature: float = 0.08,
        paint_only: bool = False,
        routing_mode: str = "softmax",
        sinkhorn_iters: int = 3,
        gumbel_tau: float = 1.0,
        self_topology_gate: bool = False,
        self_topology_blend: float = 1.0,
    ) -> None:
        super().__init__()
        self.paint_only = bool(paint_only)
        self.routing_mode = str(routing_mode).strip().lower()
        if self.routing_mode not in {"softmax", "sinkhorn", "gumbel_hard"}:
            self.routing_mode = "softmax"
        self.sinkhorn_iters = max(1, int(sinkhorn_iters))
        self.gumbel_tau = max(1e-3, float(gumbel_tau))
        self.self_topology_gate = bool(self_topology_gate)
        self.self_topology_blend = max(0.0, min(1.0, float(self_topology_blend)))
        self.norm_x = nn.GroupNorm(_resolve_group_count(dim, num_groups), dim)
        self.norm_s = nn.GroupNorm(_resolve_group_count(dim, num_groups), dim)
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.log_temp = nn.Parameter(torch.tensor([math.log(max(1e-4, float(temperature)))], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.last_attn: torch.Tensor | None = None
        self.last_k: torch.Tensor | None = None
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        nn.init.kaiming_normal_(self.gate_conv[0].weight, a=math.sqrt(5))
        if self.gate_conv[0].bias is not None:
            fan_in0, _ = nn.init._calculate_fan_in_and_fan_out(self.gate_conv[0].weight)
            bound0 = 1 / math.sqrt(fan_in0) if fan_in0 > 0 else 0
            nn.init.uniform_(self.gate_conv[0].bias, -bound0, bound0)
        nn.init.kaiming_normal_(self.gate_conv[-1].weight, a=math.sqrt(5))
        if self.gate_conv[-1].bias is not None:
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.gate_conv[-1].weight)
            bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
            nn.init.uniform_(self.gate_conv[-1].bias, -bound1, bound1)

    def forward(
        self,
        x: torch.Tensor,
        style_map: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        b, c, h_dim, w_dim = x.shape
        if style_map.shape[:1] != x.shape[:1]:
            raise ValueError(f"batch size mismatch: x={tuple(x.shape)} style_map={tuple(style_map.shape)}")
        if style_map.shape[1] != c:
            raise ValueError(f"x channels {c} must match style_map channels {style_map.shape[1]}")
        if style_map.shape[-2:] != (h_dim, w_dim):
            raise ValueError(f"spatial mismatch: x={tuple(x.shape)} style_map={tuple(style_map.shape)}")

        nx = self.norm_x(x)
        ns = self.norm_s(style_map)

        q_dehydrated = F.instance_norm(nx, eps=1e-3)
        k_dehydrated = F.instance_norm(ns, eps=1e-3)
        q = self.to_q(q_dehydrated).view(b, c, -1).transpose(1, 2)
        k = self.to_k(k_dehydrated).view(b, c, -1)
        v = self.to_v(ns).view(b, c, -1).transpose(1, 2)
        q = torch.nan_to_num(q.float(), nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
        k = torch.nan_to_num(k.float(), nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)
        v = torch.nan_to_num(v.float(), nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)

        temp = torch.exp(self.log_temp).clamp(1e-4, 10.0)
        scale = (c ** -0.5) / temp
        attn = torch.bmm(q, k) * scale
        attn = torch.nan_to_num(attn, nan=0.0, posinf=50.0, neginf=-50.0).clamp_(-50.0, 50.0)
        if self.routing_mode == "sinkhorn":
            attn = _sinkhorn_attention(attn, iters=self.sinkhorn_iters)
        elif self.routing_mode == "gumbel_hard":
            attn = _gumbel_hard_attention(attn, tau=self.gumbel_tau)
        else:
            attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0).to(dtype=x.dtype)
        self.last_attn = attn
        self.last_k = F.normalize(k, p=2, dim=1)
        painted_tokens = torch.bmm(attn.float(), v).to(dtype=x.dtype)
        if self.self_topology_gate and self.self_topology_blend > 0.0:
            self_logits = torch.bmm(q, q.transpose(1, 2)) * scale
            self_logits = torch.nan_to_num(self_logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp_(-50.0, 50.0)
            self_attn = F.softmax(self_logits, dim=-1)
            self_attn = torch.nan_to_num(self_attn, nan=0.0, posinf=1.0, neginf=0.0)
            topology_painted = torch.bmm(self_attn, painted_tokens).to(dtype=painted_tokens.dtype)
            painted_tokens = torch.lerp(painted_tokens, topology_painted, self.self_topology_blend)
        painted = painted_tokens.transpose(1, 2).view(b, c, h_dim, w_dim)

        if self.paint_only:
            return painted

        learned_gate = torch.sigmoid(self.gate_conv(nx))
        final_gate = gate if isinstance(gate, float) else gate.to(device=x.device, dtype=x.dtype)
        delta = painted * (1.0 + self.gamma) * learned_gate
        return x + final_gate * delta


class StyleBlender(nn.Module):
    def __init__(
        self,
        dim: int,
        num_groups: int = 8,
        *,
        init_logit: float = 0.5,
        residual: bool = False,
        residual_strength: float = 1.0,
        mode: str = "replace",
        mod_strength: float = 1.0,
        mod_tanh_scale: float = 0.5,
        band_strength: float = 1.0,
        band_tanh_scale: float = 0.75,
        band_outer_kernel: int = 9,
        band_gate_kernel: int = 5,
        band_gate_gamma: float = 3.0,
        band_gate_floor: float = 0.15,
        dual_low_strength: float = 0.20,
        dual_mid_strength: float = 0.70,
        dual_high_strength: float = 0.00,
        dual_low_kernel: int = 11,
        dual_mid_inner_kernel: int = 3,
        dual_mid_outer_kernel: int = 11,
        dual_phase_gamma: float = 3.0,
        dual_phase_floor: float = 0.35,
        region_bins: int = 5,
        region_gamma: float = 4.0,
        region_floor: float = 0.18,
        region_smooth_kernel: int = 7,
        region_hidden_mult: float = 0.5,
        region_low_strength: float = 0.30,
        region_mid_strength: float = 0.80,
        region_high_strength: float = 0.02,
        transport_gamma: float = 4.0,
        transport_floor: float = 0.12,
        transport_power: float = 1.0,
        transport_use_entropy: bool = True,
        transport_use_uniqueness: bool = True,
        transport_low_use_support: bool = True,
        transport_low_strength: float = 0.24,
        transport_mid_strength: float = 0.88,
        transport_high_strength: float = 0.04,
        adain_moment_kernel: int = 7,
        adain_eps: float = 1e-4,
        amp_gamma: float = 3.0,
        amp_floor: float = 0.30,
        amp_low_strength: float = 0.30,
        amp_mid_strength: float = 0.90,
        amp_high_strength: float = 0.04,
        texton_hidden_mult: float = 0.75,
        texton_tanh_scale: float = 0.45,
        texton_low_strength: float = 0.18,
        texton_mid_strength: float = 0.72,
        texton_high_strength: float = 0.05,
        token_flatten_strength: float = 0.0,
        token_flatten_kernel: int = 5,
        token_adain_gate_enable: bool = False,
        token_reader_enable: bool = False,
        token_reader_hidden: int = 32,
        token_reader_scale: float = 0.20,
        token_grammar_texture_enable: bool = False,
        token_grammar_texture_scale: float = 0.35,
        token_texton_carrier_enable: bool = False,
        token_texton_carrier_strength: float = 0.12,
        token_texton_carrier_hidden_mult: float = 0.75,
        token_texton_carrier_tanh_scale: float = 0.45,
        token_prototype_carrier_enable: bool = False,
        token_prototype_carrier_strength: float = 0.16,
        token_prototype_carrier_hidden_mult: float = 0.75,
        token_prototype_carrier_tanh_scale: float = 0.45,
        token_depthwise_filter_enable: bool = False,
        token_depthwise_filter_strength: float = 0.0,
        token_depthwise_filter_tanh_scale: float = 0.35,
        token_depthwise_filter_basis_offset: int = 8,
        token_depthwise_filter_learnable_gate: bool = False,
        token_depthwise_filter_learnable_gate_scale: float = 0.5,
        token_depthwise_filter_style_basis_gate: bool = False,
        token_depthwise_filter_style_basis_gate_scale: float = 0.75,
        token_depthwise_filter_style_basis_delta: bool = False,
        token_depthwise_filter_style_basis_delta_scale: float = 0.30,
        num_styles: int = 1,
        token_identity_dim: int = 16,
        token_grammar_dim: int = 9,
        token_band_dim: int = 3,
    ) -> None:
        super().__init__()
        groups = _resolve_group_count(dim, num_groups)
        self.norm = nn.GroupNorm(groups, dim)
        self.content_norm = nn.GroupNorm(groups, dim, affine=False)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.mod_mapper = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
        )
        nn.init.zeros_(self.mod_mapper[-1].weight)
        nn.init.zeros_(self.mod_mapper[-1].bias)
        self.alpha = nn.Parameter(torch.ones(1) * float(init_logit))
        self.residual = bool(residual)
        self.residual_strength = max(0.0, float(residual_strength))
        self.mode = str(mode).strip().lower()
        if self.mode not in {
            "replace",
            "modulate",
            "residual_band",
            "residual_dual",
            "region_paint",
            "transport_paint",
            "transport_adain",
            "transport_amp",
            "transport_texton",
        }:
            self.mode = "replace"
        self.mod_strength = max(0.0, float(mod_strength))
        self.mod_tanh_scale = max(1e-4, float(mod_tanh_scale))
        self.band_strength = max(0.0, float(band_strength))
        self.band_tanh_scale = max(1e-4, float(band_tanh_scale))
        self.band_outer_kernel = max(1, int(band_outer_kernel))
        if self.band_outer_kernel % 2 == 0:
            self.band_outer_kernel += 1
        self.band_gate_kernel = max(1, int(band_gate_kernel))
        if self.band_gate_kernel % 2 == 0:
            self.band_gate_kernel += 1
        self.band_gate_gamma = max(0.0, float(band_gate_gamma))
        self.band_gate_floor = max(0.0, min(1.0, float(band_gate_floor)))
        self.dual_low_strength = max(0.0, float(dual_low_strength))
        self.dual_mid_strength = max(0.0, float(dual_mid_strength))
        self.dual_high_strength = max(0.0, float(dual_high_strength))
        self.dual_low_kernel = max(1, int(dual_low_kernel))
        if self.dual_low_kernel % 2 == 0:
            self.dual_low_kernel += 1
        self.dual_mid_inner_kernel = max(1, int(dual_mid_inner_kernel))
        if self.dual_mid_inner_kernel % 2 == 0:
            self.dual_mid_inner_kernel += 1
        self.dual_mid_outer_kernel = max(self.dual_mid_inner_kernel + 2, int(dual_mid_outer_kernel))
        if self.dual_mid_outer_kernel % 2 == 0:
            self.dual_mid_outer_kernel += 1
        self.dual_phase_gamma = max(0.0, float(dual_phase_gamma))
        self.dual_phase_floor = max(0.0, min(1.0, float(dual_phase_floor)))
        self.region_bins = max(2, int(region_bins))
        self.region_gamma = max(1e-4, float(region_gamma))
        self.region_floor = max(0.0, min(1.0, float(region_floor)))
        self.region_smooth_kernel = max(1, int(region_smooth_kernel))
        if self.region_smooth_kernel % 2 == 0:
            self.region_smooth_kernel += 1
        self.region_low_strength = max(0.0, float(region_low_strength))
        self.region_mid_strength = max(0.0, float(region_mid_strength))
        self.region_high_strength = max(0.0, float(region_high_strength))
        self.transport_gamma = max(0.0, float(transport_gamma))
        self.transport_floor = max(0.0, min(1.0, float(transport_floor)))
        self.transport_power = max(1e-4, float(transport_power))
        self.transport_use_entropy = bool(transport_use_entropy)
        self.transport_use_uniqueness = bool(transport_use_uniqueness)
        self.transport_low_use_support = bool(transport_low_use_support)
        self.transport_low_strength = max(0.0, float(transport_low_strength))
        self.transport_mid_strength = max(0.0, float(transport_mid_strength))
        self.transport_high_strength = max(0.0, float(transport_high_strength))
        self.adain_moment_kernel = max(1, int(adain_moment_kernel))
        if self.adain_moment_kernel % 2 == 0:
            self.adain_moment_kernel += 1
        self.adain_eps = max(1e-8, float(adain_eps))
        self.amp_gamma = max(0.0, float(amp_gamma))
        self.amp_floor = max(0.0, min(1.0, float(amp_floor)))
        self.amp_low_strength = max(0.0, float(amp_low_strength))
        self.amp_mid_strength = max(0.0, float(amp_mid_strength))
        self.amp_high_strength = max(0.0, float(amp_high_strength))
        self.texton_tanh_scale = max(1e-4, float(texton_tanh_scale))
        self.texton_low_strength = max(0.0, float(texton_low_strength))
        self.texton_mid_strength = max(0.0, float(texton_mid_strength))
        self.texton_high_strength = max(0.0, float(texton_high_strength))
        self.token_flatten_strength = max(0.0, float(token_flatten_strength))
        self.token_flatten_kernel = max(1, int(token_flatten_kernel))
        if self.token_flatten_kernel % 2 == 0:
            self.token_flatten_kernel += 1
        self.token_adain_gate_enable = bool(token_adain_gate_enable)
        self.token_reader_scale = max(0.0, float(token_reader_scale))
        self.token_reader: nn.Module | None = None
        if bool(token_reader_enable):
            token_dim = max(1, int(token_identity_dim)) + max(1, int(token_grammar_dim)) + max(1, int(token_band_dim))
            hidden_dim = max(4, int(token_reader_hidden))
            self.token_reader = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )
            nn.init.zeros_(self.token_reader[-1].weight)
            nn.init.zeros_(self.token_reader[-1].bias)
        self.token_grammar_texture_enable = bool(token_grammar_texture_enable)
        self.token_grammar_texture_scale = max(0.0, float(token_grammar_texture_scale))
        self.token_texton_carrier_enable = bool(token_texton_carrier_enable)
        self.token_texton_carrier_strength = max(0.0, float(token_texton_carrier_strength))
        self.token_texton_carrier_tanh_scale = max(1e-4, float(token_texton_carrier_tanh_scale))
        self.token_texton_carrier_mapper: nn.Module | None = None
        if self.token_texton_carrier_enable and self.token_texton_carrier_strength > 0.0:
            carrier_hidden = max(dim, int(round(float(dim) * max(0.25, float(token_texton_carrier_hidden_mult)))))
            carrier_groups = _resolve_group_count(carrier_hidden, num_groups)
            self.token_texton_carrier_mapper = nn.Sequential(
                nn.Conv2d(dim * 3, carrier_hidden, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(carrier_groups, carrier_hidden),
                nn.SiLU(),
                nn.Conv2d(carrier_hidden, dim, kernel_size=3, stride=1, padding=1),
            )
            nn.init.zeros_(self.token_texton_carrier_mapper[-1].weight)
            nn.init.zeros_(self.token_texton_carrier_mapper[-1].bias)
        self.token_prototype_carrier_enable = bool(token_prototype_carrier_enable)
        self.token_prototype_carrier_strength = max(0.0, float(token_prototype_carrier_strength))
        self.token_prototype_carrier_tanh_scale = max(1e-4, float(token_prototype_carrier_tanh_scale))
        self.token_prototype_carrier_mapper: nn.Module | None = None
        if self.token_prototype_carrier_enable and self.token_prototype_carrier_strength > 0.0:
            proto_hidden = max(dim, int(round(float(dim) * max(0.25, float(token_prototype_carrier_hidden_mult)))))
            proto_groups = _resolve_group_count(proto_hidden, num_groups)
            self.token_prototype_carrier_mapper = nn.Sequential(
                nn.Conv2d(dim * 3, proto_hidden, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(proto_groups, proto_hidden),
                nn.SiLU(),
                nn.Conv2d(proto_hidden, dim, kernel_size=3, stride=1, padding=1),
            )
            nn.init.zeros_(self.token_prototype_carrier_mapper[-1].weight)
            nn.init.zeros_(self.token_prototype_carrier_mapper[-1].bias)
        self.token_depthwise_filter_enable = bool(token_depthwise_filter_enable)
        self.token_depthwise_filter_strength = max(0.0, float(token_depthwise_filter_strength))
        self.token_depthwise_filter_tanh_scale = max(1e-4, float(token_depthwise_filter_tanh_scale))
        self.token_depthwise_filter_basis_offset = max(0, int(token_depthwise_filter_basis_offset))
        self.token_depthwise_filter_learnable_gate_scale = max(0.0, float(token_depthwise_filter_learnable_gate_scale))
        self.token_depthwise_filter_style_basis_gate_scale = max(0.0, float(token_depthwise_filter_style_basis_gate_scale))
        self.token_depthwise_filter_style_basis_delta_scale = max(0.0, float(token_depthwise_filter_style_basis_delta_scale))
        self.num_styles = max(1, int(num_styles))
        depthwise_basis = self._build_token_depthwise_filter_basis()
        depthwise_basis_count = int(depthwise_basis.shape[0])
        if bool(token_depthwise_filter_learnable_gate):
            self.token_depthwise_filter_gate_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        else:
            self.register_parameter("token_depthwise_filter_gate_logits", None)
        if bool(token_depthwise_filter_style_basis_gate):
            self.token_depthwise_filter_style_basis_gate_logits = nn.Parameter(
                torch.zeros(self.num_styles, depthwise_basis_count, dtype=torch.float32)
            )
        else:
            self.register_parameter("token_depthwise_filter_style_basis_gate_logits", None)
        if bool(token_depthwise_filter_style_basis_delta):
            self.token_depthwise_filter_style_basis_delta = nn.Parameter(
                torch.zeros(self.num_styles, depthwise_basis_count, 3, 3, dtype=torch.float32)
            )
        else:
            self.register_parameter("token_depthwise_filter_style_basis_delta", None)
        self.register_buffer(
            "_token_depthwise_filter_basis",
            depthwise_basis,
            persistent=False,
        )
        self.texton_style_generator: nn.Module | None = None
        self.texton_band_allocator: nn.Module | None = None
        self.texton_mapper: nn.Module | None = None
        if self.mode == "transport_texton":
            texton_hidden = max(dim, int(round(float(dim) * max(0.25, float(texton_hidden_mult)))))
            texton_groups = _resolve_group_count(texton_hidden, num_groups)
            self.texton_mapper = nn.Sequential(
                nn.Conv2d(dim * 3, texton_hidden, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(texton_groups, texton_hidden),
                nn.SiLU(),
                nn.Conv2d(texton_hidden, dim, kernel_size=3, stride=1, padding=1),
            )
            nn.init.normal_(self.texton_mapper[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.texton_mapper[-1].bias)
        self.last_debug: dict[str, torch.Tensor] = {}

    @staticmethod
    def _build_token_depthwise_filter_basis() -> torch.Tensor:
        kernels = [
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            [[-2.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
            [[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]],
            [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        ]
        basis = torch.tensor(kernels, dtype=torch.float32).view(len(kernels), 1, 3, 3)
        basis = basis - basis.mean(dim=(2, 3), keepdim=True)
        denom = basis.abs().sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return basis / denom

    @staticmethod
    def _lowpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
        kernel = max(1, int(kernel))
        if kernel <= 1:
            return x
        if kernel % 2 == 0:
            kernel += 1
        pad = kernel // 2
        return F.avg_pool2d(F.pad(x.float(), (pad, pad, pad, pad), mode="reflect"), kernel_size=kernel, stride=1).to(dtype=x.dtype)

    def _content_support_gate(self, content_feat: torch.Tensor) -> torch.Tensor:
        if self.band_gate_gamma <= 0.0:
            return content_feat.new_ones(content_feat.shape[0], 1, content_feat.shape[2], content_feat.shape[3])
        content_band = content_feat - self._lowpass(content_feat, self.band_gate_kernel)
        energy = content_band.float().abs().mean(dim=1, keepdim=True)
        mean = energy.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        normalized = energy / mean
        gate = torch.sigmoid((normalized - 1.0) * self.band_gate_gamma)
        if self.band_gate_floor > 0.0:
            gate = self.band_gate_floor + (1.0 - self.band_gate_floor) * gate
        return gate.to(dtype=content_feat.dtype)

    def _local_mean_std(self, x: torch.Tensor, kernel: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_f = x.float()
        kernel = max(1, int(kernel))
        if kernel % 2 == 0:
            kernel += 1
        if kernel <= 1:
            mean = x_f.mean(dim=(2, 3), keepdim=True)
            var = (x_f - mean).square().mean(dim=(2, 3), keepdim=True)
            return mean, torch.sqrt(var.clamp_min(0.0) + self.adain_eps)

        pad = kernel // 2
        if x_f.shape[-2] > pad and x_f.shape[-1] > pad:
            padded = F.pad(x_f, (pad, pad, pad, pad), mode="reflect")
            padded_sq = F.pad(x_f.square(), (pad, pad, pad, pad), mode="reflect")
            mean = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
            mean_sq = F.avg_pool2d(padded_sq, kernel_size=kernel, stride=1)
        else:
            mean = x_f.mean(dim=(2, 3), keepdim=True).expand_as(x_f)
            mean_sq = x_f.square().mean(dim=(2, 3), keepdim=True).expand_as(x_f)
        var = (mean_sq - mean.square()).clamp_min(0.0)
        return mean, torch.sqrt(var + self.adain_eps)

    def _phase_gate(self, content_feat: torch.Tensor, residual_feat: torch.Tensor) -> torch.Tensor:
        if self.dual_phase_gamma <= 0.0:
            return content_feat.new_ones(content_feat.shape[0], 1, content_feat.shape[2], content_feat.shape[3])
        content_band = content_feat - self._lowpass(content_feat, self.dual_mid_outer_kernel)
        content_f = content_band.float()
        residual_f = residual_feat.float()
        numerator = (content_f * residual_f).mean(dim=1, keepdim=True)
        denom = torch.sqrt(
            content_f.square().mean(dim=1, keepdim=True)
            * residual_f.square().mean(dim=1, keepdim=True)
            + 1e-8
        )
        cosine = (numerator / denom).clamp(-1.0, 1.0)
        gate = torch.sigmoid(cosine * self.dual_phase_gamma)
        if self.dual_phase_floor > 0.0:
            gate = self.dual_phase_floor + (1.0 - self.dual_phase_floor) * gate
        return gate.to(dtype=content_feat.dtype)

    def _style_region_gate(self, content_feat: torch.Tensor) -> torch.Tensor:
        b = content_feat.shape[0]
        region = content_feat.float().mean(dim=1, keepdim=True)
        if self.region_smooth_kernel > 1:
            region = F.avg_pool2d(
                region,
                kernel_size=self.region_smooth_kernel,
                stride=1,
                padding=self.region_smooth_kernel // 2,
            )
        mean = region.flatten(1).mean(dim=1).view(b, 1, 1, 1)
        std = region.flatten(1).std(dim=1, unbiased=False).view(b, 1, 1, 1).clamp_min(1e-6)
        region = (region - mean) / std
        centers = torch.linspace(
            -1.25,
            1.25,
            self.region_bins,
            device=content_feat.device,
            dtype=torch.float32,
        ).view(1, self.region_bins, 1, 1)
        assignment = torch.softmax(-self.region_gamma * (region - centers).square(), dim=1)
        weights = content_feat.new_ones(b, self.region_bins, 1, 1)
        gate = (assignment * weights).sum(dim=1, keepdim=True)
        if self.region_floor > 0.0:
            gate = self.region_floor + (1.0 - self.region_floor) * gate
        return gate.to(device=content_feat.device, dtype=content_feat.dtype)

    def _style_amplitude_gate(self, residual_feat: torch.Tensor) -> torch.Tensor:
        if self.amp_gamma <= 0.0:
            return residual_feat.new_ones(residual_feat.shape[0], 1, residual_feat.shape[2], residual_feat.shape[3])
        envelope = residual_feat.float().abs().mean(dim=1, keepdim=True)
        if self.adain_moment_kernel > 1:
            envelope = self._lowpass(envelope, self.adain_moment_kernel).float()
        denom = envelope.flatten(1).mean(dim=1).view(residual_feat.shape[0], 1, 1, 1).clamp_min(1e-6)
        normalized = envelope / denom
        gate = torch.sigmoid((normalized - 1.0) * self.amp_gamma)
        if self.amp_floor > 0.0:
            gate = self.amp_floor + (1.0 - self.amp_floor) * gate
        return gate.to(device=residual_feat.device, dtype=residual_feat.dtype)

    def _style_texton_seed(
        self,
        content_feat: torch.Tensor,
        residual_feat: torch.Tensor,
    ) -> torch.Tensor:
        if self.texton_mapper is None:
            return residual_feat
        b, c, h_dim, w_dim = content_feat.shape
        content_band = content_feat - self._lowpass(content_feat, self.dual_mid_outer_kernel)
        carrier = self.texton_mapper(
            torch.cat(
                [
                    self.content_norm(content_feat),
                    residual_feat.to(dtype=content_feat.dtype),
                    content_band,
                ],
                dim=1,
            )
        )
        return residual_feat + carrier.to(device=content_feat.device, dtype=content_feat.dtype)

    def _style_texton_band_allocation(
        self,
        content_feat: torch.Tensor,
        style_tokens: object | None = None,
    ) -> torch.Tensor:
        b = content_feat.shape[0]
        gains = content_feat.new_ones(b, 3, 1, 1)
        token_gains = getattr(style_tokens, "band_gains", None)
        if torch.is_tensor(token_gains):
            if token_gains.ndim == 2:
                token_gains = token_gains[:, :3].view(token_gains.shape[0], 3, 1, 1)
            if token_gains.shape[0] == 1 and b > 1:
                token_gains = token_gains.expand(b, -1, -1, -1)
            elif token_gains.shape[0] != b:
                raise ValueError(f"style token batch mismatch: expected {b} or 1, got {token_gains.shape[0]}")
            gains = gains * token_gains.to(device=content_feat.device, dtype=content_feat.dtype)
        if self.token_reader is not None and style_tokens is not None and self.token_reader_scale > 0.0:
            token_input = self._style_token_reader_input(style_tokens, b, content_feat.device)
            logits = self.token_reader(token_input).view(b, 3, 1, 1)
            gains = gains * (1.0 + torch.tanh(logits) * self.token_reader_scale)
        return gains.to(device=content_feat.device, dtype=content_feat.dtype)

    @staticmethod
    def _match_token_field(field: torch.Tensor, batch: int, device: torch.device, name: str) -> torch.Tensor:
        field = field.to(device=device, dtype=torch.float32)
        if field.ndim == 1:
            field = field.unsqueeze(0)
        if field.shape[0] == 1 and batch > 1:
            field = field.expand(batch, -1)
        elif field.shape[0] != batch:
            raise ValueError(f"{name} batch mismatch: expected {batch} or 1, got {field.shape[0]}")
        return field

    def _style_token_reader_input(self, style_tokens: object, batch: int, device: torch.device) -> torch.Tensor:
        identity = getattr(style_tokens, "identity", None)
        grammar = getattr(style_tokens, "grammar", None)
        band = getattr(style_tokens, "band_logits", None)
        if not (torch.is_tensor(identity) and torch.is_tensor(grammar) and torch.is_tensor(band)):
            raise ValueError("token_reader requires identity, grammar, and band_logits fields")
        return torch.cat(
            [
                self._match_token_field(identity, batch, device, "identity"),
                self._match_token_field(grammar, batch, device, "grammar"),
                self._match_token_field(band, batch, device, "band_logits"),
            ],
            dim=1,
        )

    def _style_token_grammar_texture_alloc(
        self,
        style_tokens: object | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one = torch.ones(batch, 1, 1, 1, device=device, dtype=dtype)
        if not self.token_grammar_texture_enable or self.token_grammar_texture_scale <= 0.0 or style_tokens is None:
            return one, one
        grammar = getattr(style_tokens, "grammar", None)
        if not torch.is_tensor(grammar) or grammar.numel() == 0:
            return one, one
        grammar = self._match_token_field(grammar, batch, device, "grammar")
        mid = grammar[:, 5:6] if grammar.shape[1] > 5 else grammar.new_zeros(batch, 1)
        high = grammar[:, 6:7] if grammar.shape[1] > 6 else grammar.new_zeros(batch, 1)
        scale = float(self.token_grammar_texture_scale)
        mid_alloc = 1.0 + torch.tanh(mid).view(batch, 1, 1, 1) * scale
        high_alloc = 1.0 + torch.tanh(high).view(batch, 1, 1, 1) * scale
        return mid_alloc.to(device=device, dtype=dtype), high_alloc.to(device=device, dtype=dtype)

    def _style_token_texton_carrier_delta(
        self,
        content_feat: torch.Tensor,
        residual_feat: torch.Tensor,
        style_tokens: object | None,
        detail_gate: torch.Tensor,
        mid_alloc: torch.Tensor,
        high_alloc: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.token_texton_carrier_mapper is None
            or self.token_texton_carrier_strength <= 0.0
            or style_tokens is None
        ):
            return content_feat.new_zeros(content_feat.shape)
        grammar = getattr(style_tokens, "grammar", None)
        if not torch.is_tensor(grammar) or grammar.numel() == 0:
            return content_feat.new_zeros(content_feat.shape)
        b = content_feat.shape[0]
        grammar = self._match_token_field(grammar, b, content_feat.device, "grammar")
        g_mid = torch.tanh(grammar[:, 5:6]).view(b, 1, 1, 1) if grammar.shape[1] > 5 else grammar.new_zeros(b, 1, 1, 1)
        g_high = torch.tanh(grammar[:, 6:7]).view(b, 1, 1, 1) if grammar.shape[1] > 6 else grammar.new_zeros(b, 1, 1, 1)
        residual_f = residual_feat.float()
        inner = self._lowpass(residual_feat, self.dual_mid_inner_kernel).float()
        outer = self._lowpass(residual_feat, self.dual_mid_outer_kernel).float()
        residual_mid = inner - outer
        residual_high = residual_f - inner
        content_high = content_feat.float() - self._lowpass(content_feat, self.dual_mid_inner_kernel).float()
        token_seed = residual_mid * (1.0 + g_mid) + residual_high * (1.0 + g_high)
        carrier = self.token_texton_carrier_mapper(
            torch.cat(
                [
                    self.content_norm(content_feat).float(),
                    content_high,
                    token_seed,
                ],
                dim=1,
            ).to(dtype=content_feat.dtype)
        )
        carrier_f = carrier.float()
        carrier_inner = self._lowpass(carrier, self.dual_mid_inner_kernel).float()
        carrier_outer = self._lowpass(carrier, self.dual_mid_outer_kernel).float()
        carrier_mid = carrier_inner - carrier_outer
        carrier_high = carrier_f - carrier_inner
        scale = self.token_texton_carrier_tanh_scale
        mid_add = torch.tanh(carrier_mid / scale) * scale * detail_gate.float() * mid_alloc.float()
        high_add = torch.tanh(carrier_high / scale) * scale * detail_gate.float() * high_alloc.float()
        add = (mid_add * self.texton_mid_strength + high_add * self.texton_high_strength) * self.token_texton_carrier_strength
        return add.to(device=content_feat.device, dtype=content_feat.dtype)

    def _style_token_prototype_carrier_delta(
        self,
        content_feat: torch.Tensor,
        style_feat: torch.Tensor,
        style_tokens: object | None,
        detail_gate: torch.Tensor,
        mid_alloc: torch.Tensor,
        high_alloc: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.token_prototype_carrier_mapper is None
            or self.token_prototype_carrier_strength <= 0.0
            or style_tokens is None
        ):
            return content_feat.new_zeros(content_feat.shape)
        grammar = getattr(style_tokens, "grammar", None)
        if not torch.is_tensor(grammar) or grammar.numel() == 0:
            return content_feat.new_zeros(content_feat.shape)
        b = content_feat.shape[0]
        grammar = self._match_token_field(grammar, b, content_feat.device, "grammar")
        g_mid = torch.tanh(grammar[:, 5:6]).view(b, 1, 1, 1) if grammar.shape[1] > 5 else grammar.new_zeros(b, 1, 1, 1)
        g_high = torch.tanh(grammar[:, 6:7]).view(b, 1, 1, 1) if grammar.shape[1] > 6 else grammar.new_zeros(b, 1, 1, 1)

        style_inner = self._lowpass(style_feat, self.dual_mid_inner_kernel).float()
        style_outer = self._lowpass(style_feat, self.dual_mid_outer_kernel).float()
        style_mid = style_inner - style_outer
        style_high = style_feat.float() - style_inner
        content_high = content_feat.float() - self._lowpass(content_feat, self.dual_mid_inner_kernel).float()
        proto_seed = style_mid * (1.0 + g_mid) + style_high * (1.0 + g_high)
        carrier = self.token_prototype_carrier_mapper(
            torch.cat(
                [
                    self.content_norm(content_feat).float(),
                    proto_seed,
                    content_high,
                ],
                dim=1,
            ).to(dtype=content_feat.dtype)
        )
        carrier_f = carrier.float()
        carrier_inner = self._lowpass(carrier, self.dual_mid_inner_kernel).float()
        carrier_outer = self._lowpass(carrier, self.dual_mid_outer_kernel).float()
        carrier_mid = carrier_inner - carrier_outer
        carrier_high = carrier_f - carrier_inner
        scale = self.token_prototype_carrier_tanh_scale
        mid_add = torch.tanh(carrier_mid / scale) * scale * detail_gate.float() * mid_alloc.float()
        high_add = torch.tanh(carrier_high / scale) * scale * detail_gate.float() * high_alloc.float()
        add = (mid_add * self.texton_mid_strength + high_add * self.texton_high_strength) * self.token_prototype_carrier_strength
        return add.to(device=content_feat.device, dtype=content_feat.dtype)

    def _style_token_flatten_delta(
        self,
        content_feat: torch.Tensor,
        style_tokens: object | None,
        where_gate: torch.Tensor,
        support_gate: torch.Tensor,
    ) -> torch.Tensor:
        if self.token_flatten_strength <= 0.0 or style_tokens is None:
            return content_feat.new_zeros(content_feat.shape)
        grammar = getattr(style_tokens, "grammar", None)
        if not torch.is_tensor(grammar) or grammar.numel() == 0:
            return content_feat.new_zeros(content_feat.shape)
        b = content_feat.shape[0]
        if grammar.ndim == 1:
            grammar = grammar.unsqueeze(0)
        if grammar.shape[0] == 1 and b > 1:
            grammar = grammar.expand(b, -1)
        elif grammar.shape[0] != b:
            raise ValueError(f"style grammar batch mismatch: expected {b} or 1, got {grammar.shape[0]}")
        flatness = torch.tanh(grammar[:, 1:2].float()) if grammar.shape[1] > 1 else grammar.new_zeros(b, 1)
        suppress = torch.tanh(grammar[:, 7:8].float()) if grammar.shape[1] > 7 else grammar.new_zeros(b, 1)
        token_strength = ((flatness + suppress) * 0.5).view(b, 1, 1, 1)
        high = content_feat.float() - self._lowpass(content_feat, self.token_flatten_kernel).float()
        smooth_region = (1.0 - support_gate.float()).clamp_min(0.0)
        delta = -high * where_gate.float() * smooth_region * token_strength * self.token_flatten_strength
        return delta.to(device=content_feat.device, dtype=content_feat.dtype)

    def _style_token_depthwise_filter_delta(
        self,
        residual_feat: torch.Tensor,
        style_tokens: object | None,
        detail_gate: torch.Tensor,
        mid_alloc: torch.Tensor,
        high_alloc: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.token_depthwise_filter_enable
            or self.token_depthwise_filter_strength <= 0.0
            or style_tokens is None
        ):
            return residual_feat.new_zeros(residual_feat.shape)
        grammar = getattr(style_tokens, "grammar", None)
        if not torch.is_tensor(grammar) or grammar.numel() == 0:
            return residual_feat.new_zeros(residual_feat.shape)
        b, c, h_dim, w_dim = residual_feat.shape
        grammar = self._match_token_field(grammar, b, residual_feat.device, "grammar")
        basis = self._token_depthwise_filter_basis.to(device=residual_feat.device, dtype=torch.float32)
        basis_count = int(basis.shape[0])
        offset = int(self.token_depthwise_filter_basis_offset)
        coeff = grammar.new_zeros(b, basis_count)
        if grammar.shape[1] > offset:
            take = min(basis_count, grammar.shape[1] - offset)
            coeff[:, :take] = grammar[:, offset : offset + take]
        coeff = torch.tanh(coeff.float())
        style_basis_gate = self.token_depthwise_filter_style_basis_gate_logits
        style_id = getattr(style_tokens, "style_id", None)
        if (
            style_basis_gate is not None
            and self.token_depthwise_filter_style_basis_gate_scale > 0.0
            and torch.is_tensor(style_id)
        ):
            style_id = style_id.to(device=residual_feat.device, dtype=torch.long).view(-1)
            if style_id.shape[0] == 1 and b > 1:
                style_id = style_id.expand(b)
            elif style_id.shape[0] != b:
                style_id = style_id[:1].expand(b)
            style_id = style_id.clamp_min(0).clamp_max(self.num_styles - 1)
            basis_gate = style_basis_gate.index_select(0, style_id).float()
            coeff = coeff * (1.0 + torch.tanh(basis_gate) * self.token_depthwise_filter_style_basis_gate_scale)
        basis_for_sample = basis.unsqueeze(0).expand(b, -1, -1, -1, -1)
        basis_delta = self.token_depthwise_filter_style_basis_delta
        if (
            basis_delta is not None
            and self.token_depthwise_filter_style_basis_delta_scale > 0.0
            and torch.is_tensor(style_id)
        ):
            style_id_delta = style_id.to(device=residual_feat.device, dtype=torch.long).view(-1)
            if style_id_delta.shape[0] == 1 and b > 1:
                style_id_delta = style_id_delta.expand(b)
            elif style_id_delta.shape[0] != b:
                style_id_delta = style_id_delta[:1].expand(b)
            style_id_delta = style_id_delta.clamp_min(0).clamp_max(self.num_styles - 1)
            delta = basis_delta.index_select(0, style_id_delta).float()
            delta = delta - delta.mean(dim=(-2, -1), keepdim=True)
            delta = delta / delta.flatten(2).norm(dim=-1, keepdim=True).view(b, basis_count, 1, 1).clamp_min(1.0)
            delta = delta.unsqueeze(2)
            basis_for_sample = basis_for_sample + torch.tanh(delta) * self.token_depthwise_filter_style_basis_delta_scale
            basis_for_sample = basis_for_sample - basis_for_sample.mean(dim=(-2, -1), keepdim=True)
        kernels = (coeff.view(b, basis_count, 1, 1, 1) * basis_for_sample).sum(dim=1, keepdim=True)

        residual_f = residual_feat.float()
        inner = self._lowpass(residual_feat, self.dual_mid_inner_kernel).float()
        outer = self._lowpass(residual_feat, self.dual_mid_outer_kernel).float()
        residual_mid = inner - outer
        residual_high = residual_f - inner
        gate_logits = self.token_depthwise_filter_gate_logits
        if gate_logits is not None and self.token_depthwise_filter_learnable_gate_scale > 0.0:
            gate = 1.0 + torch.tanh(gate_logits.float()).view(1, 2, 1, 1) * self.token_depthwise_filter_learnable_gate_scale
            residual_mid = residual_mid * gate[:, 0:1]
            residual_high = residual_high * gate[:, 1:2]
        source = residual_mid * mid_alloc.float() + residual_high * high_alloc.float()
        x = source.view(1, b * c, h_dim, w_dim)
        weight = kernels.view(b, 1, 1, 3, 3).expand(b, c, 1, 3, 3).reshape(b * c, 1, 3, 3)
        filtered = F.conv2d(x, weight, padding=1, groups=b * c).view(b, c, h_dim, w_dim)
        scale = self.token_depthwise_filter_tanh_scale
        delta = (
            torch.tanh(filtered / scale)
            * scale
            * detail_gate.float()
            * self.token_depthwise_filter_strength
        )
        return delta.to(device=residual_feat.device, dtype=residual_feat.dtype)

    def _transport_confidence_gate(
        self,
        content_feat: torch.Tensor,
        semantic_attn: torch.Tensor | None,
    ) -> torch.Tensor:
        if semantic_attn is None or self.transport_gamma <= 0.0:
            return content_feat.new_ones(content_feat.shape[0], 1, content_feat.shape[2], content_feat.shape[3])
        attn = semantic_attn.float()
        b, _, h_dim, w_dim = content_feat.shape
        tokens = h_dim * w_dim
        if attn.ndim != 3 or attn.shape[0] != b or attn.shape[1] != tokens:
            return content_feat.new_ones(b, 1, h_dim, w_dim)

        topk = torch.topk(attn, k=min(2, attn.shape[-1]), dim=-1).values
        top1 = topk[..., 0]
        if topk.shape[-1] > 1:
            margin = (top1 - topk[..., 1]).clamp_min(0.0)
        else:
            margin = top1.clamp_min(0.0)
        confidence = margin

        if self.transport_use_entropy and attn.shape[-1] > 1:
            entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1)
            entropy = entropy / math.log(float(attn.shape[-1]))
            confidence = confidence * (1.0 - entropy).clamp(0.0, 1.0)

        if self.transport_use_uniqueness:
            key_load = attn.sum(dim=1).clamp_min(1e-6)
            query_load = torch.bmm(attn, key_load.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
            confidence = confidence / torch.sqrt(query_load)

        if self.transport_power != 1.0:
            confidence = confidence.clamp_min(0.0).pow(self.transport_power)
        confidence = confidence.view(b, 1, h_dim, w_dim)
        denom = confidence.flatten(1).mean(dim=1).view(b, 1, 1, 1).clamp_min(1e-6)
        gate = 1.0 - torch.exp(-self.transport_gamma * confidence / denom)
        if self.transport_floor > 0.0:
            gate = self.transport_floor + (1.0 - self.transport_floor) * gate
        return gate.to(device=content_feat.device, dtype=content_feat.dtype)

    def forward(
        self,
        content_feat: torch.Tensor,
        style_feat: torch.Tensor,
        semantic_attn: torch.Tensor | None = None,
        style_tokens: object | None = None,
    ) -> torch.Tensor:
        self.last_debug = {}
        blend_ratio = torch.sigmoid(self.alpha).to(device=content_feat.device, dtype=content_feat.dtype)
        self.last_debug["body_blend_ratio"] = blend_ratio.view(1, 1, 1, 1).detach()
        blended = torch.lerp(content_feat, style_feat, blend_ratio)
        if self.mode == "modulate":
            content_norm = self.content_norm(content_feat)
            style_delta = blended - content_feat
            params = self.mod_mapper(torch.cat([content_norm, style_delta], dim=1))
            gamma, beta = params.chunk(2, dim=1)
            scale = self.mod_tanh_scale
            gamma = torch.tanh(gamma / scale) * scale
            beta = torch.tanh(beta / scale) * scale
            modulated = content_norm * (1.0 + gamma) + beta
            delta = modulated - content_norm
            self.last_debug["body_mod_delta"] = delta.detach()
            return content_feat + delta * self.mod_strength
        if self.mode == "residual_band":
            residual = (style_feat - content_feat) * blend_ratio
            residual_band = residual - self._lowpass(residual, self.band_outer_kernel)
            gate = self._content_support_gate(content_feat)
            scale = self.band_tanh_scale
            bounded = torch.tanh(residual_band / scale) * scale
            self.last_debug["body_band_gate"] = gate.detach()
            self.last_debug["body_band_delta"] = (bounded * gate * self.band_strength).detach()
            return content_feat + bounded * gate * self.band_strength
        if self.mode == "residual_dual":
            residual = (style_feat - content_feat) * blend_ratio
            scale = self.band_tanh_scale
            low = self._lowpass(residual, self.dual_low_kernel)
            inner = self._lowpass(residual, self.dual_mid_inner_kernel)
            outer = self._lowpass(residual, self.dual_mid_outer_kernel)
            mid = inner - outer
            high = residual - inner
            support_gate = self._content_support_gate(content_feat)
            phase_gate = self._phase_gate(content_feat, mid + high)
            detail_gate = support_gate * phase_gate
            low_add = torch.tanh(low / scale) * scale * self.dual_low_strength
            mid_add = torch.tanh(mid / scale) * scale * detail_gate * self.dual_mid_strength
            high_add = torch.tanh(high / scale) * scale * detail_gate * self.dual_high_strength
            self.last_debug["body_dual_support_gate"] = support_gate.detach()
            self.last_debug["body_dual_phase_gate"] = phase_gate.detach()
            self.last_debug["body_dual_low_delta"] = low_add.detach()
            self.last_debug["body_dual_mid_delta"] = mid_add.detach()
            self.last_debug["body_dual_high_delta"] = high_add.detach()
            return content_feat + low_add + mid_add + high_add
        if self.mode == "region_paint":
            remapped = self.conv(self.norm(blended))
            residual = remapped - content_feat
            scale = self.band_tanh_scale
            low = self._lowpass(residual, self.dual_low_kernel)
            inner = self._lowpass(residual, self.dual_mid_inner_kernel)
            outer = self._lowpass(residual, self.dual_mid_outer_kernel)
            mid = inner - outer
            high = residual - inner
            region_gate = self._style_region_gate(content_feat)
            support_gate = self._content_support_gate(content_feat)
            phase_gate = self._phase_gate(content_feat, mid + high)
            detail_gate = region_gate * support_gate * phase_gate
            low_add = torch.tanh(low / scale) * scale * region_gate * self.region_low_strength
            mid_add = torch.tanh(mid / scale) * scale * detail_gate * self.region_mid_strength
            high_add = torch.tanh(high / scale) * scale * detail_gate * self.region_high_strength
            self.last_debug["body_region_gate"] = region_gate.detach()
            self.last_debug["body_region_support_gate"] = support_gate.detach()
            self.last_debug["body_region_phase_gate"] = phase_gate.detach()
            self.last_debug["body_region_low_delta"] = low_add.detach()
            self.last_debug["body_region_mid_delta"] = mid_add.detach()
            self.last_debug["body_region_high_delta"] = high_add.detach()
            return content_feat + low_add + mid_add + high_add
        if self.mode == "transport_paint":
            remapped = self.conv(self.norm(blended))
            residual = remapped - content_feat
            scale = self.band_tanh_scale
            low = self._lowpass(residual, self.dual_low_kernel)
            inner = self._lowpass(residual, self.dual_mid_inner_kernel)
            outer = self._lowpass(residual, self.dual_mid_outer_kernel)
            mid = inner - outer
            high = residual - inner
            transport_gate = self._transport_confidence_gate(content_feat, semantic_attn)
            support_gate = self._content_support_gate(content_feat)
            phase_gate = self._phase_gate(content_feat, mid + high)
            low_gate = transport_gate * support_gate if self.transport_low_use_support else transport_gate
            detail_gate = transport_gate * support_gate * phase_gate
            low_add = torch.tanh(low / scale) * scale * low_gate * self.transport_low_strength
            mid_add = torch.tanh(mid / scale) * scale * detail_gate * self.transport_mid_strength
            high_add = torch.tanh(high / scale) * scale * detail_gate * self.transport_high_strength
            self.last_debug["body_transport_gate"] = transport_gate.detach()
            self.last_debug["body_transport_support_gate"] = support_gate.detach()
            self.last_debug["body_transport_phase_gate"] = phase_gate.detach()
            self.last_debug["body_transport_low_gate"] = low_gate.detach()
            self.last_debug["body_transport_low_delta"] = low_add.detach()
            self.last_debug["body_transport_mid_delta"] = mid_add.detach()
            self.last_debug["body_transport_high_delta"] = high_add.detach()
            return content_feat + low_add + mid_add + high_add
        if self.mode == "transport_adain":
            content_mean, content_std = self._local_mean_std(content_feat, self.adain_moment_kernel)
            style_mean, style_std = self._local_mean_std(style_feat, self.adain_moment_kernel)
            normalized = (content_feat.float() - content_mean) / content_std
            target = normalized * style_std + style_mean
            residual = target.to(device=content_feat.device, dtype=content_feat.dtype) - content_feat
            scale = self.band_tanh_scale
            low = self._lowpass(residual, self.dual_low_kernel)
            inner = self._lowpass(residual, self.dual_mid_inner_kernel)
            outer = self._lowpass(residual, self.dual_mid_outer_kernel)
            mid = inner - outer
            high = residual - inner
            transport_gate = self._transport_confidence_gate(content_feat, semantic_attn)
            support_gate = self._content_support_gate(content_feat)
            phase_gate = self._phase_gate(content_feat, mid + high)
            low_gate = transport_gate * support_gate if self.transport_low_use_support else transport_gate
            detail_gate = transport_gate * support_gate * phase_gate
            if self.token_adain_gate_enable and style_tokens is not None:
                band_alloc = self._style_texton_band_allocation(content_feat, style_tokens)
                low_alloc = band_alloc[:, 0:1]
                mid_alloc = band_alloc[:, 1:2]
                high_alloc = band_alloc[:, 2:3]
                grammar_mid_alloc, grammar_high_alloc = self._style_token_grammar_texture_alloc(
                    style_tokens,
                    content_feat.shape[0],
                    content_feat.device,
                    content_feat.dtype,
                )
                mid_alloc = mid_alloc * grammar_mid_alloc
                high_alloc = high_alloc * grammar_high_alloc
            else:
                band_alloc = content_feat.new_ones(content_feat.shape[0], 3, 1, 1)
                low_alloc = mid_alloc = high_alloc = content_feat.new_ones(content_feat.shape[0], 1, 1, 1)
                grammar_mid_alloc = grammar_high_alloc = content_feat.new_ones(content_feat.shape[0], 1, 1, 1)
            low_add = torch.tanh(low / scale) * scale * low_gate * self.transport_low_strength * low_alloc
            mid_add = torch.tanh(mid / scale) * scale * detail_gate * self.transport_mid_strength * mid_alloc
            high_add = torch.tanh(high / scale) * scale * detail_gate * self.transport_high_strength * high_alloc
            texton_carrier_add = (
                self._style_token_texton_carrier_delta(
                    content_feat,
                    residual,
                    style_tokens,
                    detail_gate,
                    mid_alloc,
                    high_alloc,
                )
                if self.token_adain_gate_enable
                else content_feat.new_zeros(content_feat.shape)
            )
            prototype_carrier_add = (
                self._style_token_prototype_carrier_delta(
                    content_feat,
                    style_feat,
                    style_tokens,
                    detail_gate,
                    mid_alloc,
                    high_alloc,
                )
                if self.token_adain_gate_enable
                else content_feat.new_zeros(content_feat.shape)
            )
            flatten_add = (
                self._style_token_flatten_delta(content_feat, style_tokens, transport_gate, support_gate)
                if self.token_adain_gate_enable
                else content_feat.new_zeros(content_feat.shape)
            )
            depthwise_filter_add = (
                self._style_token_depthwise_filter_delta(
                    residual,
                    style_tokens,
                    detail_gate,
                    mid_alloc,
                    high_alloc,
                )
                if self.token_adain_gate_enable
                else content_feat.new_zeros(content_feat.shape)
            )
            self.last_debug["body_transport_adain_gate"] = transport_gate.detach()
            self.last_debug["body_transport_adain_support_gate"] = support_gate.detach()
            self.last_debug["body_transport_adain_phase_gate"] = phase_gate.detach()
            self.last_debug["body_transport_adain_band_alloc"] = band_alloc.detach()
            self.last_debug["body_transport_adain_grammar_mid_alloc"] = grammar_mid_alloc.detach()
            self.last_debug["body_transport_adain_grammar_high_alloc"] = grammar_high_alloc.detach()
            self.last_debug["body_transport_adain_low_gate"] = low_gate.detach()
            self.last_debug["body_transport_adain_low_delta"] = low_add.detach()
            self.last_debug["body_transport_adain_mid_delta"] = mid_add.detach()
            self.last_debug["body_transport_adain_high_delta"] = high_add.detach()
            self.last_debug["body_transport_adain_token_texton_delta"] = texton_carrier_add.detach()
            self.last_debug["body_transport_adain_token_prototype_delta"] = prototype_carrier_add.detach()
            self.last_debug["body_transport_adain_flatten_delta"] = flatten_add.detach()
            self.last_debug["body_transport_adain_depthwise_filter_delta"] = depthwise_filter_add.detach()
            return (
                content_feat
                + low_add
                + mid_add
                + high_add
                + texton_carrier_add
                + prototype_carrier_add
                + flatten_add
                + depthwise_filter_add
            )
        if self.mode == "transport_amp":
            content_mean, content_std = self._local_mean_std(content_feat, self.adain_moment_kernel)
            style_mean, style_std = self._local_mean_std(style_feat, self.adain_moment_kernel)
            normalized = (content_feat.float() - content_mean) / content_std
            target = normalized * style_std + style_mean
            residual = target.to(device=content_feat.device, dtype=content_feat.dtype) - content_feat
            scale = self.band_tanh_scale
            low = self._lowpass(residual, self.dual_low_kernel)
            inner = self._lowpass(residual, self.dual_mid_inner_kernel)
            outer = self._lowpass(residual, self.dual_mid_outer_kernel)
            mid = inner - outer
            high = residual - inner
            where_gate = self._transport_confidence_gate(content_feat, semantic_attn)
            amp_gate = self._style_amplitude_gate(residual)
            support_gate = self._content_support_gate(content_feat)
            phase_gate = self._phase_gate(content_feat, mid + high)
            low_gate = where_gate * amp_gate
            detail_gate = where_gate * amp_gate * support_gate * phase_gate
            low_add = torch.tanh(low / scale) * scale * low_gate * self.amp_low_strength
            mid_add = torch.tanh(mid / scale) * scale * detail_gate * self.amp_mid_strength
            high_add = torch.tanh(high / scale) * scale * detail_gate * self.amp_high_strength
            self.last_debug["body_transport_amp_where_gate"] = where_gate.detach()
            self.last_debug["body_transport_amp_amp_gate"] = amp_gate.detach()
            self.last_debug["body_transport_amp_support_gate"] = support_gate.detach()
            self.last_debug["body_transport_amp_phase_gate"] = phase_gate.detach()
            self.last_debug["body_transport_amp_low_delta"] = low_add.detach()
            self.last_debug["body_transport_amp_mid_delta"] = mid_add.detach()
            self.last_debug["body_transport_amp_high_delta"] = high_add.detach()
            return content_feat + low_add + mid_add + high_add
        if self.mode == "transport_texton":
            content_mean, content_std = self._local_mean_std(content_feat, self.adain_moment_kernel)
            style_mean, style_std = self._local_mean_std(style_feat, self.adain_moment_kernel)
            normalized = (content_feat.float() - content_mean) / content_std
            target = normalized * style_std + style_mean
            residual = target.to(device=content_feat.device, dtype=content_feat.dtype) - content_feat
            carrier = self._style_texton_seed(content_feat, residual)
            scale = self.texton_tanh_scale
            low = self._lowpass(carrier, self.dual_low_kernel)
            inner = self._lowpass(carrier, self.dual_mid_inner_kernel)
            outer = self._lowpass(carrier, self.dual_mid_outer_kernel)
            mid = inner - outer
            high = carrier - inner
            where_gate = self._transport_confidence_gate(content_feat, semantic_attn)
            amp_gate = self._style_amplitude_gate(carrier)
            support_gate = self._content_support_gate(content_feat)
            phase_gate = self._phase_gate(content_feat, mid + high)
            band_alloc = self._style_texton_band_allocation(content_feat, style_tokens)
            low_alloc = band_alloc[:, 0:1]
            mid_alloc = band_alloc[:, 1:2]
            high_alloc = band_alloc[:, 2:3]
            low_gate = where_gate * amp_gate
            detail_gate = where_gate * amp_gate * support_gate * phase_gate
            low_add = torch.tanh(low / scale) * scale * low_gate * self.texton_low_strength * low_alloc
            mid_add = torch.tanh(mid / scale) * scale * detail_gate * self.texton_mid_strength * mid_alloc
            high_add = torch.tanh(high / scale) * scale * detail_gate * self.texton_high_strength * high_alloc
            flatten_add = self._style_token_flatten_delta(content_feat, style_tokens, where_gate, support_gate)
            self.last_debug["body_transport_texton_where_gate"] = where_gate.detach()
            self.last_debug["body_transport_texton_amp_gate"] = amp_gate.detach()
            self.last_debug["body_transport_texton_support_gate"] = support_gate.detach()
            self.last_debug["body_transport_texton_phase_gate"] = phase_gate.detach()
            self.last_debug["body_transport_texton_band_alloc"] = band_alloc.detach()
            self.last_debug["body_transport_texton_carrier"] = carrier.detach()
            self.last_debug["body_transport_texton_low_delta"] = low_add.detach()
            self.last_debug["body_transport_texton_mid_delta"] = mid_add.detach()
            self.last_debug["body_transport_texton_high_delta"] = high_add.detach()
            self.last_debug["body_transport_texton_flatten_delta"] = flatten_add.detach()
            return content_feat + low_add + mid_add + high_add + flatten_add
        remapped = self.conv(self.norm(blended))
        if not self.residual:
            self.last_debug["body_replace_delta"] = (remapped - content_feat).detach()
            return remapped
        delta = remapped - content_feat
        self.last_debug["body_residual_delta"] = delta.detach()
        return content_feat + torch.tanh(delta / 4.0) * 4.0 * self.residual_strength


class SimpleResBlock(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8) -> None:
        super().__init__()
        groups = _resolve_group_count(dim, num_groups)
        self.norm1 = nn.GroupNorm(groups, dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(groups, dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h



@dataclass
class StyleMaps:
    map_16: torch.Tensor | None = None
    style_id: torch.Tensor | None = None
