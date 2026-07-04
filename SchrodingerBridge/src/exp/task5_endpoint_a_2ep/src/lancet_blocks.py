from __future__ import annotations

from dataclasses import dataclass, field
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


def _route_attention_logits(
    logits: torch.Tensor,
    *,
    routing_mode: str,
    sinkhorn_iters: int = 3,
    gumbel_tau: float = 1.0,
) -> torch.Tensor:
    mode = str(routing_mode).strip().lower()
    if mode == "sinkhorn":
        return _sinkhorn_attention(logits, iters=sinkhorn_iters)
    if mode == "gumbel_hard":
        return _gumbel_hard_attention(logits, tau=gumbel_tau)
    return F.softmax(logits, dim=-1)


class CrossAttnAdaGN(nn.Module):
    """
    Cross-attention style modulation with learnable style tokens.
    """

    def __init__(
        self,
        dim: int,
        style_dim: int,
        num_groups: int = 4,
        num_tokens: int = 64,
        num_heads: int = 4,
        sharpen_scale: float = 2.0,
        attn_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        groups = max(1, min(int(num_groups), int(dim)))
        while dim % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, dim, affine=False)
        self.dim = int(dim)
        self.num_tokens = max(1, int(num_tokens))
        self.num_heads = max(1, min(int(num_heads), int(dim)))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        self.sharpen_scale = max(0.1, float(sharpen_scale))
        self.attn_temperature = max(1e-3, float(attn_temperature))

        self.global_proj = nn.Linear(style_dim, dim * 2)
        nn.init.normal_(self.global_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.global_proj.bias)
        with torch.no_grad():
            self.global_proj.bias[:dim] = 1.0

        self.style_tokens_basis = nn.Parameter(torch.randn(self.num_tokens, dim) * 0.02)
        self.style_proj = nn.Linear(style_dim, dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.token_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self._coord_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}

    def _get_coord_grid(self, h_dim: int, w_dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(h_dim), int(w_dim), str(device), str(dtype))
        cached = self._coord_cache.get(key)
        if cached is not None:
            return cached
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h_dim, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w_dim, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).contiguous()
        self._coord_cache[key] = coords
        return coords

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        b, c, h_dim, w_dim = x.shape
        normalized = self.norm(x)
        scale, shift = self.global_proj(style_code).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)

        style_bias = self.style_proj(style_code).unsqueeze(1)
        style_tokens = self.style_tokens_basis.unsqueeze(0) + style_bias
        style_tokens = self.token_norm(style_tokens)

        coords = self._get_coord_grid(h_dim, w_dim, device=x.device, dtype=x.dtype).expand(b, -1, -1, -1)
        pos = coords.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, 2)
        pos_emb = self.pos_proj(pos)
        q_in = self.query_norm(normalized.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c) + pos_emb)

        q = self.q_proj(q_in).view(b, h_dim * w_dim, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(style_tokens).view(b, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(style_tokens).view(b, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        base_scale = 1.0 / math.sqrt(float(self.head_dim))
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=(base_scale * self.sharpen_scale) / self.attn_temperature,
        )
        style_content = attn_out.transpose(1, 2).reshape(b, h_dim * w_dim, c)
        style_content = self.out_proj(style_content)
        style_content = style_content + self.ffn(self.ffn_norm(style_content))
        style_content = style_content.transpose(1, 2).reshape(b, c, h_dim, w_dim)

        style_residual = shift + (style_content * self.gamma)
        # Clamp style-only residual energy to prevent shallow-layer MA spikes from
        # detonating the latent before the residual anchor can stabilize it.
        adagn = normalized * scale + torch.tanh(style_residual) * 3.0
        final_gate = gate if isinstance(gate, float) else gate.to(device=x.device, dtype=x.dtype)
        return normalized + final_gate * (adagn - normalized)


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


def _build_style_modulator(
    *,
    dim: int,
    style_dim: int,
    num_groups: int,
    attn_num_tokens: int,
    attn_num_heads: int,
    attn_sharpen_scale: float,
    attn_temperature: float,
) -> nn.Module:
    return CrossAttnAdaGN(
        dim=dim,
        style_dim=style_dim,
        num_groups=num_groups,
        num_tokens=attn_num_tokens,
        num_heads=attn_num_heads,
        sharpen_scale=attn_sharpen_scale,
        attn_temperature=attn_temperature,
    )


class ResBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        style_dim: int,
        num_groups: int = 8,
        style_attn_num_tokens: int = 16,
        style_attn_num_heads: int = 4,
        style_attn_sharpen_scale: float = 2.0,
        style_attn_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.norm1 = _build_style_modulator(
            dim=dim,
            style_dim=style_dim,
            num_groups=num_groups,
            attn_num_tokens=style_attn_num_tokens,
            attn_num_heads=style_attn_num_heads,
            attn_sharpen_scale=style_attn_sharpen_scale,
            attn_temperature=style_attn_temperature,
        )
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = _build_style_modulator(
            dim=dim,
            style_dim=style_dim,
            num_groups=num_groups,
            attn_num_tokens=style_attn_num_tokens,
            attn_num_heads=style_attn_num_heads,
            attn_sharpen_scale=style_attn_sharpen_scale,
            attn_temperature=style_attn_temperature,
        )
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        h = self.act(self.norm1(x, style_code, gate=gate))
        h = self.conv1(h)
        h = self.act(self.norm2(h, style_code, gate=gate))
        h = self.conv2(h)
        return x + h


class SpatialSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mode: str = "global_attn",
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = max(1, min(int(num_heads), self.dim))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        self.mode = _normalize_feature_block_type(mode)
        self.window_size = max(1, int(window_size))
        self.qkv = nn.Conv2d(self.dim, self.dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False)

    def _reshape_windows(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
        b, c, h_dim, w_dim = x.shape
        ws = min(self.window_size, h_dim, w_dim)
        if (h_dim % ws) != 0 or (w_dim % ws) != 0:
            return x, (b, c, h_dim, w_dim, 0)
        x = (
            x.view(b, c, h_dim // ws, ws, w_dim // ws, ws)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, ws * ws, c)
        )
        return x, (b, c, h_dim, w_dim, ws)

    def _restore_windows(self, x: torch.Tensor, meta: tuple[int, int, int, int, int]) -> torch.Tensor:
        b, c, h_dim, w_dim, ws = meta
        if ws == 0:
            return x
        return (
            x.view(b, h_dim // ws, w_dim // ws, ws, ws, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, c, h_dim, w_dim)
        )

    def forward(self, x: torch.Tensor, shift: bool = False) -> torch.Tensor:
        b, c, h_dim, w_dim = x.shape
        input_is_channels_last = x.is_contiguous(memory_format=torch.channels_last)
        shift_size = 0
        if self.mode == "window_attn" and shift:
            ws = min(self.window_size, h_dim, w_dim)
            if ws > 1 and (h_dim % ws) == 0 and (w_dim % ws) == 0:
                shift_size = ws // 2
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        if self.mode == "window_attn":
            q_tokens, meta = self._reshape_windows(q)
            k_tokens, _ = self._reshape_windows(k)
            v_tokens, _ = self._reshape_windows(v)
            if meta[-1] == 0:
                q_tokens = q.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
                k_tokens = k.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
                v_tokens = v.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
                used_windows = False
            else:
                used_windows = True
        else:
            q_tokens = q.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
            k_tokens = k.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
            v_tokens = v.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
            used_windows = False

        batch_tokens = q_tokens.shape[0]
        seq_len = q_tokens.shape[1]
        q_heads = q_tokens.view(batch_tokens, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = k_tokens.view(batch_tokens, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v_tokens.view(batch_tokens, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q_heads, k_heads, v_heads)
        out = attn.transpose(1, 2).reshape(batch_tokens, seq_len, c)

        if used_windows:
            out = self._restore_windows(out, meta)
        else:
            out = out.view(b, h_dim, w_dim, c).permute(0, 3, 1, 2)
        if shift_size > 0:
            out = torch.roll(out, shifts=(shift_size, shift_size), dims=(2, 3))
        if input_is_channels_last:
            out = out.contiguous(memory_format=torch.channels_last)
        return self.proj(out)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        style_dim: int,
        num_groups: int = 8,
        style_attn_num_tokens: int = 16,
        style_attn_num_heads: int = 4,
        style_attn_sharpen_scale: float = 2.0,
        feature_attn_num_heads: int = 4,
        style_attn_temperature: float = 0.5,
        attn_mode: str = "global_attn",
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = _build_style_modulator(
            dim=dim,
            style_dim=style_dim,
            num_groups=num_groups,
            attn_num_tokens=style_attn_num_tokens,
            attn_num_heads=style_attn_num_heads,
            attn_sharpen_scale=style_attn_sharpen_scale,
            attn_temperature=style_attn_temperature,
        )
        self.attn = SpatialSelfAttention(
            dim=dim,
            num_heads=feature_attn_num_heads,
            mode=attn_mode,
            window_size=window_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
        shift: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, style_code, gate=gate), shift=shift)
        return x


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
        # Keep the modulator near identity, but make the style path live at step 0.
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 0.05))
        self.last_attn: torch.Tensor | None = None
        self.last_k: torch.Tensor | None = None
        self.last_topology_attn: torch.Tensor | None = None
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
        k_style = self.to_k(k_dehydrated).view(b, c, -1)
        v = self.to_v(ns).view(b, c, -1).transpose(1, 2)

        temp = torch.exp(self.log_temp).clamp(1e-4, 10.0)
        scale = (c ** -0.5) / temp
        attn_logits = torch.bmm(q, k_style) * scale
        self.last_topology_attn = None
        k_debug = k_style
        if self.self_topology_gate and self.self_topology_blend > 0.0:
            k_content = self.to_k(q_dehydrated).view(b, c, -1)
            topology_logits = torch.bmm(q, k_content) * scale
            self.last_topology_attn = F.softmax(topology_logits, dim=-1)
            attn_logits = torch.lerp(attn_logits, topology_logits, self.self_topology_blend)
            k_debug = torch.lerp(k_style, k_content, self.self_topology_blend)
        attn = _route_attention_logits(
            attn_logits,
            routing_mode=self.routing_mode,
            sinkhorn_iters=self.sinkhorn_iters,
            gumbel_tau=self.gumbel_tau,
        )
        self.last_attn = attn
        self.last_k = F.normalize(k_debug, p=2, dim=1)
        painted = torch.bmm(attn, v).transpose(1, 2).view(b, c, h_dim, w_dim)

        if self.paint_only:
            return painted

        learned_gate = torch.sigmoid(self.gate_conv(nx))
        final_gate = gate if isinstance(gate, float) else gate.to(device=x.device, dtype=x.dtype)
        delta = painted * (1.0 + self.gamma) * learned_gate
        return x + final_gate * delta


def _spatial_distance_bias(
    h_dim: int,
    w_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords_y, coords_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h_dim, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w_dim, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([coords_y.reshape(-1), coords_x.reshape(-1)], dim=1)
    return torch.cdist(coords, coords, p=2)


class SpatialModulatedSelfAttn(nn.Module):
    def __init__(self, dim: int, num_groups: int = 4, temperature: float = 0.08) -> None:
        super().__init__()
        groups = _resolve_group_count(dim, num_groups)
        self.norm_x = nn.GroupNorm(groups, dim)
        self.norm_s = nn.GroupNorm(groups, dim)
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_gamma = nn.Conv2d(dim, dim, kernel_size=1)
        self.to_beta = nn.Conv2d(dim, dim, kernel_size=1)
        self.log_temp = nn.Parameter(torch.tensor([math.log(max(1e-4, float(temperature)))], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.last_attn: torch.Tensor | None = None
        self.last_k: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, style_map: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        bsz, channels, h_dim, w_dim = x.shape
        nx = self.norm_x(x)
        ns = self.norm_s(style_map)
        q = self.to_q(nx).view(bsz, channels, -1).transpose(1, 2)
        k = self.to_k(nx).view(bsz, channels, -1)
        gamma = torch.tanh(self.to_gamma(ns))
        beta = self.to_beta(ns)
        mixed = nx * (1.0 + gamma) + beta
        v = self.to_v(mixed).view(bsz, channels, -1).transpose(1, 2)
        scale = (channels ** -0.5) / torch.exp(self.log_temp).clamp(1e-4, 10.0)
        attn = F.softmax(torch.bmm(q, k) * scale, dim=-1)
        self.last_attn = attn
        self.last_k = F.normalize(k, p=2, dim=1)
        painted = torch.bmm(attn, v).transpose(1, 2).view(bsz, channels, h_dim, w_dim)
        gain = gate if isinstance(gate, float) else gate.to(device=x.device, dtype=x.dtype)
        return x + painted * (1.0 + self.gamma) * gain


class GWOTAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_groups: int = 4,
        temperature: float = 0.08,
        spatial_lambda: float = 0.25,
        sinkhorn_iters: int = 3,
    ) -> None:
        super().__init__()
        groups = _resolve_group_count(dim, num_groups)
        self.norm_x = nn.GroupNorm(groups, dim)
        self.norm_s = nn.GroupNorm(groups, dim)
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.log_temp = nn.Parameter(torch.tensor([math.log(max(1e-4, float(temperature)))], dtype=torch.float32))
        self.spatial_lambda = float(spatial_lambda)
        self.sinkhorn_iters = max(1, int(sinkhorn_iters))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.last_attn: torch.Tensor | None = None
        self.last_k: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, style_map: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        bsz, channels, h_dim, w_dim = x.shape
        nx = self.norm_x(x)
        ns = self.norm_s(style_map)
        q = self.to_q(nx).view(bsz, channels, -1).transpose(1, 2)
        k = self.to_k(ns).view(bsz, channels, -1)
        v = self.to_v(ns).view(bsz, channels, -1).transpose(1, 2)
        scale = (channels ** -0.5) / torch.exp(self.log_temp).clamp(1e-4, 10.0)
        logits = torch.bmm(q, k) * scale
        if self.spatial_lambda > 0.0:
            spatial_bias = _spatial_distance_bias(h_dim, w_dim, device=x.device, dtype=logits.dtype)
            logits = logits - spatial_bias.unsqueeze(0) * self.spatial_lambda
        attn = _sinkhorn_attention(logits, iters=self.sinkhorn_iters)
        self.last_attn = attn
        self.last_k = F.normalize(k, p=2, dim=1)
        painted = torch.bmm(attn, v).transpose(1, 2).view(bsz, channels, h_dim, w_dim)
        gain = gate if isinstance(gate, float) else gate.to(device=x.device, dtype=x.dtype)
        return x + painted * (1.0 + self.gamma) * gain


class GatedSpadeAttention(nn.Module):
    def __init__(self, dim: int, num_groups: int = 4, temperature: float = 0.08) -> None:
        super().__init__()
        groups = _resolve_group_count(dim, num_groups)
        self.base = SpatialModulatedSelfAttn(dim, num_groups=num_groups, temperature=temperature)
        self.style_proj = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.gate_proj = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.Conv2d(dim, dim // 2 if dim > 1 else 1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim // 2 if dim > 1 else 1, 1, kernel_size=1),
        )
        self.last_attn: torch.Tensor | None = None
        self.last_k: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, style_map: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        base_out = self.base(x, x, gate=1.0)
        style_delta = self.style_proj(style_map.float()).to(dtype=x.dtype)
        local_gate = torch.sigmoid(self.gate_proj(style_map.float())).to(dtype=x.dtype)
        gain = gate if isinstance(gate, float) else gate.to(device=x.device, dtype=x.dtype)
        self.last_attn = self.base.last_attn
        self.last_k = self.base.last_k
        return x + ((base_out - x) * (1.0 - local_gate) + style_delta * local_gate) * gain


class PnPSelfAttentionInject(nn.Module):
    def __init__(self, dim: int, num_groups: int = 4, temperature: float = 0.08) -> None:
        super().__init__()
        groups = _resolve_group_count(dim, num_groups)
        self.norm_x = nn.GroupNorm(groups, dim)
        self.norm_s = nn.GroupNorm(groups, dim)
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.log_temp = nn.Parameter(torch.tensor([math.log(max(1e-4, float(temperature)))], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.last_attn: torch.Tensor | None = None
        self.last_k: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, style_map: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        bsz, channels, h_dim, w_dim = x.shape
        nx = self.norm_x(x)
        ns = self.norm_s(style_map)
        q = self.to_q(nx).view(bsz, channels, -1).transpose(1, 2)
        k = self.to_k(nx).view(bsz, channels, -1)
        v = self.to_v(ns).view(bsz, channels, -1).transpose(1, 2)
        scale = (channels ** -0.5) / torch.exp(self.log_temp).clamp(1e-4, 10.0)
        attn = F.softmax(torch.bmm(q, k) * scale, dim=-1)
        self.last_attn = attn
        self.last_k = F.normalize(k, p=2, dim=1)
        painted = torch.bmm(attn, v).transpose(1, 2).view(bsz, channels, h_dim, w_dim)
        gain = gate if isinstance(gate, float) else gate.to(device=x.device, dtype=x.dtype)
        return x + painted * (1.0 + self.gamma) * gain


class StyleBlender(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_resolve_group_count(dim, num_groups), dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        blend_ratio = torch.sigmoid(self.alpha).to(device=content_feat.device, dtype=content_feat.dtype)
        blended = torch.lerp(content_feat, style_feat, blend_ratio)
        return self.conv(self.norm(blended))


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


def _build_feature_block(
    block_type: str,
    *,
    dim: int,
    style_dim: int,
    num_groups: int,
    style_attn_num_tokens: int,
    style_attn_num_heads: int,
    style_attn_sharpen_scale: float,
    feature_attn_num_heads: int,
    style_attn_temperature: float,
    window_attn_window_size: int,
) -> nn.Module:
    kind = _normalize_feature_block_type(block_type)
    if kind == "conv":
        return ResBlock(
            dim=dim,
            style_dim=style_dim,
            num_groups=num_groups,
            style_attn_num_tokens=style_attn_num_tokens,
            style_attn_num_heads=style_attn_num_heads,
            style_attn_sharpen_scale=style_attn_sharpen_scale,
            style_attn_temperature=style_attn_temperature,
        )
    return AttentionBlock(
        dim=dim,
        style_dim=style_dim,
        num_groups=num_groups,
        style_attn_num_tokens=style_attn_num_tokens,
        style_attn_num_heads=style_attn_num_heads,
        style_attn_sharpen_scale=style_attn_sharpen_scale,
        feature_attn_num_heads=feature_attn_num_heads,
        style_attn_temperature=style_attn_temperature,
        attn_mode=kind,
        window_size=window_attn_window_size,
    )


class NormFreeModulation(nn.Module):
    """
    Decoder-side style modulation without spatial normalization.
    Preserves local contrast while injecting high-frequency style controls.
    """

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.mapper = nn.Linear(style_dim, channels * 2)
        # From-scratch training benefits from a live style path on step 0.
        nn.init.normal_(self.mapper.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mapper.bias)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        params = self.mapper(style_code).view(x.shape[0], -1, 1, 1)
        gamma, beta = params.chunk(2, dim=1)
        if isinstance(gate, torch.Tensor):
            gate_t = gate.to(device=x.device, dtype=x.dtype)
        else:
            gate_t = x.new_tensor(float(gate))
        gamma = gamma * gate_t
        beta = beta * gate_t
        return x * (1.0 + gamma) + beta


class DecoderTextureBlock(nn.Module):
    def __init__(self, dim: int, style_dim: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_resolve_group_count(dim, num_groups), dim, affine=True)
        self.mapper = nn.Sequential(
            nn.Linear(style_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        nn.init.normal_(self.mapper[-1].weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.mapper[-1].bias)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        normalized = self.norm(x)
        gamma, beta = self.mapper(style_code).chunk(2, dim=-1)
        gamma = gamma.view(-1, gamma.shape[1], 1, 1).to(dtype=x.dtype)
        beta = beta.view(-1, beta.shape[1], 1, 1).to(dtype=x.dtype)

        h = normalized * (1.0 + gamma) + beta
        delta_raw = self.conv(self.act(h))
        local_mean = F.avg_pool2d(delta_raw, kernel_size=5, stride=1, padding=2)
        delta_texture = delta_raw - local_mean
        final_gate = gate if isinstance(gate, float) else gate.to(device=x.device, dtype=x.dtype)
        return x + final_gate * torch.tanh(delta_texture) * 3.0


class StyleRoutingSkip(nn.Module):
    """
    Unified skip ablation module.
    Supports 4 modes: none, naive, adaptive, normalized.
    """

    def __init__(
        self,
        channels: int,
        style_dim: int,
        mode: str = "normalized",
        content_retention_boost: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mode = str(mode).strip().lower()
        self.gate_mapper = nn.Sequential(
            nn.Linear(style_dim, self.channels),
            nn.Sigmoid(),
        )
        self.rewrite_mapper = nn.Linear(style_dim, self.channels)
        self.content_retention_boost = max(0.0, min(1.0, float(content_retention_boost)))
        # Stable init for adaptive branch.
        nn.init.zeros_(self.rewrite_mapper.weight)
        nn.init.zeros_(self.rewrite_mapper.bias)
        # Normalized mode components.
        groups = max(1, min(8, self.channels))
        while self.channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, self.channels, affine=False)
        self.style_scale = nn.Linear(style_dim, self.channels)
        self.style_shift = nn.Linear(style_dim, self.channels)
        nn.init.normal_(self.style_scale.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.style_scale.bias)
        nn.init.normal_(self.style_shift.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.style_shift.bias)

    def forward(
        self,
        skip_feat: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
        naive_gain: float = 1.0,
    ) -> torch.Tensor:
        b, c, _, _ = skip_feat.shape
        if isinstance(gate, torch.Tensor):
            gate_t = gate.to(device=skip_feat.device, dtype=skip_feat.dtype)
            if gate_t.ndim == 0:
                gate_t = gate_t.view(1, 1, 1, 1)
            elif gate_t.ndim == 1:
                gate_t = gate_t.view(-1, 1, 1, 1)
            else:
                gate_t = gate_t.view(gate_t.shape[0], 1, 1, 1)
        else:
            gate_t = skip_feat.new_tensor(float(gate)).view(1, 1, 1, 1)
        mode = self.mode
        if mode == "none":
            return skip_feat * (1.0 - gate_t)
        if mode == "naive":
            return skip_feat * (1.0 - gate_t) + (skip_feat * float(naive_gain)) * gate_t
        if mode == "adaptive":
            erase_gate = self.gate_mapper(style_code).view(b, c, 1, 1).to(dtype=skip_feat.dtype)
            rewrite_bias = self.rewrite_mapper(style_code).view(b, c, 1, 1).to(dtype=skip_feat.dtype)
            if self.content_retention_boost > 0.0:
                erase_gate = erase_gate + (1.0 - erase_gate) * self.content_retention_boost
            effective_erase = 1.0 - (1.0 - erase_gate) * gate_t
            effective_bias = rewrite_bias * gate_t
            return skip_feat * effective_erase + effective_bias
        if mode == "normalized":
            normalized_skip = self.norm(skip_feat)
            scale = self.style_scale(style_code).view(b, c, 1, 1).to(dtype=skip_feat.dtype)
            shift = self.style_shift(style_code).view(b, c, 1, 1).to(dtype=skip_feat.dtype)
            modulated_skip = normalized_skip * scale + shift
            return skip_feat * (1.0 - gate_t) + modulated_skip * gate_t
        raise ValueError(f"Unknown skip mode: {self.mode}")


@dataclass
class StyleMaps:
    map_16: torch.Tensor | None = None
    gate_16: torch.Tensor | None = None
    mask_16: torch.Tensor | None = None
    aux_16: torch.Tensor | None = None
    family: str = "legacy_factorized"
    debug: dict[str, object] = field(default_factory=dict)
