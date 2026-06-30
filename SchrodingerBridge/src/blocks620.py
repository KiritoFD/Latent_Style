"""Spatial Bridge Block 620 — cleaned active path only.

628/629 cleanup: removed dead attn_modes (gated/gated_raw/style_select/sparsemax),
FiLM modulation, style MoE, content_dino query, learnable shortcut, skip_coarse,
top-k truncation, style_bias. Kept RMSNorm (E4+ uses it) and softmax/relu2 modes.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Normalization — preserves mean (unlike GroupNorm).

    GroupNorm normalizes both mean→0 and variance→1, which destroys
    style statistics (brightness = mean, contrast = std).
    RMSNorm only normalizes by RMS (≈std), keeping the mean intact.
    This is critical for style transfer where color/brightness carry style identity.
    """
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] or [B, L, C]
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = (x.float() / rms).to(x.dtype)
        return x_norm * self.weight[None, ...] if x.dim() == 3 else x_norm * self.weight[None, :, None, None]


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    """Factory: create normalization layer."""
    if norm_type == "rms_norm":
        return RMSNorm(dim)
    else:
        return nn.GroupNorm(1, dim, affine=False)


class SpatialBridgeBlock620(nn.Module):
    """AdaLN(time) → Self-Attention → Cross-Attention(style) → FFN.

    Active path (clean_base_v2): relu2 attention, tanh_gate, single k/v proj,
    no FiLM, no MoE, no skip_coarse. Constructor keeps legacy params for
    backward compatibility but ignores dead ones.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        style_gate_init: float = 0.5,
        style_gate_mode: str = "tanh_gate",
        style_moe_enabled: bool = False,
        style_moe_num_experts: int = 4,
        style_moe_router_hidden_dim: int = 128,
        style_kv_moe_content_routed: bool = False,
        style_shortcut_alpha: Any = 1.0,
        style_query_source: str = "concat",
        style_cross_attn_skip_coarse: bool = False,
        style_attn_topk: int = 0,
        layer_idx: int = 0,
        num_layers: int = 4,
        dino_dim: int = 384,
        film_enabled: bool = False,
        film_init_std: float = 0.02,
        attn_mode: str = "softmax",
        attn_temperature: float = 1.0,
        gate_warmup_steps: int = 0,
        norm_type: str = "group_norm",
    ) -> None:
        super().__init__()
        # Ignore dead params (kept for call-site compatibility with spectral_bridge620.py)
        del style_moe_enabled, style_moe_num_experts, style_moe_router_hidden_dim
        del style_kv_moe_content_routed, style_query_source, style_cross_attn_skip_coarse
        del style_attn_topk, dino_dim, film_enabled, film_init_std

        self.layer_idx = int(layer_idx)
        self.num_layers = int(num_layers)
        self.gate_warmup_steps = max(0, int(gate_warmup_steps))
        self._current_step = 0
        self.max_entropy_queries = 256
        self.cross_attn_entropy = torch.tensor(0.0)

        self.shortcut_alpha = float(style_shortcut_alpha) if not isinstance(style_shortcut_alpha, (list, tuple)) else (
            float(style_shortcut_alpha[self.layer_idx]) if self.layer_idx < len(style_shortcut_alpha) else 1.0
        )
        self.dim = int(dim)
        self.num_heads = max(1, min(int(num_heads), self.dim))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        self.norm_type = norm_type
        self.norm1 = _make_norm(norm_type, self.dim)
        self.time_adaln = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, self.dim * 3))
        # Self-attention: content Q/K/V
        self.sa_qkv = nn.Linear(self.dim, self.dim * 3)
        self.sa_out = nn.Linear(self.dim, self.dim)
        # Cross-attention: content Q, style K/V (single projection, no MoE)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm2 = _make_norm(norm_type, self.dim)
        self.ffn = nn.Sequential(
            _make_norm(norm_type, self.dim),
            nn.Conv2d(self.dim, self.dim * 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.dim * 4, self.dim, kernel_size=1),
        )
        self.style_gate = nn.Parameter(torch.tensor(float(style_gate_init)))
        self.style_gate_mode = str(style_gate_mode).strip().lower()
        self._gate_init = float(style_gate_init)
        self.attn_mode = str(attn_mode).strip().lower()
        self.attn_temperature = float(attn_temperature)
        nn.init.zeros_(self.time_adaln[-1].weight)
        nn.init.zeros_(self.time_adaln[-1].bias)
        nn.init.zeros_(self.sa_out.bias)
        nn.init.zeros_(self.out_proj.bias)
        self.last_debug: dict[str, torch.Tensor] = {}

    def set_step(self, step: int) -> None:
        """Update the current training step for gate warmup scheduling."""
        self._current_step = int(step)

    def _effective_gate_value(self) -> torch.Tensor:
        """Compute effective gate value with warmup schedule."""
        raw = torch.tanh(self.style_gate)
        if self.gate_warmup_steps <= 0 or not self.training:
            return raw
        warmup_factor = min(1.0, self._current_step / max(1, self.gate_warmup_steps))
        return raw * warmup_factor

    def forward(
        self,
        x: torch.Tensor,
        *,
        time_emb: torch.Tensor,
        style_tokens: torch.Tensor,
        style_global: torch.Tensor | None = None,
        content_dino_patches: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del style_global, content_dino_patches  # unused (FiLM/content_dino removed)
        b, c, h, w = x.shape

        # --- Self-attention with AdaLN(time) ---
        normed1 = self.norm1(x)
        scale, shift, gate_t = self.time_adaln(time_emb).to(dtype=x.dtype).chunk(3, dim=1)
        h_time = normed1 * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        sa_in = h_time.permute(0, 2, 3, 1).reshape(b, h * w, c)
        sa_qkv = self.sa_qkv(sa_in)
        sa_q, sa_k, sa_v = sa_qkv.chunk(3, dim=-1)
        sa_q = sa_q.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        sa_k = sa_k.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        sa_v = sa_v.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        sa_out = F.scaled_dot_product_attention(sa_q, sa_k, sa_v, dropout_p=0.0, is_causal=False)
        sa_out = sa_out.transpose(1, 2).reshape(b, h * w, c)
        sa_out = self.sa_out(sa_out)
        time_gate = torch.sigmoid(gate_t[:, :, None, None]).to(dtype=x.dtype)
        sa_delta = time_gate * sa_out.transpose(1, 2).reshape(b, c, h, w)
        x = x + sa_delta

        # --- Cross-attention (content × style) ---
        ca_in = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q = self.q_proj(ca_in).view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k_tokens = self.k_proj(style_tokens)
        v_tokens = self.v_proj(style_tokens)
        k = k_tokens.view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v_tokens.view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        attn_entropy, pixel_entropy = self._attention_stats(q, k, h=h, w=w, dtype=x.dtype)

        # --- Attention computation (only relu2 + softmax kept) ---
        scale_attn = 1.0 / math.sqrt(float(self.head_dim))
        temp = max(self.attn_temperature, 1e-4)
        if self.attn_mode == "relu2":
            # ReLU^2 attention: sparse, magnitude-preserving (clean_base_v2 active mode)
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale_attn / temp
            gates = torch.relu(logits) ** 2
            attended = torch.matmul(gates, v)
            with torch.no_grad():
                gate_mean = gates.detach().float().mean()
                gate_std = gates.detach().float().std()
                active_ratio = (gates > 0.0).float().mean()
                actual_attn_entropy = -torch.log(active_ratio.clamp_min(1e-8))
        else:
            # Standard softmax attention (clean_base fallback mode)
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            gate_mean = torch.tensor(0.0, device=x.device)
            gate_std = torch.tensor(0.0, device=x.device)
            actual_attn_entropy = attn_entropy

        attended = attended.transpose(1, 2).reshape(b, h * w, c)
        ca_input_std = ca_in.detach().float().std()
        ca_output_std = attended.detach().float().std()
        attended = self.out_proj(attended)
        attended_2d = attended.transpose(1, 2).reshape(b, c, h, w)
        style_delta = self._effective_gate_value().to(dtype=x.dtype) * attended_2d
        self.pixel_entropy = pixel_entropy
        self.cross_attn_entropy = attn_entropy

        # Apply shortcut alpha (float only — learnable variant removed)
        alpha = self.shortcut_alpha
        x = alpha * x + style_delta

        # --- FFN ---
        x = x + self.ffn(self.norm2(x))

        # --- Debug ---
        self.last_debug = {
            "style_gate_value": self._effective_gate_value().detach().abs(),
            "cross_attn_entropy": attn_entropy.detach(),
            "actual_attn_entropy": actual_attn_entropy.detach() if isinstance(actual_attn_entropy, torch.Tensor) else torch.tensor(0.0, device=x.device),
            "gate_mean": gate_mean.detach() if isinstance(gate_mean, torch.Tensor) else torch.tensor(0.0, device=x.device),
            "gate_std": gate_std.detach() if isinstance(gate_std, torch.Tensor) else torch.tensor(0.0, device=x.device),
            "cross_attn_delta_abs": style_delta.detach().float().abs().mean(),
            "cross_attn_token_count": torch.tensor(float(style_tokens.shape[1]), device=x.device),
            "sa_input_std": h_time.detach().float().std(),
            "sa_output_std": sa_out.detach().float().std(),
            "ca_input_std": ca_input_std,
            "ca_output_std": ca_output_std,
        }
        return x

    def _attention_stats(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        h: int,
        w: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            q_stat = q.detach().float()
            k_stat = k.detach().float()
            total_queries = int(q_stat.shape[-2])
            if total_queries <= self.max_entropy_queries:
                q_small = q_stat
                sample_idx = None
            else:
                sample_idx = torch.linspace(
                    0,
                    total_queries - 1,
                    steps=self.max_entropy_queries,
                    device=q_stat.device,
                ).round().long()
                q_small = q_stat.index_select(dim=2, index=sample_idx)
            logits_small = torch.matmul(q_small, k_stat.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
            attn_small = torch.softmax(logits_small, dim=-1)
            entropy_small = -(attn_small * attn_small.clamp_min(1e-8).log()).sum(dim=-1)
            attn_entropy = entropy_small.mean().to(device=q.device, dtype=dtype)
            pixel_entropy = torch.zeros(q.shape[0], 1, h * w, device=q.device, dtype=dtype)
            if sample_idx is None:
                per_query = entropy_small.mean(dim=1, keepdim=True).to(device=q.device, dtype=dtype)
                pixel_entropy = per_query
            else:
                per_query_small = entropy_small.mean(dim=1).to(device=q.device, dtype=dtype)
                pixel_entropy.scatter_(dim=2, index=sample_idx.view(1, 1, -1).expand(q.shape[0], 1, -1), src=per_query_small.unsqueeze(1))
                pixel_entropy = F.interpolate(pixel_entropy.view(q.shape[0], 1, 1, h * w), size=(1, h * w), mode="nearest")
            return attn_entropy, pixel_entropy.view(q.shape[0], 1, h, w)


def sinusoidal_time_embedding_620(t: torch.Tensor, dim: int) -> torch.Tensor:
    t = t.float().view(-1, 1)
    half = max(1, dim // 2)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
    emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]
