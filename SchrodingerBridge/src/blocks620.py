from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SpatialBridgeBlock620(nn.Module):
    """AdaLN(time) → Self-Attention → Cross-Attention(style) → FFN."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        style_gate_init: float = 0.05,
        style_moe_enabled: bool = False,
        style_moe_num_experts: int = 4,
        style_moe_router_hidden_dim: int = 128,
        style_kv_moe_content_routed: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = max(1, min(int(num_heads), self.dim))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        self.style_moe_enabled = bool(style_moe_enabled)
        self.style_moe_num_experts = max(1, int(style_moe_num_experts))
        self.style_kv_moe_content_routed = bool(style_kv_moe_content_routed)
        self.norm1 = nn.GroupNorm(1, self.dim, affine=False)
        self.time_adaln = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, self.dim * 3))
        # Self-attention: content Q/K/V
        self.sa_qkv = nn.Linear(self.dim, self.dim * 3)
        self.sa_out = nn.Linear(self.dim, self.dim)
        # Cross-attention: content Q, style K/V
        self.q_proj = nn.Linear(self.dim, self.dim)
        if self.style_moe_enabled:
            router_hidden = max(1, int(style_moe_router_hidden_dim))
            router_in_dim = self.dim * 2 if self.style_kv_moe_content_routed else self.dim
            self.style_moe_router = nn.Sequential(
                nn.LayerNorm(router_in_dim),
                nn.Linear(router_in_dim, router_hidden),
                nn.SiLU(),
                nn.Linear(router_hidden, self.style_moe_num_experts),
            )
            self.k_proj_experts = nn.ModuleList(nn.Linear(self.dim, self.dim) for _ in range(self.style_moe_num_experts))
            self.v_proj_experts = nn.ModuleList(nn.Linear(self.dim, self.dim) for _ in range(self.style_moe_num_experts))
        else:
            self.k_proj = nn.Linear(self.dim, self.dim)
            self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm2 = nn.GroupNorm(1, self.dim, affine=False)
        self.ffn = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.Conv2d(self.dim, self.dim * 4, kernel_size=1),  # 4x expansion (was 2x)
            nn.SiLU(),
            nn.Conv2d(self.dim * 4, self.dim, kernel_size=1),
        )
        self.style_gate = nn.Parameter(torch.tensor(float(style_gate_init)))
        nn.init.zeros_(self.time_adaln[-1].weight)
        nn.init.zeros_(self.time_adaln[-1].bias)
        nn.init.zeros_(self.sa_out.bias)
        nn.init.zeros_(self.out_proj.bias)
        self.last_debug: dict[str, torch.Tensor] = {}

    def _style_kv(self, style_tokens: torch.Tensor, *, content_features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.style_moe_enabled:
            return self.k_proj(style_tokens), self.v_proj(style_tokens), None
        style_summary = style_tokens.float().mean(dim=1).to(dtype=style_tokens.dtype)
        if self.style_kv_moe_content_routed and content_features is not None:
            content_summary = content_features.float().mean(dim=[2, 3]).to(dtype=style_tokens.dtype)
            router_in = torch.cat([content_summary, style_summary], dim=-1)
        else:
            router_in = style_summary
        router_probs = torch.softmax(self.style_moe_router(router_in).float(), dim=-1).to(dtype=style_tokens.dtype)
        k_experts = torch.stack([proj(style_tokens) for proj in self.k_proj_experts], dim=1)
        v_experts = torch.stack([proj(style_tokens) for proj in self.v_proj_experts], dim=1)
        weights = router_probs[:, :, None, None]
        k = (k_experts * weights).sum(dim=1)
        v = (v_experts * weights).sum(dim=1)
        return k, v, router_probs

    def forward(self, x: torch.Tensor, *, time_emb: torch.Tensor, style_tokens: torch.Tensor) -> torch.Tensor:
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
        x = x + time_gate * sa_out.transpose(1, 2).reshape(b, c, h, w)

        # --- Cross-attention (content × style) ---
        ca_in = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q = self.q_proj(ca_in).view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k_tokens, v_tokens, router_probs = self._style_kv(style_tokens, content_features=x)
        k = k_tokens.view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v_tokens.view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(b, h * w, c)
        attended = self.out_proj(attended)
        style_delta = torch.tanh(self.style_gate).to(dtype=x.dtype) * attended.transpose(1, 2).reshape(b, c, h, w)
        x = x + style_delta

        # --- FFN ---
        x = x + self.ffn(self.norm2(x))

        # --- Debug ---
        with torch.no_grad():
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(float(self.head_dim))
            attn = torch.softmax(logits, dim=-1)
            entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self.last_debug = {
            "style_gate_value": torch.tanh(self.style_gate.detach()).abs(),
            "cross_attn_entropy": entropy.detach(),
            "cross_attn_delta_abs": style_delta.detach().float().abs().mean(),
            "cross_attn_token_count": torch.tensor(float(style_tokens.shape[1]), device=x.device),
        }
        if router_probs is not None:
            with torch.no_grad():
                probs = router_probs.detach().float()
                router_entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
                router_max = probs.max(dim=-1).values.mean()
            self.last_debug["style_moe_router_entropy"] = router_entropy
            self.last_debug["style_moe_router_max_prob"] = router_max
        return x


def sinusoidal_time_embedding_620(t: torch.Tensor, dim: int) -> torch.Tensor:
    t = t.float().view(-1, 1)
    half = max(1, dim // 2)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
    emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]
