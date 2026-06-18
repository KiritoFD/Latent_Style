from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SpatialBridgeBlock620(nn.Module):
    """Time-only AdaLN plus true style-token cross-attention."""

    def __init__(self, *, dim: int, num_heads: int, style_gate_init: float = 0.05) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = max(1, min(int(num_heads), self.dim))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        self.norm = nn.GroupNorm(1, self.dim, affine=False)
        self.time_adaln = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, self.dim * 3))
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.ffn = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.Conv2d(self.dim, self.dim * 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1),
        )
        self.style_gate = nn.Parameter(torch.tensor(float(style_gate_init)))
        nn.init.zeros_(self.time_adaln[-1].weight)
        nn.init.zeros_(self.time_adaln[-1].bias)
        nn.init.zeros_(self.out_proj.bias)
        self.last_debug: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, *, time_emb: torch.Tensor, style_tokens: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        normed = self.norm(x)
        scale, shift, gate_t = self.time_adaln(time_emb).to(dtype=x.dtype).chunk(3, dim=1)
        h_time = normed * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        q_in = h_time.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q = self.q_proj(q_in).view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(style_tokens).view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(style_tokens).view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(b, h * w, c)
        attended = self.out_proj(attended).transpose(1, 2).reshape(b, c, h, w)
        style_delta = torch.tanh(self.style_gate).to(dtype=x.dtype) * attended
        time_gate = torch.sigmoid(gate_t[:, :, None, None]).to(dtype=x.dtype)
        out = x + time_gate * style_delta
        out = out + self.ffn(out)

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
        return out


def sinusoidal_time_embedding_620(t: torch.Tensor, dim: int) -> torch.Tensor:
    t = t.float().view(-1, 1)
    half = max(1, dim // 2)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
    emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]
