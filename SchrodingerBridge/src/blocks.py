"""Residual attention block used by WEAVE."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wavelet import dwt2_haar, idwt2_haar


def _make_norm(dim: int) -> nn.Module:
    """Normalization layer (RMSNorm removed — GroupNorm hardcoded)."""
    return nn.GroupNorm(1, dim, affine=False)


class StyleAdaIN(nn.Module):
    """Per-instance AdaIN with style-conditioned affine parameters.

    Math:  x_norm = (x - μ(x)) / σ(x)          (remove instance stats)
           styled = x_norm * (1 + γ(s)) + β(s)  (apply style stats)
    where γ, β are generated from style_pooled via linear projection.

    Key difference from AdaLN (time_style_adaln):
        - AdaLN uses LayerNorm (per-layer statistics, preserves content scale)
        - AdaIN uses InstanceNorm (per-instance statistics, removes content mean/std)
        - AdaIN is more aggressive: explicitly strips per-sample content statistics
          and replaces them with style-conditioned affine parameters.

    DINOv2 CLS captures global image statistics (mean color, contrast, texture scale).
    AdaIN directly modulates these statistics at the feature-map level, making it
    a natural mechanism for DINO-S improvement.
    """

    def __init__(self, dim: int, style_dim: int, init_std: float = 0.02):
        super().__init__()
        self.gamma_proj = nn.Linear(style_dim, dim)
        self.beta_proj = nn.Linear(style_dim, dim)
        # Small random init: initially weak style modulation, grows with training
        nn.init.normal_(self.gamma_proj.weight, std=init_std)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.normal_(self.beta_proj.weight, std=init_std)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, style_pooled: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        x_norm = (x - mean) / std
        gamma = self.gamma_proj(style_pooled.to(dtype=x.dtype))[:, :, None, None]
        beta = self.beta_proj(style_pooled.to(dtype=x.dtype))[:, :, None, None]
        return x_norm * (1.0 + gamma) + beta


class ResidualBlock(nn.Module):
    """AdaLN(time) → Self-Attention → Cross-Attention(style) → FFN.

    Active path (clean_base_v2): relu2 attention, tanh_gate, single k/v proj,
    no FiLM, no MoE, no skip_coarse.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        style_gate_init: float = 0.5,
        cross_attention_enabled: bool = True,
        style_shortcut_alpha: Any = 1.0,
        layer_idx: int = 0,
        num_layers: int = 4,
        attn_temperature: float = 1.0,
        gate_warmup_steps: int = 0,
        dwt_route: bool = False,
        dwt_route_train_prob: float = 0.0,
        attn_mode: str = "relu2",
        style_adaln_enabled: bool = False,
        style_adaln_nonzero_init: bool = False,
        style_adaln_init_std: float = 0.1,
        # Round 11: StyleAdaIN — per-instance normalization with style-conditioned affine
        style_adain_enabled: bool = False,
        style_adain_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.layer_idx = int(layer_idx)
        self.cross_attention_enabled = bool(cross_attention_enabled)
        self.num_layers = int(num_layers)
        self.gate_warmup_steps = max(0, int(gate_warmup_steps))
        self._current_step = 0
        self.max_entropy_queries = 256
        self.cross_attn_entropy = torch.tensor(0.0)
        self.cross_attn_guidance: torch.Tensor | None = None
        self.dwt_route = bool(dwt_route)
        self.dwt_route_train_prob = float(dwt_route_train_prob)
        self.shortcut_alpha = float(style_shortcut_alpha) if not isinstance(style_shortcut_alpha, (list, tuple)) else (
            float(style_shortcut_alpha[self.layer_idx]) if self.layer_idx < len(style_shortcut_alpha) else 1.0
        )
        self.dim = int(dim)
        self.num_heads = max(1, min(int(num_heads), self.dim))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        self.norm1 = _make_norm(self.dim)
        # 712 Phase StyleInject 方向2: Style-AdaLN
        # 默认 False: time_adaln(time_emb) -> scale, shift, gate (baseline 行为)
        # True: time_style_adaln(concat([time_emb, style_pooled])) -> scale, shift, gate
        # 零初始化保证启用时初始等价于 baseline（style 部分贡献为 0）
        self.style_adaln_enabled = bool(style_adaln_enabled)
        self.style_adaln_nonzero_init = bool(style_adaln_nonzero_init)
        self.style_adaln_init_std = float(style_adaln_init_std)
        if self.style_adaln_enabled:
            self.time_style_adaln = nn.Sequential(
                nn.SiLU(), nn.Linear(self.dim * 2, self.dim * 3)
            )
            if self.style_adaln_nonzero_init:
                # 非零初始化: 强制 style 通路从训练初期就激活, 避免 style_pooled 梯度永远为零
                nn.init.normal_(self.time_style_adaln[-1].weight, std=self.style_adaln_init_std)
                nn.init.normal_(self.time_style_adaln[-1].bias, std=self.style_adaln_init_std)
            else:
                nn.init.zeros_(self.time_style_adaln[-1].weight)
                nn.init.zeros_(self.time_style_adaln[-1].bias)
        else:
            self.time_adaln = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, self.dim * 3))
        # Round 11: StyleAdaIN — per-instance normalization with style-conditioned affine
        self.style_adain_enabled = bool(style_adain_enabled)
        if self.style_adain_enabled:
            self.style_adain = StyleAdaIN(self.dim, self.dim, init_std=float(style_adain_init_std))
        else:
            self.style_adain = None
        # Self-attention: content Q/K/V
        self.sa_qkv = nn.Linear(self.dim, self.dim * 3)
        self.sa_out = nn.Linear(self.dim, self.dim)
        # Cross-attention: content Q, style K/V (single projection, no MoE)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm2 = _make_norm(self.dim)
        self.ffn = nn.Sequential(
            _make_norm(self.dim),
            nn.Conv2d(self.dim, self.dim * 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.dim * 4, self.dim, kernel_size=1),
        )
        self.style_gate = nn.Parameter(torch.tensor(float(style_gate_init)))
        self._gate_init = float(style_gate_init)
        self.attn_mode = str(attn_mode).lower().strip()  # default "relu2" (629 D19-D22 最优); "softmax" 用 flash attention 适配像素空间大分辨率
        self.attn_temperature = float(attn_temperature)
        # Zero-init the AdaLN output projection (time_adaln or time_style_adaln)
        # 注意: style_adaln_nonzero_init=True 时已在上面设置非零初始化, 这里不覆盖
        if not (self.style_adaln_enabled and self.style_adaln_nonzero_init):
            _adaln_mod = self.time_style_adaln if self.style_adaln_enabled else self.time_adaln
            nn.init.zeros_(_adaln_mod[-1].weight)
            nn.init.zeros_(_adaln_mod[-1].bias)
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

    def _compute_use_dwt(self) -> bool:
        """Decide whether to use DWT-routed cross-attention for this forward call.

        - Inference: always use DWT route when enabled (LL bypass protects content).
        - Training with dwt_route_train_prob > 0: stochastic route (T11 SOTA config),
          so q_proj / style_mem jointly learn DWT coefficients and full style.
        - Training with dwt_route_train_prob == 0: deterministic route when enabled
          (4J.1 original behavior).
        """
        if not self.dwt_route:
            return False
        if not self.training:
            return True
        if self.dwt_route_train_prob > 0.0:
            return torch.rand(1).item() < self.dwt_route_train_prob
        return True

    def forward(
        self,
        x: torch.Tensor,
        *,
        time_emb: torch.Tensor,
        style_tokens: torch.Tensor,
        style_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape

        # --- Self-attention with AdaLN(time) or AdaLN(time+style) ---
        normed1 = self.norm1(x)
        if self.style_adaln_enabled and style_pooled is not None:
            # 方向2: concat [time_emb, style_pooled] -> scale, shift, gate
            ts = torch.cat([time_emb, style_pooled], dim=-1).to(dtype=x.dtype)
            scale, shift, gate_t = self.time_style_adaln(ts).chunk(3, dim=1)
        else:
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

        if not self.cross_attention_enabled:
            zero = x.new_tensor(0.0)
            self.cross_attn_entropy = zero
            self.cross_attn_guidance = torch.zeros_like(x[:, :1])
            x = x + self.ffn(self.norm2(x))
            self.last_debug = {
                "style_gate_value": zero,
                "cross_attn_entropy": zero,
                "actual_attn_entropy": zero,
                "gate_mean": zero,
                "gate_std": zero,
                "cross_attn_delta_abs": zero,
                "cross_attn_token_count": zero,
                "sa_input_std": h_time.detach().float().std(),
                "sa_output_std": sa_out.detach().float().std(),
                "ca_input_std": zero,
                "ca_output_std": zero,
            }
            return x

        # --- Cross-attention (content × style) ---
        # 630 Phase 4J.1: DWT-Routed Cross-Attention (方案 B)
        # 理论: 对特征图做 Haar DWT, LL bypass (保结构), 仅高频(LH/HL/HH) query style_mem
        # 解放 style_mem: 100% 容量表达笔触/色彩, 不再被迫学"维持结构"
        ca_is_dwt = False
        use_dwt = self._compute_use_dwt()
        if use_dwt and h >= 2 and w >= 2 and (h % 2 == 0) and (w % 2 == 0):
            x_f = x.float()
            ll_f, lh_f, hl_f, hh_f = dwt2_haar(x_f)
            hf_h, hf_w = ll_f.shape[-2], ll_f.shape[-1]
            lh_tokens = lh_f.permute(0, 2, 3, 1).reshape(b, hf_h * hf_w, c)
            hl_tokens = hl_f.permute(0, 2, 3, 1).reshape(b, hf_h * hf_w, c)
            hh_tokens = hh_f.permute(0, 2, 3, 1).reshape(b, hf_h * hf_w, c)
            ca_in = torch.cat([lh_tokens, hl_tokens, hh_tokens], dim=1)
            ca_h, ca_w = 3 * hf_h, hf_w
            ca_is_dwt = True
        else:
            ca_in = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            ca_h, ca_w = h, w
        q = self.q_proj(ca_in).view(b, ca_in.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k_tokens = self.k_proj(style_tokens)
        v_tokens = self.v_proj(style_tokens)
        k = k_tokens.view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v_tokens.view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        attn_entropy, pixel_entropy = self._attention_stats(q, k, h=ca_h, w=ca_w, dtype=x.dtype)

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

        attended = attended.transpose(1, 2).reshape(b, ca_in.shape[1], c)
        ca_input_std = ca_in.detach().float().std()
        ca_output_std = attended.detach().float().std()
        attended = self.out_proj(attended)
        if ca_is_dwt:
            attended = attended.float()
            n_hf = hf_h * hf_w
            lh_out = attended[:, :n_hf, :].permute(0, 2, 1).reshape(b, c, hf_h, hf_w)
            hl_out = attended[:, n_hf:2*n_hf, :].permute(0, 2, 1).reshape(b, c, hf_h, hf_w)
            hh_out = attended[:, 2*n_hf:, :].permute(0, 2, 1).reshape(b, c, hf_h, hf_w)
            attended_2d = idwt2_haar(ll_f, lh_out, hl_out, hh_out).to(dtype=x.dtype)
        else:
            attended_2d = attended.transpose(1, 2).reshape(b, c, h, w)
        gate_val = self._effective_gate_value()
        style_delta = gate_val.to(dtype=x.dtype) * attended_2d
        self.pixel_entropy = pixel_entropy
        self.cross_attn_entropy = attn_entropy
        self.cross_attn_guidance = style_delta.detach().float().abs().mean(dim=1, keepdim=True).to(dtype=x.dtype)

        # Apply shortcut alpha (float only — learnable variant removed)
        alpha = self.shortcut_alpha
        x = alpha * x + style_delta

        # --- FFN ---
        x = x + self.ffn(self.norm2(x))

        # Round 11: StyleAdaIN — per-instance normalization with style-conditioned affine
        # 剥除 per-sample 内容统计 (mean/std), 替换为 style 调制
        if self.style_adain is not None and style_pooled is not None:
            x = self.style_adain(x, style_pooled)

        # --- Debug ---
        self.last_debug = {
            "style_gate_value": torch.tanh(self.style_gate).detach().abs(),
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


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    t = t.float().view(-1, 1)
    half = max(1, dim // 2)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
    emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]
